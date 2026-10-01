"""Final V2 classifier analysis on the seed-42 corpus, using the FROZEN artefact.

Inputs (never modified):
  models/risk_model_v2.pkl, models/metrics_v2.json  - the frozen V2 model and its
      validation-selected threshold (0.1123)
  evaluation_v2/metrics.json, evaluation_v2/csv/test_predictions.csv - the
      previously reported V2 test results, used only for comparison
  backend.ml.dataset_v2.load_or_generate_v2(seed=42) - the corpus (cache is
      fingerprint-checked against generator, corpus and coercion-weight hashes)

Split: scripts/train_model_v2.splits (random_state 42) - identical to the split
the model was frozen on. The threshold is read from the frozen artefact; this
script never searches for a threshold on the test split. The validation sweep
re-derives the threshold only to confirm the stored one.

Writes everything under results/final_v2/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, matthews_corrcoef, precision_recall_curve,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.ml.dataset_v2 import FEATURE_GROUPS, FEATURES_V2, load_or_generate_v2  # noqa: E402
from backend.ml import dataset as v1gen                                             # noqa: E402
from train_model_v2 import recall_at_fpr, splits                                    # noqa: E402

from families import FAMILY_OF, FAMILY_ORDER                                        # noqa: E402
from backend.app.services.coercion import FEATURE_ORDER as C_FEATURE_ORDER          # noqa: E402
import style                                                                        # noqa: E402
from style import FRAUD, LEGIT, INK2, MUTED, AQUA                                   # noqa: E402

R = ROOT / "results" / "final_v2"
SCEN = ["social_engineering", "mule_collection", "account_takeover", "card_testing"]
SCEN_LABEL = {"social_engineering": "Social engineering", "mule_collection": "Mule collection",
              "account_takeover": "Account takeover", "card_testing": "Card testing"}


def wjson(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def md_table(df: pd.DataFrame, path: Path, title: str, note: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = f"**{title}**\n\n" + df.to_markdown(index=False) + (f"\n\n{note}\n" if note else "\n")
    path.write_text(txt, encoding="utf-8")
    df.to_csv(path.with_suffix(".csv"), index=False)


def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn); spec = tn / (tn + fp)
    base = float(y.mean()); brier = float(brier_score_loss(y, p)); ref = float(np.mean((y - base) ** 2))
    return {
        "n": int(len(y)), "positives": int(y.sum()), "negatives": int((y == 0).sum()), "prevalence": base,
        "threshold": float(thr), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc_average_precision": float(average_precision_score(y, p)),
        "accuracy": float((tp + tn) / len(y)), "precision": float(prec), "recall": float(rec),
        "f1": float(2 * prec * rec / (prec + rec)), "specificity": float(spec), "fpr": float(fp / (fp + tn)),
        "fnr": float(fn / (fn + tp)), "npv": float(tn / (tn + fn)), "balanced_accuracy": float((rec + spec) / 2),
        "mcc": float(matthews_corrcoef(y, pred)), "brier": brier, "brier_reference_base_rate": ref,
        "brier_skill_score": float(1 - brier / ref), "alerts": int(tp + fp), "alert_rate": float((tp + fp) / len(y)),
    }


def count_realised_payers(v2frame) -> int:
    """Distinct simulated payers that actually received a transaction (seed 42).

    The V2 frame stores no payer identifier, so the V1 generator is re-run with
    its per-row generators wrapped to record which payer object each row was
    drawn for. The wrappers call the originals unchanged (same RNG stream), and
    the regenerated rows are asserted identical to the cached V2 frame.
    """
    import backend.ml.dataset as g
    seen = []
    orig_legit, orig_sc = g._legit, dict(g.SCENARIOS)
    g._legit = lambda rng, user: (seen.append(id(user)), orig_legit(rng, user))[1]
    for k, f in orig_sc.items():
        g.SCENARIOS[k] = (lambda f_: (lambda rng, user: (seen.append(id(user)), f_(rng, user))[1]))(f)
    try:
        v1 = g.generate(120_000, 0.012, 2_000, 42)
    finally:
        g._legit = orig_legit; g.SCENARIOS.clear(); g.SCENARIOS.update(orig_sc)
    cols = list(g.FEATURES) + ["is_fraud", "scenario"]
    assert v1[cols].reset_index(drop=True).equals(v2frame[cols].reset_index(drop=True))
    assert len(seen) == len(v1)
    return len(set(seen))


def ece(y, p, n_bins=10, strategy="uniform"):
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    rows, total = [], 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & ((p <= hi) if i == len(edges) - 2 else (p < hi))
        if not m.any():
            continue
        gap = abs(p[m].mean() - y[m].mean())
        total += m.sum() / len(y) * gap
        rows.append({"bin_lower": float(lo), "bin_upper": float(hi), "n": int(m.sum()),
                     "fraud_in_bin": int(y[m].sum()), "mean_predicted": float(p[m].mean()),
                     "observed_fraud_rate": float(y[m].mean()), "abs_gap": float(gap)})
    return float(total), pd.DataFrame(rows)


def main() -> int:
    for sub in ("dataset", "classifier", "confusion_matrix", "roc_pr", "calibration", "threshold", "feature_importance",
                "scenario_analysis", "tables", "figures"):
        (R / sub).mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / "models/metrics_v2.json").read_text())
    reported = json.loads((ROOT / "evaluation_v2/metrics.json").read_text())
    model = joblib.load(ROOT / "models/risk_model_v2.pkl")
    cols = frozen["features"]; assert cols == list(FEATURES_V2) and len(cols) == 47
    thr = float(frozen["validation_results"]["threshold"])

    df, fp_ = load_or_generate_v2(seed=42)
    assert fp_ == frozen["dataset_fingerprint"], "dataset fingerprint differs from the frozen one"
    y, i_tr, i_va, i_te = splits(df)
    scen = df["scenario"].to_numpy()
    p_va = model.predict_proba(df[cols].iloc[i_va])[:, 1]
    p_te = model.predict_proba(df[cols].iloc[i_te])[:, 1]
    y_va, y_te = y[i_va], y[i_te]

    # ── 0. reproduction check ─────────────────────────────────────────────
    old = pd.read_csv(ROOT / "evaluation_v2/csv/test_predictions.csv")
    rep = {"reported_file_rows": int(len(old))}
    mine = pd.Series(p_te, index=i_te)
    rep["same_test_row_indices"] = bool(set(old.dataset_row_index) == set(i_te))
    rep["max_abs_prob_diff_vs_reported_predictions_per_row"] = float(
        np.max(np.abs(old.set_index("dataset_row_index").predicted_probability.to_numpy()
                      - mine.reindex(old.dataset_row_index).to_numpy())))
    te = binary_metrics(y_te, p_te, thr); va = binary_metrics(y_va, p_va, thr)
    rt = reported["test_metrics"]
    rep["comparison"] = {k: {"reported": rt[k2], "reproduced": te[k]} for k, k2 in (
        ("pr_auc_average_precision", "average_precision"), ("roc_auc", "roc_auc"), ("recall", "recall"),
        ("precision", "precision"), ("f1", "f1"), ("fpr", "false_positive_rate"), ("brier", "brier"),
        ("brier_skill_score", "brier_skill_score"))}
    rep["confusion_reported"] = rt["confusion_matrix"]
    rep["confusion_reproduced"] = {k: te[k] for k in ("tn", "fp", "fn", "tp")}
    rep["identical_confusion"] = rep["confusion_reported"] == rep["confusion_reproduced"]
    rep["max_abs_metric_diff"] = max(abs(v["reported"] - v["reproduced"]) for v in rep["comparison"].values())
    rep["threshold_frozen"] = thr
    r_va, thr_va = recall_at_fpr(y_va, p_va, 0.01)
    rep["threshold_rederived_on_validation"] = float(thr_va)
    rep["threshold_rederived_rounded"] = round(float(thr_va), 4)
    rep["threshold_matches"] = round(float(thr_va), 4) == thr
    fv = frozen["validation_results"]["confusion_matrix"]
    full_thr = float(frozen["validation_results"]["threshold_full_precision"])
    rep["validation_note"] = {
        "frozen_validation_confusion_at_full_precision_threshold": fv,
        "validation_confusion_at_stored_threshold_0.1123": {k: va[k] for k in ("tn", "fp", "fn", "tp")},
        "validation_confusion_at_full_precision_threshold_recomputed": {k: binary_metrics(y_va, p_va, full_thr)[k] for k in ("tn", "fp", "fn", "tp")},
        "test_confusion_at_full_precision_threshold": {k: binary_metrics(y_te, p_te, full_thr)[k] for k in ("tn", "fp", "fn", "tp")},
        "rows_between_thresholds": {"validation": int(((p_va >= full_thr) & (p_va < thr)).sum()),
                                    "test": int(((p_te >= full_thr) & (p_te < thr)).sum())},
        "reading": "models/metrics_v2.json records validation results at the full-precision threshold; the stored and served threshold "
                   "is the 4-d.p. value 0.1123, at which the reported test results were computed. One validation fraud row and one test "
                   "legitimate row lie between the two values."}
    wjson(R / "classifier/reproduction_check.json", rep)

    # ── 1. dataset statistics ─────────────────────────────────────────────
    rows = []
    for name, idx in (("All", np.arange(len(df))), ("Train", i_tr), ("Validation", i_va), ("Test", i_te)):
        yy = y[idx]
        rows.append({"Split": name, "Total": int(len(idx)), "Legitimate": int((yy == 0).sum()),
                     "Fraud": int(yy.sum()), "Fraud %": f"{100 * yy.mean():.4f}"})
    t1 = pd.DataFrame(rows)
    payers = count_realised_payers(df)
    md_table(t1, R / "tables/table01_dataset_splits.md", "Table 1 — Dataset and split statistics (V2 corpus, seed 42)",
             f"Simulated payers: {payers:,} distinct payers realised (generator setting 2,000; counted by instrumenting the generator, whose output was "
             "checked to be identical to the cached V2 frame — the V2 frame stores no payer identifier). Synthetic data.")
    mix = dict(v1gen.SCENARIO_MIX)
    fr = df[df.is_fraud == 1]
    rows = []
    for s in SCEN:
        rows.append({"Scenario": SCEN_LABEL[s], "Generator share": f"{100*mix[s]:.0f}%",
                     "Realised count (all)": int((fr.scenario == s).sum()),
                     "Realised share (all fraud)": f"{100*(fr.scenario == s).mean():.2f}%",
                     "Test count": int(((scen[i_te] == s) & (y_te == 1)).sum()),
                     "Test share": f"{100*((scen[i_te] == s) & (y_te == 1)).sum() / y_te.sum():.2f}%"})
    t2 = pd.DataFrame(rows)
    md_table(t2, R / "tables/table02_scenarios.md", "Table 2 — Fraud scenario composition (synthetic scenario categories)",
             "Generator shares are simulator settings (backend/ml/dataset.SCENARIO_MIX), not estimates of real UPI fraud prevalence.")
    wjson(R / "dataset/dataset_statistics.json", {"splits": t1.to_dict("records"), "scenarios": t2.to_dict("records"),
          "simulated_payers": payers, "fingerprint": fp_,
          "payee_history_hidden_rows": int((df["payee_history_available"] == 0).sum()),
          "payee_history_hidden_share": float((df["payee_history_available"] == 0).mean()),
          "fraud_rate_hidden": float(df.loc[df.payee_history_available == 0, "is_fraud"].mean()),
          "fraud_rate_visible": float(df.loc[df.payee_history_available == 1, "is_fraud"].mean()),
          "message_available_share": float(df["message_available"].mean()),
          "intent_stated_share": float(df["intent_stated"].mean()),
          "graph_available_share": float(df["graph_available"].mean())})

    # ── 3. feature inventory ──────────────────────────────────────────────
    doc = pd.read_csv(ROOT / "evaluation_v2/feature_documentation.csv")
    groups = {c: g for g, cs in FEATURE_GROUPS.items() for c in cs}
    side = {"Payer behaviour": "payer-side", "Amount context": "contextual (payment)", "Payee history": "payee-side",
            "Relationship": "relational", "Graph": "graph", "Address and QR": "QR/address",
            "Message pressure": "text", "Link safety": "link", "Stated intent": "contextual (payer-stated)"}
    meaning = {
        "amount": "Payment amount (Rs).", "amount_over_user_avg": "Amount divided by the payer's median prior payment.",
        "is_night": "1 if the hour is < 6 or >= 22 (IST).", "hours_from_usual": "Circular distance (h) from the payer's most active hour; 0 when the usual hour is unknown.",
        "seconds_since_last_txn": "Seconds since the payer's previous payment, capped at one day.",
        "txns_last_hour": "Payer's payments in the preceding hour.", "txns_today_over_avg": "Payer's payments today divided by their daily average.",
        "payee_is_new": "1 if this payer has not paid this payee before (payer-relative novelty).",
        "payee_age_days": "Days since the payee was first observed in the store.", "payee_distinct_payers": "Distinct payers observed paying the payee.",
        "payee_repeat_ratio": "Share of the payee's payers who paid more than once (missing below 5 payers).",
        "payee_history_available": "1 if the payee has observed history, 0 otherwise.",
        # message / link / intent: from dataset_v2.message_features, link_features, intent_features
        "msg_coercion_probability": "Coercion model's logistic probability over the ten cue indicators (coercion.analyse_message).",
        "msg_coercion_score": "Production coercion score (0-100, after the weak-cue policy) divided by 100.",
        "msg_weak_cues_only": "1 if only weak cues (authority/urgency-type words) fired, so the score is capped.",
        "msg_length_chars": "Number of characters in the message.", "msg_word_count": "Number of words in the message.",
        "msg_digit_ratio": "Share of message characters that are digits.",
        "msg_uppercase_ratio": "Share of letters that are upper case.",
        "message_available": "1 if the payer supplied a message; 0 otherwise (all other message features are then missing, not zero).",
        "link_present": "1 if the message contains a URL, 0 if a message without a URL; missing when there is no message.",
        "link_suspicious_score": "Worst URL's scam_link.analyse_links score / 100 (0 if no URL; missing when there is no message).",
        "link_brand_conflict": "1 if a URL claims a brand different from the payee's brand (0 otherwise; missing when there is no message).",
        "intent_stated": "1 if the payer stated a purpose.",
        "intent_is_refund_or_asked": "1 if the stated purpose is refund, 'asked to' or investment; missing when no purpose is stated.",
        "intent_payee_mismatch": "1 if the purpose is bill or investment and the payee is a personal phone-number address; missing when no purpose is stated.",
        "payee_is_phone_address": "1 if the payee address is a personal phone-number VPA rather than a merchant-style handle.",
    }
    for _, r in doc.iterrows():
        meaning.setdefault(r["feature"], r["definition"])
    inv = []
    for c in cols:
        d = meaning.get(c)
        if d is None and c.startswith("msg_") and c[4:] in C_FEATURE_ORDER:
            d = f"1 if the coercion cue '{c[4:]}' fired in the message (coercion.extract_features); missing when there is no message."
        if d is None:
            for _, r in doc.iterrows():
                if "/" in r["feature"] and c in r["feature"].replace(" ", "").split("/"):
                    d = r["definition"]
                if r["feature"].startswith("msg_<10") and c.startswith("msg_") and c[4:] in r["definition"]:
                    d = "Binary: coercion pattern '" + c[4:] + "' fired in the message."
        s = df[c]
        inv.append({"#": len(inv) + 1, "Feature": c, "Evidence family": FAMILY_OF[c], "V2 group (code)": groups[c],
                    "Type": "binary" if set(s.dropna().unique()) <= {0.0, 1.0} else "numeric",
                    "Evidence kind": side[FAMILY_OF[c]], "Missing possible": "yes" if s.isna().any() else "no",
                    "Missing % (seed 42)": round(100 * s.isna().mean(), 2), "Pre-payment": "yes", "Meaning": d or "—"})
    t3 = pd.DataFrame(inv)
    md_table(t3, R / "tables/table03_feature_inventory.md", "Table 3 — V2 feature inventory (47 features)",
             "Evidence family: the live evidence family reading the same observation (scripts/final_v2/families.py). "
             "The tenth family, Classifier, is the V2 model itself (model-derived evidence; all 47 columns are its inputs). "
             "'Missing possible' is measured on the seed-42 corpus. All features are computed from state before the payment.")

    # ── 4. overall performance (validation and test) ─────────────────────
    wjson(R / "classifier/v2_metrics.json", {"validation": va, "test": te, "threshold_source": "models/metrics_v2.json (validation split)"})
    keys = [("roc_auc", "ROC-AUC"), ("pr_auc_average_precision", "PR-AUC (average precision)"), ("accuracy", "Accuracy"),
            ("precision", "Precision"), ("recall", "Recall (TPR)"), ("f1", "F1"), ("specificity", "Specificity (TNR)"),
            ("fpr", "FPR"), ("fnr", "FNR"), ("npv", "NPV"), ("balanced_accuracy", "Balanced accuracy"), ("mcc", "MCC"),
            ("brier", "Brier score"), ("brier_skill_score", "Brier skill score"), ("threshold", "Threshold"),
            ("prevalence", "Fraud prevalence"), ("alert_rate", "Alert rate")]
    t4 = pd.DataFrame([{"Metric": lab, "Validation (n=18,000)": f"{va[k]:.4f}", "Test (n=30,000)": f"{te[k]:.4f}"} for k, lab in keys])
    md_table(t4, R / "tables/table04_overall_performance.md", "Table 4 — Overall V2 classification performance (frozen model, seed 42)",
             "Threshold 0.1123 selected on the validation split (highest validation recall with validation FPR ≤ 1%). "
             "Validation values are the data the threshold was chosen on; test values are the single held-out measurement.")

    # ── 5. confusion matrix ──────────────────────────────────────────────
    cm = np.array([[te["tn"], te["fp"]], [te["fn"], te["tp"]]])
    pd.DataFrame(cm, index=["Actual Legitimate", "Actual Fraud"], columns=["Predicted Legitimate", "Predicted Fraud"]).to_csv(R / "confusion_matrix/confusion_matrix_test.csv")
    t5 = pd.DataFrame([{"": "Actual Legitimate", "Predicted Legitimate": f"TN = {te['tn']:,}", "Predicted Fraud": f"FP = {te['fp']:,}", "Total": f"{te['tn']+te['fp']:,}"},
                       {"": "Actual Fraud", "Predicted Legitimate": f"FN = {te['fn']:,}", "Predicted Fraud": f"TP = {te['tp']:,}", "Total": f"{te['fn']+te['tp']:,}"}])
    md_table(t5, R / "tables/table05_confusion_matrix.md", "Table 5 — V2 test confusion matrix (threshold 0.1123)",
             f"Precision {te['precision']:.4f} · Recall {te['recall']:.4f} · Specificity {te['specificity']:.4f} · FPR {te['fpr']:.5f} · "
             f"FNR {te['fnr']:.4f} · NPV {te['npv']:.5f} · F1 {te['f1']:.4f} · Accuracy {te['accuracy']:.4f} · "
             f"Balanced accuracy {te['balanced_accuracy']:.4f} · MCC {te['mcc']:.4f}")
    import matplotlib.pyplot as plt
    for norm in (False, True):
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        mat = cm / cm.sum(axis=1, keepdims=True) if norm else cm
        ax.imshow(mat if norm else np.log10(cm + 1), cmap="Blues", vmin=0)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                txt = f"{100*mat[i,j]:.2f}%\n({cm[i,j]:,})" if norm else f"{cm[i,j]:,}"
                dark = (mat[i, j] > 0.5) if norm else (np.log10(cm[i, j] + 1) > 2.5)
                ax.text(j, i, txt, ha="center", va="center", fontsize=11, color="white" if dark else "#0b0b0b")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted\nLegitimate", "Predicted\nFraud"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual\nLegitimate", "Actual\nFraud"])
        ax.set_title(("Normalised by actual class" if norm else "Raw counts") + "\nV2 test split, threshold 0.1123")
        style.save(fig, R / f"figures/fig01{'b_confusion_matrix_normalised' if norm else 'a_confusion_matrix'}")

    # ── 6/7. ROC and PR ──────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_te, p_te)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot(fpr, tpr, color=LEGIT, label=f"V2 (ROC-AUC = {te['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="Chance")
    ax.plot(te["fpr"], te["recall"], "o", ms=8, color=FRAUD, label=f"Operating point (FPR {te['fpr']:.4f}, TPR {te['recall']:.4f})")
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate (recall)")
    ax.set_title("ROC curve — V2, test split"); ax.legend(loc="lower right", fontsize=8)
    ins = ax.inset_axes([0.42, 0.3, 0.5, 0.38]); ins.plot(fpr, tpr, color=LEGIT, lw=1.5)
    ins.plot(te["fpr"], te["recall"], "o", ms=6, color=FRAUD); ins.set_xlim(0, 0.05); ins.set_ylim(0.4, 1.0)
    ins.set_title("FPR 0–5%", fontsize=7); ins.tick_params(labelsize=6)
    style.save(fig, R / "figures/fig02_roc_curve")
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(R / "roc_pr/roc_curve_test.csv", index=False)
    prec, rec, _ = precision_recall_curve(y_te, p_te)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot(rec, prec, color=LEGIT, label=f"V2 (PR-AUC = {te['pr_auc_average_precision']:.4f})")
    ax.axhline(te["prevalence"], ls="--", lw=1, color=MUTED, label=f"No-skill baseline (prevalence {te['prevalence']:.4f})")
    ax.plot(te["recall"], te["precision"], "o", ms=8, color=FRAUD, label=f"Operating point (R {te['recall']:.3f}, P {te['precision']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Precision–recall curve — V2, test split"); ax.legend(loc="lower left", fontsize=8)
    style.save(fig, R / "figures/fig03_pr_curve")
    pd.DataFrame({"recall": rec, "precision": prec}).to_csv(R / "roc_pr/pr_curve_test.csv", index=False)

    # ── 8. threshold analysis (validation grid; frozen threshold on test) ─
    grid = np.unique(np.concatenate([np.round(np.logspace(-3, np.log10(0.95), 80), 6), [thr]]))
    rows = []
    for split_name, yy, pp in (("validation", y_va, p_va), ("test", y_te, p_te)):
        for t in grid:
            m = binary_metrics(yy, pp, t)
            rows.append({"split": split_name, "threshold": t, **{k: m[k] for k in ("precision", "recall", "f1", "fpr", "fnr", "specificity", "alert_rate", "tp", "fp", "fn", "tn")}})
    sweep = pd.DataFrame(rows); sweep.rename(columns={"alert_rate": "predicted_fraud_rate"}, inplace=True)
    sweep.to_csv(R / "threshold/threshold_sweep.csv", index=False)
    vsw = sweep[sweep.split == "validation"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for col, lab, c in (("precision", "Precision", LEGIT), ("recall", "Recall", FRAUD), ("f1", "F1", AQUA)):
        ax.plot(vsw.threshold, vsw[col], label=lab, color=c)
    ax.plot(vsw.threshold, vsw.fpr * 10, label="FPR × 10", color=INK2, ls="--", lw=1.5)
    ax.axvline(thr, color="#0b0b0b", lw=1, ls=":"); ax.text(thr * 0.92, 1.72, f"selected θ = {thr}\n(validation FPR ≤ 1%)", fontsize=8, ha="right")
    ax.set_xscale("log"); ax.set_xlabel("Decision threshold (log scale)"); ax.set_ylabel("Metric value")
    ax.set_title("Threshold analysis on the validation split — V2"); ax.legend(loc="center left", fontsize=8)
    style.save(fig, R / "figures/fig06_threshold_analysis")
    op = pd.DataFrame([{"Split": "Validation (selection)", "Threshold": thr, "Recall": f"{va['recall']:.4f}", "FPR": f"{va['fpr']:.5f}",
                        "Precision": f"{va['precision']:.4f}", "F1": f"{va['f1']:.4f}", "Specificity": f"{va['specificity']:.4f}",
                        "Predicted fraud rate": f"{va['alert_rate']:.4f}"},
                       {"Split": "Test (frozen threshold)", "Threshold": thr, "Recall": f"{te['recall']:.4f}", "FPR": f"{te['fpr']:.5f}",
                        "Precision": f"{te['precision']:.4f}", "F1": f"{te['f1']:.4f}", "Specificity": f"{te['specificity']:.4f}",
                        "Predicted fraud rate": f"{te['alert_rate']:.4f}"}])
    md_table(op, R / "tables/table04b_selected_operating_point.md", "Table 4b — Selected operating point",
             f"Selection rule: highest validation recall with validation FPR ≤ 1% (train_model_v2.recall_at_fpr). Re-derived threshold "
             f"{thr_va:.6f} → rounded {round(float(thr_va), 4)} (stored {thr}).")

    # ── 9. calibration ───────────────────────────────────────────────────
    cal = {}
    for split_name, yy, pp in (("validation", y_va, p_va), ("test", y_te, p_te)):
        e_u, b_u = ece(yy, pp, 10, "uniform"); e_q, b_q = ece(yy, pp, 10, "quantile")
        b_u.to_csv(R / f"calibration/reliability_uniform10_{split_name}.csv", index=False)
        b_q.to_csv(R / f"calibration/reliability_quantile10_{split_name}.csv", index=False)
        cal[split_name] = {"ece_uniform_10": e_u, "ece_quantile_10_unique_edges": e_q, "bins_quantile_populated": int(len(b_q)),
                           "brier": float(brier_score_loss(yy, pp)), "mean_predicted": float(pp.mean()), "observed_rate": float(yy.mean())}
    wjson(R / "calibration/calibration_summary.json", cal)
    _, b_u = ece(y_te, p_te, 10, "uniform")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="Perfect calibration")
    ax.plot(b_u.mean_predicted, b_u.observed_fraud_rate, "o-", color=LEGIT, ms=6, label="V2, 10 equal-width bins")
    for _, r in b_u.iterrows():
        ax.annotate(f"n={int(r.n):,}", (r.mean_predicted, r.observed_fraud_rate), fontsize=6.5, color=INK2,
                    xytext=(4, -9), textcoords="offset points")
    ax.set_xlabel("Mean predicted fraud probability"); ax.set_ylabel("Observed fraud frequency")
    ax.set_title(f"Reliability diagram — test split\nECE (10 bins) = {cal['test']['ece_uniform_10']:.4f}, Brier = {te['brier']:.5f}")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    ax.bar(np.arange(len(b_u)), b_u.n, color=LEGIT, width=0.7)
    ax.set_yscale("log"); ax.set_xticks(np.arange(len(b_u)))
    ax.set_xticklabels([f"{r.bin_lower:.1f}–{r.bin_upper:.1f}" for _, r in b_u.iterrows()], rotation=60, fontsize=7)
    ax.set_xlabel("Predicted-probability bin"); ax.set_ylabel("Payments in bin (log)"); ax.set_title("Bin counts")
    style.save(fig, R / "figures/fig04_reliability_diagram")

    # ── 10. risk score distribution ──────────────────────────────────────
    stats = {}
    for lab, m in (("legitimate", y_te == 0), ("fraud", y_te == 1)):
        q = np.quantile(p_te[m], [0.05, 0.25, 0.5, 0.75, 0.95])
        stats[lab] = {"n": int(m.sum()), "mean": float(p_te[m].mean()), "p05": q[0], "p25": q[1], "median": q[2],
                      "p75": q[3], "p95": q[4], "share_above_threshold": float((p_te[m] >= thr).mean())}
    wjson(R / "classifier/score_distribution_test.json", stats)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(p_te[y_te == 0], bins=bins, color=LEGIT, alpha=0.55, label=f"Legitimate (n={int((y_te==0).sum()):,})")
    ax.hist(p_te[y_te == 1], bins=bins, color=FRAUD, alpha=0.75, label=f"Fraud (n={int(y_te.sum())})", histtype="stepfilled")
    ax.axvline(thr, color="#0b0b0b", ls=":", lw=1.2, label=f"Threshold {thr}")
    ax.set_yscale("log"); ax.set_xlabel("Predicted fraud probability (calibrated V2)"); ax.set_ylabel("Payments (log scale)")
    ax.set_title("Risk-score distribution — V2, test split"); ax.legend(fontsize=8)
    style.save(fig, R / "figures/fig05_risk_score_distribution")

    # ── 13. classification report ────────────────────────────────────────
    cr = classification_report(y_te, (p_te >= thr).astype(int), labels=[0, 1], target_names=["legitimate", "fraud"],
                               output_dict=True, digits=4)
    crd = pd.DataFrame([{"Class": k, "Precision": f"{v['precision']:.4f}", "Recall": f"{v['recall']:.4f}", "F1": f"{v['f1-score']:.4f}",
                         "Support": int(v["support"])} for k, v in cr.items() if k != "accuracy"])
    md_table(crd, R / "tables/table04c_classification_report.md", "Table 4c — Classification report (V2, test split, threshold 0.1123)",
             f"Accuracy {cr['accuracy']:.4f} (dominated by the 98.8% legitimate class; not a headline metric).")

    # ── 14. scenario analysis ────────────────────────────────────────────
    rows = []
    flagged = p_te >= thr; s_te = scen[i_te]
    for s in SCEN:
        m = (s_te == s) & (y_te == 1)
        rows.append({"Scenario (synthetic)": SCEN_LABEL[s], "Test fraud": int(m.sum()), "Detected (TP)": int(flagged[m].sum()),
                     "Missed (FN)": int((~flagged[m]).sum()), "Recall": f"{flagged[m].mean():.4f}",
                     "Median probability": f"{np.median(p_te[m]):.4f}", "Mean probability": f"{p_te[m].mean():.4f}"})
    rows.append({"Scenario (synthetic)": "Legitimate (false alarms)", "Test fraud": int((y_te == 0).sum()), "Detected (TP)": "—",
                 "Missed (FN)": "—", "Recall": f"FPR {te['fpr']:.5f} ({te['fp']} FP)", "Median probability": f"{np.median(p_te[y_te==0]):.6f}",
                 "Mean probability": f"{p_te[y_te==0].mean():.4f}"})
    t6 = pd.DataFrame(rows)
    md_table(t6, R / "tables/table06_scenario_performance.md", "Table 6 — Scenario-level fraud detection (V2, test split, threshold 0.1123)",
             "Per-scenario precision and F1 are not reported: false positives are legitimate payments and belong to no scenario, "
             "so a scenario-specific precision is not defined. Scenario categories and proportions are simulator settings.")
    rows[-1]["Test fraud"] = int((y_te == 0).sum())
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    rec_ = [flagged[(s_te == s) & (y_te == 1)].mean() for s in SCEN]
    ns = [int(((s_te == s) & (y_te == 1)).sum()) for s in SCEN]
    yy_ = np.arange(len(SCEN))[::-1]
    ax.barh(yy_, rec_, color=FRAUD, height=0.55)
    for yv, r_, n_, s in zip(yy_, rec_, ns, SCEN):
        ax.text(r_ + 0.01, yv, f"{100*r_:.1f}%  ({int(round(r_*n_))}/{n_})", va="center", fontsize=8, color="#0b0b0b")
    ax.axvline(te["recall"], color=INK2, ls="--", lw=1); ax.text(te["recall"] - 0.01, 3.45, f"overall {100*te['recall']:.1f}%", fontsize=7.5, ha="right", color=INK2)
    ax.set_yticks(yy_); ax.set_yticklabels([SCEN_LABEL[s] for s in SCEN]); ax.set_xlim(0, 1.18)
    ax.set_xlabel("Recall at threshold 0.1123"); ax.set_title("Recall by synthetic fraud scenario — V2, test split")
    ax.grid(axis="y", visible=False)
    style.save(fig, R / "figures/fig11_scenario_performance")

    # ── 15. feature importance ───────────────────────────────────────────
    from sklearn.inspection import permutation_importance
    X_te = df[cols].iloc[i_te]
    perm = permutation_importance(model, X_te, y_te, scoring="average_precision", n_repeats=5, random_state=42, n_jobs=1)
    imp = pd.DataFrame({"feature": cols, "family": [FAMILY_OF[c] for c in cols],
                        "perm_importance_mean": perm.importances_mean, "perm_importance_std": perm.importances_std})
    old_perm = pd.read_csv(ROOT / "evaluation_v2/csv/feature_importance_permutation.csv")
    oc = [c for c in old_perm.columns if "mean" in c or "drop" in c][0]
    mm = imp.merge(old_perm[["feature", oc]], on="feature")
    rng = np.random.default_rng(42)
    fam_rows = []
    base_ap = average_precision_score(y_te, p_te)
    for fam in FAMILY_ORDER[:-1]:
        fc = [c for c in cols if FAMILY_OF[c] == fam]
        drops = []
        for rep_ in range(5):
            Xp = X_te.copy(); perm_idx = rng.permutation(len(Xp))
            Xp[fc] = Xp[fc].to_numpy()[perm_idx]
            drops.append(base_ap - average_precision_score(y_te, model.predict_proba(Xp)[:, 1]))
        fam_rows.append({"Family": fam, "Number of features": len(fc), "Grouped permutation importance (AP drop)": float(np.mean(drops)),
                         "sd": float(np.std(drops)), "Sum of single-feature importances": float(imp[imp.family == fam].perm_importance_mean.sum())})
    fam_df = pd.DataFrame(fam_rows).sort_values("Grouped permutation importance (AP drop)", ascending=False)
    shap_df = None
    try:
        import shap
        take = np.sort(np.random.default_rng(42).choice(len(X_te), size=200, replace=False))
        booster = model.calibrated_classifiers_[0].estimator
        sv = shap.TreeExplainer(booster).shap_values(X_te.iloc[take])
        shap_df = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        imp = imp.merge(shap_df, on="feature")
        old_shap = pd.read_csv(ROOT / "evaluation_v2/csv/feature_importance_shap.csv")
        shap_cmp = float(np.max(np.abs(old_shap.set_index("feature").mean_abs_shap.reindex(cols).to_numpy() - shap_df.mean_abs_shap.to_numpy())))
    except Exception as exc:          # recorded, not hidden
        shap_cmp = f"SHAP not computed: {exc}"
    imp.sort_values("perm_importance_mean", ascending=False).to_csv(R / "feature_importance/feature_importance_v2.csv", index=False)
    fam_df.to_csv(R / "feature_importance/family_importance_v2.csv", index=False)
    wjson(R / "feature_importance/importance_verification.json", {
        "permutation_max_abs_diff_vs_reported": float(np.max(np.abs(mm.perm_importance_mean - mm[oc]))),
        "shap_max_abs_diff_vs_reported": shap_cmp,
        "method": "permutation importance on the frozen calibrated model, test split, scoring = average precision, 5 repeats, random_state 42; "
                  "SHAP TreeExplainer on the uncalibrated booster of calibration fold 1, 200 test rows (as evaluate_model_v2.py)"})
    top = imp.sort_values("perm_importance_mean", ascending=False).head(15).reset_index(drop=True)
    t12 = pd.DataFrame({"Rank": np.arange(1, len(top) + 1), "Feature": top.feature, "Evidence family": top.family,
                        "Permutation importance (AP drop)": top.perm_importance_mean.map(lambda v: f"{v:.4f}"),
                        "sd": top.perm_importance_std.map(lambda v: f"{v:.4f}"),
                        "Mean |SHAP| (booster)": (top.mean_abs_shap.map(lambda v: f"{v:.4f}") if "mean_abs_shap" in top else "—")})
    md_table(t12, R / "tables/table12a_top_features.md", "Table 12a — Top 15 V2 features by permutation importance (test split)",
             "Importance = mean drop in average precision when the column is shuffled (5 repeats). Correlated features share importance; "
             "importance is predictive usefulness under this protocol, not causal influence, and is distinct from the ten-family risk aggregation.")
    t12b = fam_df.copy()
    for c in ("Grouped permutation importance (AP drop)", "sd", "Sum of single-feature importances"):
        t12b[c] = t12b[c].map(lambda v: f"{v:.4f}")
    md_table(t12b, R / "tables/table12b_family_importance.md", "Table 12b — Evidence-family importance (V2, test split)",
             "Grouped permutation importance shuffles all columns of a family together (5 repeats), which avoids splitting "
             "importance between correlated columns. The Classifier family has no input column of its own.")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    t20 = imp.sort_values("perm_importance_mean", ascending=False).head(20)[::-1]
    fam_col = {f: c for f, c in zip(FAMILY_ORDER, [style.BLUE, style.ORANGE, style.AQUA, style.YELLOW, style.MAGENTA, style.GREEN, style.VIOLET, style.RED, "#7f7f7f"])}
    ax.barh(t20.feature, t20.perm_importance_mean, xerr=t20.perm_importance_std, color=[fam_col[f] for f in t20.family],
            height=0.65, error_kw=dict(lw=0.8, capsize=2))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=fam_col[f], label=f) for f in FAMILY_ORDER[:-1] if f in set(t20.family)], fontsize=7, loc="lower right")
    ax.set_xlabel("Mean drop in average precision when shuffled"); ax.set_title("Top 20 V2 features — permutation importance (test split)")
    ax.grid(axis="y", visible=False)
    style.save(fig, R / "figures/fig07_feature_importance")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    fd = fam_df[::-1]
    ax.barh(fd.Family, fd["Grouped permutation importance (AP drop)"], xerr=fd.sd, color=[fam_col[f] for f in fd.Family], height=0.6,
            error_kw=dict(lw=0.8, capsize=2))
    for yv, (v, n_) in enumerate(zip(fd["Grouped permutation importance (AP drop)"], fd["Number of features"])):
        ax.text(v + 0.005, yv, f"{v:.3f}  ({n_} feat.)", va="center", fontsize=7.5)
    ax.set_xlabel("Drop in average precision when the whole family is shuffled"); ax.set_title("Evidence-family importance — V2, test split")
    ax.grid(axis="y", visible=False)
    style.save(fig, R / "figures/fig08_family_importance")
    if shap_df is not None:
        sv_df = pd.DataFrame(sv, columns=cols)
        order = shap_df.sort_values("mean_abs_shap", ascending=False).feature.head(15).tolist()[::-1]
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        Xs = X_te.iloc[take]
        for k_, c in enumerate(order):
            v = sv_df[c].to_numpy(); x = Xs[c].to_numpy(dtype=float)
            ok = np.isfinite(x); col = np.full(len(x), 0.5)
            if ok.any() and np.nanmax(x) > np.nanmin(x):
                col[ok] = (x[ok] - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
            jitter = np.random.default_rng(k_).uniform(-0.25, 0.25, len(v))
            sc = ax.scatter(v, k_ + jitter, c=np.where(ok, col, np.nan), cmap="coolwarm", s=6, vmin=0, vmax=1)
            ax.scatter(v[~ok], (k_ + jitter)[~ok], color="#9a9a9a", s=6)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7.5); ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("SHAP value (booster log-odds scale)"); ax.set_title("SHAP summary — top 15 features (200 test rows)\ncolour: feature value low→high; grey: missing")
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01).set_label("Feature value (scaled)", fontsize=7)
        style.save(fig, R / "figures/fig07b_shap_summary")
    wjson(R / "classifier/run_summary.json", {"test": te, "validation": va, "reproduction": rep})
    print(json.dumps({"reproduction": {k: rep[k] for k in ("identical_confusion", "max_abs_metric_diff", "threshold_matches")},
                      "test": {k: te[k] for k in ("pr_auc_average_precision", "roc_auc", "recall", "fpr", "precision", "f1", "mcc", "brier", "brier_skill_score")},
                      "calibration": cal["test"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
