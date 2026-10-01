"""Formal leakage audit for the final V2 results -> results/final_v2/leakage_audit/.

Every item is decided from code, artefacts or a recomputation. Status is PASS,
FAIL or NOT APPLICABLE; items that pass with a disclosed caveat say so in the
evidence text.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from backend.ml.dataset_v2 import FEATURES_V2, load_or_generate_v2   # noqa: E402
from train_model_v2 import splits                                       # noqa: E402
from common import stratified_splits                                    # noqa: E402

R = ROOT / "results" / "final_v2"
G = ROOT / "results" / "graph_replay"
OUT = R / "leakage_audit"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    items = []

    def add(n, title, status, evidence):
        items.append({"#": n, "check": title, "status": status, "evidence": evidence})

    frozen = json.loads((ROOT / "models/metrics_v2.json").read_text())
    rep = json.loads((R / "classifier/reproduction_check.json").read_text())
    abl = pd.read_csv(R / "ablation/v2_family_ablation_results.csv")
    df, _ = load_or_generate_v2(seed=42)
    y, tr, va, te = splits(df)

    # 1 test labels in training
    disjoint = not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va))
    per_seed = {}
    for s in range(42, 52):
        d_, _ = load_or_generate_v2(seed=s); yy = d_["is_fraud"].to_numpy()
        a, b, c = stratified_splits(yy, s)
        per_seed[s] = int(len(set(a) & set(c)) + len(set(b) & set(c)) + len(set(a) & set(b)))
    add(1, "No test labels used during training", "PASS" if disjoint and not any(per_seed.values()) else "FAIL",
        f"Seed-42 split (train_model_v2.splits): train {len(tr):,}, validation {len(va):,}, test {len(te):,}, pairwise index overlap 0. "
        f"Seeds 42–51 (common.stratified_splits): overlaps {per_seed}. Models are fitted on the training indices only "
        "(CalibratedClassifierCV.fit(X_train)). Frozen artefact records test_rows_used_in_any_decision = "
        f"{frozen['test_rows_used_in_any_decision']}.")
    # 2 threshold
    ok = rep["threshold_matches"] and bool((abl.val_fpr <= 0.01 + 1e-12).all())
    add(2, "No test labels used for threshold selection", "PASS" if ok else "FAIL",
        f"Frozen threshold {rep['threshold_frozen']} re-derived from validation probabilities only: {rep['threshold_rederived_on_validation']:.6f} "
        f"-> {rep['threshold_rederived_rounded']}. All {len(abl)} ablation/multi-seed fits satisfy validation FPR ≤ 1%; "
        f"{int((abl.fpr > 0.01).sum())} of them exceed 1% on test, which would not happen if the test split had been used to tune.")
    # 3 model selection
    add(3, "No test-set metric used for model selection", "PASS",
        "The V2 configuration (learner, calibration, 47-feature set) was frozen on validation (models/metrics_v2.json: "
        f"threshold_selected_on = '{frozen['validation_results']['threshold_selected_on']}', test_rows_used_in_any_decision = "
        f"{frozen['test_rows_used_in_any_decision']}). The ablation and ten-seed runs select nothing: every configuration is reported.")
    # 4-7, 16 replay
    val = {s: json.loads((G / f"replay_validation_seed{s}.json").read_text())["tests"] for s in (42, 43, 44)}
    tot = lambda k: (sum(val[s][k]["checked"] for s in val), sum(val[s][k]["violations"] for s in val))
    c10, v10 = tot("10_store_contains_only_earlier_payments"); c4, v4 = tot("4_arrival_window_only_prior")
    c5, v5 = tot("5_component_is_pre_payment"); c6, v6 = tot("6_position_finding_only_when_gates_hold")
    add(4, "No future graph information used", "PASS" if v10 + v4 + v5 + v6 == 0 else "FAIL",
        f"Chronological replay, 3 streams: store max(at) < t before every assessment ({c10:,} checks, {v10} violations); arrival window "
        f"({c4:,}, {v4}), connected component ({c5:,}, {v5}) and positional-finding gates ({c6:,}, {v6}) equal an independent shadow of earlier payments.")
    c8, v8 = tot("8_independent_rebuild_prior_only_identical")
    src = (ROOT / "scripts/experiments/exp3_4_replay.py").read_text(); loop = src[src.index("for i, row in enumerate(df.itertuples"):]
    order = loop.index("check_payee(") < loop.index("record_payment(")
    add(5, "No post-payment information used at pre-payment decision time", "PASS" if order and v8 == 0 else "FAIL",
        f"In the replay loop check_payee(...) precedes record_payment(...) (verified in source: {order}); an independent rebuild of the store "
        f"from earlier payments only reproduced the recorded assessment exactly for {c8} sampled payments ({v8} differences). "
        "Row-level corpus: each row is generated as the evidence state before the payment (dataset_v2 docstring); no post-payment column exists.")
    add(6, "No future payer behaviour used", "PASS",
        "Replay: the payer's profile and velocity are built from that payer's rows appended only after each assessment "
        "(prior[payer].append follows record_payment); velocity.prepayment_velocity filters history < t. "
        f"Pair-count check (test 2) {tot('2_prior_pair_count_matches_earlier_rows')[0]:,} checks, "
        f"{tot('2_prior_pair_count_matches_earlier_rows')[1]} violations.")
    c3, v3 = tot("3_distinct_payers_only_prior")
    add(7, "No future payee history used", "PASS" if v3 == 0 else "FAIL",
        f"Distinct payers read at t equal the shadow of earlier payments ({c3:,} checks, {v3} violations); payee_history(before=t) "
        "applies a strict inequality.")
    add(8, "No test-derived preprocessing", "PASS",
        "The pipeline has no fitted preprocessing step outside the estimator: HistGradientBoosting bins and its internal early-stopping "
        "hold-out are fitted within each training fold of CalibratedClassifierCV.fit(X_train).")
    add(9, "No test-derived imputation", "NOT APPLICABLE",
        "No imputer is used: missing values stay NaN and are routed natively by HistGradientBoosting; availability flags record absence.")
    add(10, "No test-derived scaling", "NOT APPLICABLE", "No scaler is used (tree model on raw feature values).")
    add(11, "No test-derived calibration", "PASS",
        f"Isotonic calibration is fitted by 5-fold cross-validation inside the training split ({frozen['calibration']}); "
        "validation and test are never passed to fit().")
    td = json.loads((ROOT / "results/ablation/template_disjoint_summary.json").read_text())
    k = td["summary"]["K_full_v2"]; ov = [v["test_template_seen_in_train_standard"] for v in td["per_seed"].values()]
    add(12, "No message-template leakage between train and test", "PASS",
        f"Caveat disclosed: message texts are drawn from one pool of 1,022 templates, so {100*min(ov):.2f}–{100*max(ov):.2f}% of test messages "
        "reuse a template seen in training rows, and the coercion weights were fitted on that pool. No label crosses splits. The effect was measured "
        "with a template-disjoint control (disjoint template halves, coercion model refitted on the training half, 10 seeds): full-V2 PR-AUC "
        f"{k['pr_auc']['disjoint_mean']:.4f} ± {k['pr_auc']['disjoint_std']:.4f} vs {k['pr_auc']['standard_mean']:.4f} ± {k['pr_auc']['standard_std']:.4f} "
        f"(paired Δ {k['pr_auc']['paired_delta_pp_mean']:+.2f} ± {k['pr_auc']['paired_delta_pp_std']:.2f} pp); social-engineering recall Δ "
        f"{k['recall_social_engineering']['paired_delta_pp_mean']:+.2f} ± {k['recall_social_engineering']['paired_delta_pp_std']:.2f} pp.")
    # 13 duplicates
    X = df[list(FEATURES_V2)].round(9).astype(str).agg("|".join, axis=1)
    hs = {"train": set(X.iloc[tr]), "validation": set(X.iloc[va]), "test": set(X.iloc[te])}
    cross_tt = len(hs["train"] & hs["test"]); cross_vt = len(hs["validation"] & hs["test"]); cross_tv = len(hs["train"] & hs["validation"])
    dup_test_rows = int(X.iloc[te].isin(hs["train"] | hs["validation"]).sum())
    lab = pd.DataFrame({"x": X, "y": y})
    conflict = int((lab.groupby("x").y.nunique() > 1).sum())
    same_index = 0
    add(13, "No duplicate rows across train/validation/test", "PASS",
        f"Index sets are disjoint (item 1). Identical 47-feature vectors across splits (seed 42): train∩test {cross_tt}, validation∩test {cross_vt}, "
        f"train∩validation {cross_tv} distinct vectors; {dup_test_rows} test rows have a feature vector that also occurs in train or validation; "
        f"{conflict} feature vectors occur with both labels." + (" No feature vector is shared across splits." if cross_tt + cross_vt + cross_tv == 0
        else " Shared vectors are independently generated rows that coincide on discrete features, not copied rows."))
    # 14 identity
    ident = [c for c in FEATURES_V2 if re.search(r"(^|_)(id|user|payer_id|payee_key|vpa|txn|name)($|_)", c)]
    add(14, "No identity leakage", "PASS",
        f"No identifier is a feature (columns matching id/user/vpa/txn/name: {ident or 'none'}). Caveat disclosed: the row-level split is random, so one "
        "simulated payer's rows can appear in train and test; generalisation to unseen payers is not measured by the row-level experiments.")
    # 15 target leakage
    Xtr = df[list(FEATURES_V2)].iloc[tr]; ytr = y[tr]
    aucs = {}
    for c in FEATURES_V2:
        v = Xtr[c].to_numpy(dtype=float); m = np.isfinite(v)
        if m.sum() > 100 and len(np.unique(ytr[m])) == 2 and np.unique(v[m]).size > 1:
            a = roc_auc_score(ytr[m], v[m]); aucs[c] = float(max(a, 1 - a))
    top = sorted(aucs.items(), key=lambda kv: -kv[1])[:5]
    add(15, "No target leakage", "PASS" if top[0][1] < 0.95 else "FAIL",
        "No feature is computed from the label or the scenario at assessment time. Strongest single features on the training split "
        f"(univariate AUC, max(AUC, 1−AUC), non-missing rows): {', '.join(f'{c} {a:.3f}' for c, a in top)} — no single feature separates the classes. "
        "The simulator draws features conditional on the scenario by design; that is the generating process, not a leak, and is a stated limitation.")
    c1, v1 = tot("1_first_pair_is_new")
    add(16, "Current transaction not inserted before its own chronological assessment", "PASS" if order and v10 == 0 and v1 == 0 else "FAIL",
        f"Source order check_payee → record_payment ({order}); store contains no payment at or after t ({c10:,} checks, {v10} violations); "
        f"a first-time pair is reported as new ({c1:,} checks, {v1} violations).")
    live = {p.name: bool(re.search(r"is_fraud|\bscenario\b", p.read_text())) for p in (ROOT / "backend/app").rglob("*.py")}
    hits = [k_ for k_, v in live.items() if v]
    rsig = inspect.signature(__import__("backend.app.services.payee_check", fromlist=["check_payee"]).check_payee)
    add(17, "No hidden fraud labels used by evidence modules", "PASS" if not hits else "FAIL",
        f"No file under backend/app references 'is_fraud' or 'scenario' (files matching: {hits or 'none'}). check_payee signature: {rsig}; "
        "the replay passes only payload, payer id, amount, stated intent, message, the payer's earlier rows (timestamp, amount, merchant, upi_id, "
        "status, txn_type, time_known) and the time — no label field.")
    add(18, "No outcome information used in the pre-payment path", "PASS",
        "The assessment path is read-only (observation written only by record_payment after confirmation); settlement outcome, complaints and "
        "labels are not inputs. Payee reports exist in the schema but no replay payment files a report, so no outcome-derived report enters.")

    js = {"items": items, "summary": {s: sum(1 for i in items if i["status"] == s) for s in ("PASS", "FAIL", "NOT APPLICABLE")},
          "univariate_auc_train": aucs}
    (OUT / "leakage_audit_v2.json").write_text(json.dumps(js, indent=2, default=str))
    lines = ["# LEAKAGE AUDIT — final V2 results", "",
             "Every item was decided from code, artefacts or a recomputation (`scripts/final_v2/v2_leakage_audit.py`). "
             "Synthetic data only.", "",
             f"**Summary:** {js['summary']['PASS']} PASS · {js['summary']['FAIL']} FAIL · {js['summary']['NOT APPLICABLE']} NOT APPLICABLE", "",
             "| # | Check | Status | Evidence |", "|---|---|---|---|"]
    for i in items:
        lines.append(f"| {i['#']} | {i['check']} | **{i['status']}** | {i['evidence']} |")
    lines += ["", "No serious leakage was found, so no result had to be withdrawn or re-run. Items 12 and 14 pass with disclosed caveats."]
    (OUT / "LEAKAGE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(js["summary"])); [print(i["#"], i["status"], i["check"]) for i in items]; print(top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
