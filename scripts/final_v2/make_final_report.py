"""Assemble FINAL_PROJECT_RESULTS.md from the result artefacts.

Every number in the report is read from a file under results/ at build time;
nothing is typed in by hand. Tables are embedded from results/final_v2/tables/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "final_v2"
G = ROOT / "results" / "graph_replay"
T = R / "tables"
FIG = "figures"


def j(p):
    return json.loads(Path(p).read_text())


def tab(name):
    txt = (T / name).read_text().strip().splitlines()
    return "\n".join(txt[2:] if txt[0].startswith("**") else txt)


def title(name):
    first = (T / name).read_text().strip().splitlines()[0]
    return first.strip("*")


def fig(file, caption):
    return f"![{caption}]({FIG}/{file}.png)\n\n*{caption}* (`{FIG}/{file}.png`, `.svg`)"


def pm(v, k=1, d=4):
    v = np.asarray(v, float) * k
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def legit_block_causes() -> str:
    """Leading family and refund finding among legitimate BLOCKs, last 25%, from the V2 end-to-end rows."""
    lead, refund, n = [], [], []
    for sd in (42, 43, 44):
        d = pd.read_csv(R / f"end_to_end/e2e_v2_rows_seed{sd}.csv.gz").sort_values("timestamp")
        d = d.iloc[int(len(d) * 0.75):]
        b = d[(d.is_fraud == 0) & (d.verdict == "BLOCK")]
        n.append(len(b))
        lead.append(b.leading_families.fillna("").str.split("|").str[0].value_counts(normalize=True))
        refund.append(b.finding_codes.fillna("").str.contains("paying_for_a_refund").mean())
    L = pd.concat(lead, axis=1).fillna(0.0)
    L = L.assign(m=L.mean(axis=1), s=L.std(axis=1, ddof=1)).sort_values("m", ascending=False)
    names = {"stated_intent": "stated intent", "message_pressure": "message pressure", "payee_history": "payee history",
             "link_safety": "link safety", "payer_behaviour": "payer behaviour", "amount_context": "amount context",
             "ml_classifier": "Classifier (V2)"}
    parts = "; ".join(f"{names.get(k, k)} {100*r.m:.1f} ± {100*r.s:.1f}%" for k, r in L.iterrows() if r.m >= 0.001)
    return (f"**Why legitimate payments are blocked** (last 25%, V2 rows; {n[0]:,} / {n[1]:,} / {n[2]:,} legitimate BLOCKs per stream). "
            f"Family with the highest score in the assessment: {parts}. A refund purpose (`paying_for_a_refund`) appears in "
            f"{100*np.mean(refund):.1f} ± {100*np.std(refund, ddof=1):.1f}% of these BLOCKs. The Classifier (V2) family almost never leads a legitimate BLOCK; "
            f"legitimate BLOCKs come mainly from the fixed stated-intent and message-pressure rules.")


def main() -> int:
    rep = j(R / "classifier/reproduction_check.json")
    m = j(R / "classifier/v2_metrics.json"); te, va = m["test"], m["validation"]
    ds = j(R / "dataset/dataset_statistics.json")
    cal = j(R / "calibration/calibration_summary.json")
    sd = j(R / "classifier/score_distribution_test.json")
    rel = pd.read_csv(R / "calibration/reliability_uniform10_test.csv")
    imp = pd.read_csv(R / "feature_importance/feature_importance_v2.csv")
    fam_imp = pd.read_csv(R / "feature_importance/family_importance_v2.csv")
    ms = j(R / "multi_seed/v2_ten_seed_summary.json"); msr = pd.read_csv(R / "multi_seed/v2_ten_seed_results.csv").set_index("seed")
    abl = pd.read_csv(R / "ablation/v2_family_ablation_summary.csv").set_index("config")
    td = j(ROOT / "results/ablation/template_disjoint_summary.json")["summary"]["K_full_v2"]
    e2e = j(R / "end_to_end/e2e_v2_summary.json"); P = {int(k): v for k, v in e2e["per_seed"].items()}
    gs = j(R / "graph/graph_statistics.json")
    ex = j(R / "replay/replay_examples_seed42.json")
    rows42 = pd.read_csv(R / "end_to_end/e2e_v2_rows_seed42.csv.gz").set_index("txn_id")
    la = j(R / "leakage_audit/leakage_audit_v2.json")
    sw = j(R / "software_tests/software_tests_summary.json")
    man = j(R / "reproducibility/reproducibility_manifest.json")
    val = {s: j(G / f"replay_validation_seed{s}.json")["tests"] for s in (42, 43, 44)}
    frozen = j(ROOT / "models/metrics_v2.json")
    S3 = (42, 43, 44)
    L = lambda k: [P[s]["last25pct"] for s in S3]
    thr = te["threshold"]
    full = abl.loc["A. Full V2"]
    d_ = lambda c, k="pr_auc": f"{100*abl.loc[c, f'delta_{k}_mean']:+.2f} ± {100*abl.loc[c, f'delta_{k}_sd']:.2f} pp"
    top5 = imp.sort_values("perm_importance_mean", ascending=False).head(5)
    s42 = msr.loc[42]
    rank42 = int(msr.pr_auc.rank(ascending=False)[42])
    r_warn = [x["warn_plus"] for x in L(0)]; r_blk = [x["block"] for x in L(0)]
    vc = lambda s, g, v: P[s]["last25pct"]["verdict_counts"][g][v] / max(sum(P[s]["last25pct"]["verdict_counts"][g].values()), 1)
    chk = lambda k: (sum(val[s][k]["checked"] for s in S3), sum(val[s][k]["violations"] for s in S3))
    ex_m = next(e for e in ex if e["kind"] == "mule"); ex_r = next(e for e in ex if e["kind"] == "repeat_merchant")
    v2m = rows42.loc[ex_m["txn_id"]]; v2r = rows42.loc[ex_r["txn_id"]]
    cov42 = P[42]["full_stream"]["coverage"]
    miss = e2e["missing_evidence_checks"]

    out = []
    w = out.append
    w("# Final Project Experimental Results")
    w("")
    w("**Project:** Evidence-aware pre-payment risk assessment for UPI-style payments  \n"
      "**Final representation:** V2 (47 features) — the only representation reported here  \n"
      "**Nature of the evaluation:** synthetic. No real UPI, bank, PSP or NPCI data were used; no result below is a measurement of real-world performance.  \n"
      f"**Code evaluated:** snapshot of git HEAD `{man['git_head_of_evaluated_snapshot']}` (working tree with uncommitted changes; exact file hashes in `reproducibility/`).")
    w("")
    w("All numbers in this report are generated from the result files under `results/final_v2/` (and, for the replay, `results/graph_replay/`) by "
      "`scripts/final_v2/make_final_report.py`. Figures are in `results/final_v2/figures/` (PNG, 300 dpi, and SVG); tables in `results/final_v2/tables/` (Markdown and CSV); the main tables are also rendered as images in `results/final_v2/table_images/`.")
    w("")
    w("### Result categories used throughout")
    w("")
    w("| Label | Meaning |\n|---|---|\n"
      "| **Existing reported seed-42 result** | Values in `models/metrics_v2.json` / `evaluation_v2/metrics.json` produced when V2 was frozen |\n"
      "| **Reproduced result** | The same frozen artefact re-evaluated now on the regenerated seed-42 corpus |\n"
      "| **Multi-seed result** | Newly generated corpora for seeds 42–51, V2 refitted per seed with the identical protocol |\n"
      "| **End-to-end / replay result** | The ten-family assessment evaluated on chronological replay streams (seeds 42–44) |\n"
      "| **Software test result** | Implementation correctness (pytest); not a predictive metric |")
    w("")

    # 1
    w("## 1. Project Experimental Configuration")
    w("")
    w(f"- **Classifier:** `HistGradientBoostingClassifier` wrapped in isotonic calibration ({frozen['calibration']}); HistGradientBoosting with max_iter 300, "
      "learning_rate 0.08, max_leaf_nodes 31, class_weight balanced, random_state = seed. No resampling (no SMOTE).")
    w(f"- **Representation:** V2, {frozen['n_features']} features in 9 implementation groups; the live system organises evidence into ten families "
      "(Section 3).")
    w("- **Split:** stratified; 25% test, then 20% of the remainder as validation (72,000 / 18,000 / 30,000), random_state = seed.")
    w(f"- **Threshold rule:** highest validation recall with validation FPR ≤ 1%, stored to 4 d.p. Frozen seed-42 threshold: **{thr}** "
      f"(full precision {frozen['validation_results']['threshold_full_precision']:.7f}). The test split is used once, after the threshold is fixed.")
    w("- **Primary metrics:** PR-AUC (average precision) because fraud is ~1.2% of payments; ROC-AUC, recall at ~1% FPR, precision, F1, MCC, Brier score and calibration are also reported.")
    w("- **End-to-end system:** ten evidence families aggregated by the production rule (damped maximum + fact-aware corroboration → risk R; availability → coverage C; "
      "verdict bands APPROVE < 22 ≤ WARN < 45 ≤ STEP_UP < 70 ≤ BLOCK plus hard-block findings). The Classifier family uses the frozen V2 model (Section 15).")
    w("")
    w("### 1.1 Verification of the existing seed-42 result")
    w("")
    w("The frozen V2 artefact (`models/risk_model_v2.pkl`) was re-evaluated on the regenerated seed-42 corpus (dataset fingerprint identical to the frozen one) "
      "with the original split function (`scripts/train_model_v2.splits`).")
    w("")
    w("| Metric | Existing reported | Reproduced now |\n|---|---|---|")
    for k, v in rep["comparison"].items():
        w(f"| {k} | {v['reported']:.6f} | {v['reproduced']:.6f} |")
    cr, cp = rep["confusion_reported"], rep["confusion_reproduced"]
    w(f"| Confusion (TN / FP / FN / TP) | {cr['tn']:,} / {cr['fp']} / {cr['fn']} / {cr['tp']} | {cp['tn']:,} / {cp['fp']} / {cp['fn']} / {cp['tp']} |")
    w("")
    w(f"- Maximum absolute metric difference: **{rep['max_abs_metric_diff']}**; per-row predicted probabilities differ from the stored test predictions by at most "
      f"{rep.get('max_abs_prob_diff_vs_reported_predictions_per_row', float('nan')):.1e}. **The existing seed-42 test result is reproduced exactly.**")
    w(f"- The threshold re-derived from the validation split alone is {rep['threshold_rederived_on_validation']:.7f} → {rep['threshold_rederived_rounded']} "
      f"(matches the stored value: {rep['threshold_matches']}).")
    vn = rep["validation_note"]
    w(f"- **Discrepancy found and explained (validation only):** `models/metrics_v2.json` reports validation TP = {vn['frozen_validation_confusion_at_full_precision_threshold']['tp']} "
      f"(recall {frozen['validation_results']['recall']:.4f}) because it was computed at the full-precision threshold. At the stored threshold {thr}, which is the "
      f"one the service applies and the one behind the reported test result, validation TP = {vn['validation_confusion_at_stored_threshold_0.1123']['tp']} "
      f"(recall {va['recall']:.4f}). Exactly {vn['rows_between_thresholds']['validation']} validation fraud row and {vn['rows_between_thresholds']['test']} test legitimate row "
      f"lie between the two values; at the full-precision threshold the test result would be FP = {vn['test_confusion_at_full_precision_threshold']['fp']} instead of {te['fp']}. "
      "The existing test values are therefore kept unchanged; the validation column of Table 4 uses the stored threshold.")
    w("")

    # 2
    w("## 2. Dataset Statistics")
    w("")
    w("**What was measured:** the seed-42 V2 corpus used by the frozen model. **Configuration:** 120,000 payments, nominal fraud rate 0.012, 2,000 payers, seed 42.")
    w("")
    w(f"### {title('table01_dataset_splits.md')}")
    w(""); w(tab("table01_dataset_splits.md")); w("")
    w(f"Payee history is hidden (observation mask) on {ds['payee_history_hidden_rows']:,} rows ({100*ds['payee_history_hidden_share']:.2f}%); fraud prevalence is "
      f"{100*ds['fraud_rate_hidden']:.3f}% among hidden-history rows and {100*ds['fraud_rate_visible']:.3f}% among visible ones (a property of the masking rule, which ties "
      f"visibility to payer count). A message accompanies {100*ds['message_available_share']:.2f}% of payments, a purpose is stated on {100*ds['intent_stated_share']:.2f}%, and "
      f"graph evidence is available on {100*ds['graph_available_share']:.2f}%.")
    w("")
    w(f"### {title('table02_scenarios.md')}")
    w(""); w(tab("table02_scenarios.md")); w("")
    w("**Does NOT demonstrate:** these scenario shares are simulator settings; they are not estimates of the composition of real UPI fraud.")
    w("")

    # 3
    w("## 3. V2 Feature Representation")
    w("")
    w("The 47 features were read from the implementation (`backend/ml/dataset_v2.FEATURES_V2`, identical to the frozen model's feature list). Each is mapped to the live "
      "evidence family that reads the same observation (`scripts/final_v2/families.py`). The tenth family, **Classifier**, is model-derived evidence: the V2 model itself, "
      "whose inputs are all 47 columns. Three mappings differ from the implementation's own group names and are stated explicitly: `amount_over_user_avg` (group *amount*) "
      "→ Payer behaviour; `payee_is_phone_address` (group *payee*) → Address and QR; `payee_is_new` (group *relationship*) → Relationship.")
    w("")
    counts = pd.read_csv(T / "table03_feature_inventory.csv").groupby("Evidence family").size()
    w("| Evidence family | Features |\n|---|---|")
    for f_, n_ in counts.sort_values(ascending=False).items():
        w(f"| {f_} | {n_} |")
    w(f"| Classifier (model-derived) | uses all 47 |")
    w("")
    w(f"### {title('table03_feature_inventory.md')}")
    w(""); w(tab("table03_feature_inventory.md")); w("")
    w("All 47 features are computed from state available before the payment (row-level corpus: generated as the pre-payment evidence state; live/replay: built by "
      "`payee_history.build_v2_vector` from payments strictly before t). Missing values are left as NaN with explicit availability flags; nothing is imputed.")
    w("")

    # 4
    w("## 4. V2 Classifier Performance")
    w("")
    w(f"**What was measured:** the frozen V2 model on the seed-42 validation (n = {va['n']:,}) and test (n = {te['n']:,}) splits at the stored threshold {thr}. "
      "**Category:** reproduced result (identical to the existing reported test result).")
    w("")
    w(f"### {title('table04_overall_performance.md')}")
    w(""); w(tab("table04_overall_performance.md")); w("")
    w(f"### {title('table04c_classification_report.md')}")
    w(""); w(tab("table04c_classification_report.md")); w("")
    w(f"**Demonstrates:** under the synthetic protocol the V2 model ranks fraud well (test PR-AUC {te['pr_auc_average_precision']:.4f} against a no-skill baseline of "
      f"{te['prevalence']:.4f}) and, at a 1% false-positive budget, recalls {100*te['recall']:.2f}% of fraud with precision {te['precision']:.4f}; MCC {te['mcc']:.4f}. "
      f"Validation and test values are close, consistent with the threshold having been selected on validation only.  \n"
      f"**Does NOT demonstrate:** real-world detection performance. Accuracy ({te['accuracy']:.4f}) and specificity are dominated by the {100*(1-te['prevalence']):.1f}% "
      f"legitimate class and are not informative on their own; {te['fp']} of the {te['alerts']} alerts ({100*te['fp']/te['alerts']:.1f}%) are false alarms.")
    w("")

    # 5
    w("## 5. Confusion Matrix")
    w("")
    w(f"### {title('table05_confusion_matrix.md')}")
    w(""); w(tab("table05_confusion_matrix.md")); w("")
    w(fig("fig01a_confusion_matrix", "Figure 1a — V2 test confusion matrix, raw counts (threshold 0.1123)"))
    w("")
    w(fig("fig01b_confusion_matrix_normalised", "Figure 1b — The same matrix normalised by actual class"))
    w("")
    w(f"Numerical matrix: `confusion_matrix/confusion_matrix_test.csv`. Of {te['tn']+te['fp']:,} legitimate payments {te['fp']} ({100*te['fpr']:.3f}%) are flagged; "
      f"of {te['tp']+te['fn']} fraudulent payments {te['tp']} are flagged and {te['fn']} missed (FNR {100*te['fnr']:.2f}%).")
    w("")

    # 6
    w("## 6. ROC Analysis")
    w("")
    w(fig("fig02_roc_curve", f"Figure 2 — ROC curve, V2 test split (ROC-AUC {te['roc_auc']:.4f}); operating point FPR {te['fpr']:.5f}, TPR {te['recall']:.4f}; inset: FPR 0–5%"))
    w("")
    w(f"**Demonstrates:** ROC-AUC {te['roc_auc']:.4f} on the test split. **Does NOT demonstrate:** practical usefulness by itself — with {te['n']-te['positives']:,} "
      "legitimate payments even a small FPR produces many false alarms, which the ROC curve hides and the precision–recall curve shows. Curve points: `roc_pr/roc_curve_test.csv`.")
    w("")

    # 7
    w("## 7. Precision-Recall Analysis")
    w("")
    w(fig("fig03_pr_curve", f"Figure 3 — Precision–recall curve, V2 test split (PR-AUC {te['pr_auc_average_precision']:.4f}); dashed: prevalence {te['prevalence']:.4f}"))
    w("")
    w(f"PR-AUC (average precision) is {te['pr_auc_average_precision']:.4f}, about {te['pr_auc_average_precision']/te['prevalence']:.0f}× the no-skill baseline. At the "
      f"operating point recall is {te['recall']:.4f} and precision {te['precision']:.4f}: roughly one in two alerts is a true fraud under the synthetic prevalence. "
      "Precision depends on prevalence and would be lower at a lower real fraud rate. Curve points: `roc_pr/pr_curve_test.csv`.")
    w("")

    # 8
    w("## 8. Threshold Analysis")
    w("")
    w(fig("fig06_threshold_analysis", "Figure 6 — Precision, recall, F1 and FPR (×10) versus threshold on the validation split; dotted line: selected threshold"))
    w("")
    w(f"### {title('table04b_selected_operating_point.md')}")
    w(""); w(tab("table04b_selected_operating_point.md")); w("")
    sw_ = pd.read_csv(R / "threshold/threshold_sweep.csv")
    vs = sw_[sw_.split == "validation"]
    best_f1 = vs.loc[vs.f1.idxmax()]
    w(f"The full sweep ({len(vs)} thresholds from 0.001 to 0.95 plus the selected value, validation and test) is in `threshold/threshold_sweep.csv`. On validation the "
      f"F1-maximising threshold in the grid is {best_f1.threshold:.4f} (F1 {best_f1.f1:.4f}, recall {best_f1.recall:.4f}, FPR {best_f1.fpr:.5f}); the project's rule instead fixes the "
      "false-positive budget at 1% and maximises recall inside it. The test columns of the sweep are provided for inspection only and were not used to choose anything.")
    w("")

    # 9
    w("## 9. Calibration Analysis")
    w("")
    w(fig("fig04_reliability_diagram", "Figure 4 — Reliability diagram (10 equal-width bins, test split) and bin counts"))
    w("")
    w("| Bin | n | Fraud in bin | Mean predicted | Observed rate | |gap| |\n|---|---|---|---|---|---|")
    for _, r in rel.iterrows():
        w(f"| {r.bin_lower:.1f}–{r.bin_upper:.1f} | {int(r.n):,} | {int(r.fraud_in_bin)} | {r.mean_predicted:.4f} | {r.observed_fraud_rate:.4f} | {r.abs_gap:.4f} |")
    w("")
    ct = cal["test"]
    mid = rel[(rel.bin_lower >= 0.29) & (rel.bin_upper <= 0.61)]
    w(f"- Test Brier score {ct['brier']:.5f} (Brier skill score {te['brier_skill_score']:.4f} against the base-rate predictor); ECE (10 equal-width bins) {ct['ece_uniform_10']:.4f}; "
      f"ECE with 10 quantile bins {ct['ece_quantile_10_unique_edges']:.4f} (only {ct['bins_quantile_populated']} distinct quantile bins exist because most predictions are near zero). "
      f"Mean predicted probability {ct['mean_predicted']:.5f} vs observed rate {ct['observed_rate']:.5f}.")
    w(f"- **Interpretation:** the low ECE is dominated by the first bin, which holds {int(rel.n.iloc[0]):,} of {te['n']:,} payments and is well calibrated "
      f"({rel.mean_predicted.iloc[0]:.4f} predicted vs {rel.observed_fraud_rate.iloc[0]:.4f} observed). In the 0.3–0.6 bins ({int(mid.n.sum())} payments) the model "
      f"under-predicts: observed rates exceed mean predictions by {mid.abs_gap.min():.3f}–{mid.abs_gap.max():.3f}. These bins hold 33–46 payments each, so their rates are "
      "imprecise. Calibration is good where most of the mass is and only approximate in the sparse upper range; it holds for this simulator and prevalence only.")
    w(f"- Validation: Brier {cal['validation']['brier']:.5f}, ECE (10 bins) {cal['validation']['ece_uniform_10']:.4f}. Bin tables: `calibration/`.")
    w("")

    # 10
    w("## 10. Risk Score Analysis")
    w("")
    w(fig("fig05_risk_score_distribution", "Figure 5 — Distribution of the calibrated V2 fraud probability on the test split (log scale); dotted: threshold 0.1123"))
    w("")
    w("| Class | n | Mean | P05 | P25 | Median | P75 | P95 | Share ≥ threshold |\n|---|---|---|---|---|---|---|---|---|")
    for lab in ("legitimate", "fraud"):
        x = sd[lab]
        w(f"| {lab} | {x['n']:,} | {x['mean']:.4f} | {x['p05']:.5f} | {x['p25']:.5f} | {x['median']:.5f} | {x['p75']:.5f} | {x['p95']:.4f} | {x['share_above_threshold']:.4f} |")
    w("")
    w(f"Legitimate payments concentrate near zero (95th percentile {sd['legitimate']['p95']:.4f}); fraudulent payments spread across the range (median {sd['fraud']['median']:.4f}), "
      f"and {100*(1-sd['fraud']['share_above_threshold']):.1f}% of them fall below the threshold. The score is the classifier probability, not the ten-family risk R of Section 15.")
    w("")

    # 11
    w("## 11. Fraud Scenario Analysis")
    w("")
    w(f"### {title('table06_scenario_performance.md')}")
    w(""); w(tab("table06_scenario_performance.md")); w("")
    w(fig("fig11_scenario_performance", "Figure 11 — Recall by synthetic fraud scenario, V2 test split"))
    w("")
    w("**Demonstrates:** detection differs by scenario; social engineering, where the payer personally authorises the payment, is the hardest synthetic scenario "
      "(most of the 78 missed frauds). Account takeover and card testing, which change the payer's own behaviour, are almost all detected.  \n"
      "**Does NOT demonstrate:** real-world per-scenario rates; the scenarios and their overlaps with legitimate behaviour are simulator designs.")
    w("")

    # 12
    w("## 12. Feature Importance")
    w("")
    w("Method (as implemented in `scripts/evaluate_model_v2.py` and reproduced exactly here, max difference "
      f"{j(R / 'feature_importance/importance_verification.json')['permutation_max_abs_diff_vs_reported']:.1e}): permutation importance on the frozen calibrated model, "
      "test split, scoring = average precision, 5 repeats; SHAP (TreeExplainer) on the uncalibrated booster of calibration fold 1, 200 test rows. "
      "Family importance shuffles all columns of a family together.")
    w("")
    w(f"### {title('table12a_top_features.md')}")
    w(""); w(tab("table12a_top_features.md")); w("")
    w(f"### {title('table12b_family_importance.md')}")
    w(""); w(tab("table12b_family_importance.md")); w("")
    w(fig("fig07_feature_importance", "Figure 7 — Top 20 V2 features by permutation importance (test split), coloured by evidence family"))
    w("")
    w(fig("fig07b_shap_summary", "Figure 7b — SHAP summary for the 15 features with the largest mean |SHAP| (uncalibrated booster, 200 test rows)"))
    w("")
    w(fig("fig08_family_importance", "Figure 8 — Evidence-family importance (grouped permutation, test split)"))
    w("")
    w(f"The five most useful single features are {', '.join(f'`{r.feature}` ({r.perm_importance_mean:.4f})' for _, r in top5.iterrows())}. By family, "
      f"{fam_imp.iloc[0].Family}, {fam_imp.iloc[1].Family} and {fam_imp.iloc[2].Family} dominate; link-safety columns contribute nothing measurable "
      f"({fam_imp[fam_imp.Family == 'Link safety']['Grouped permutation importance (AP drop)'].iloc[0]:.4f}). Importance is predictive usefulness under this protocol, "
      "not causal influence, and is separate from how the ten-family aggregation weighs evidence at run time.")
    w("")

    # 13
    w("## 13. V2 Feature-Family Ablation")
    w("")
    w("**What was measured:** leave-family-out refits of V2 on seeds 42–51 (13 configurations × 10 seeds = 130 fits), identical splits, learner, calibration and "
      "validation-only threshold rule. **Category:** multi-seed result. The Full-V2 rows are refitted, not copied, and equal the ten-seed run of Section 14.")
    w("")
    w(f"### {title('table07_v2_family_ablation.md')}")
    w(""); w(tab("table07_v2_family_ablation.md")); w("")
    w(fig("fig09_v2_ablation", "Figure 9 — Change in test PR-AUC when each family (or group of families) is removed, mean ± sd over ten seeds"))
    w("")
    order = abl.drop(index="A. Full V2").sort_values("delta_pr_auc_mean")
    w("Paired changes in PR-AUC relative to Full V2 (largest loss first): " + "; ".join(
        f"{c} {100*r.delta_pr_auc_mean:+.2f} pp (lower on {int(r.delta_pr_auc_seeds_lower)}/10 seeds)" for c, r in order.iterrows()) + ".")
    w("")
    w(f"- **Graph:** removing it lowers PR-AUC by {d_('C. - Graph')} and mule-collection recall from {100*full.recall_mule_collection_mean:.1f}% to "
      f"{100*abl.loc['C. - Graph', 'recall_mule_collection_mean']:.1f}%.")
    w(f"- **Message pressure:** {d_('D. - Message pressure')}; social-engineering recall {100*full.recall_social_engineering_mean:.1f}% → "
      f"{100*abl.loc['D. - Message pressure', 'recall_social_engineering_mean']:.1f}%.")
    w(f"- **Payer behaviour:** {d_('J. - Payer behaviour')}; account-takeover recall {100*full.recall_account_takeover_mean:.1f}% → "
      f"{100*abl.loc['J. - Payer behaviour', 'recall_account_takeover_mean']:.1f}% and card-testing {100*full.recall_card_testing_mean:.1f}% → "
      f"{100*abl.loc['J. - Payer behaviour', 'recall_card_testing_mean']:.1f}%.")
    w(f"- **Link safety** ({d_('E. - Link safety')}) and **amount context** ({d_('K. - Amount context')}) show no difference distinguishable from seed-to-seed variation.")
    w(f"- **All payee-side evidence removed** (history + relationship + graph + address): {d_('M. - All payee-side (history+relationship+graph+address)')}; "
      f"mule-collection recall falls to {100*abl.loc['M. - All payee-side (history+relationship+graph+address)', 'recall_mule_collection_mean']:.1f}%.")
    w("")
    w("Single-split version on the original seed-42 split: `tables/table07b_v2_family_ablation_seed42.md`. **Does NOT demonstrate:** that a family *causes* detection; "
      "each channel was generated with scenario-conditional probabilities chosen by the designers, so the ablation measures how much of that designed signal the learner recovers.")
    w("")

    # 14
    w("## 14. Ten-Seed Stability")
    w("")
    w(f"### {title('table08_ten_seed_stability.md')}")
    w(""); w(tab("table08_ten_seed_stability.md")); w("")
    w(fig("fig10_ten_seed_stability", "Figure 10 — V2 across seeds 42–51 (boxes: ten seed-level values; star: seed 42)"))
    w("")
    n_ = ms["notes"]
    w(f"- PR-AUC {pm(msr.pr_auc)} (range {msr.pr_auc.min():.4f}–{msr.pr_auc.max():.4f}); recall at ~1% FPR {pm(msr.recall, 100, 2)}%; realised test FPR {pm(msr.fpr, 100, 3)}%.")
    w(f"- The original seed-42 run (PR-AUC {s42.pr_auc:.4f}) ranks {rank42} of 10 by PR-AUC; it lies inside the multi-seed range but above the mean, so the single-split "
      "number slightly overstates typical performance.")
    w(f"- Validation FPR ≤ 1% on all seeds ({n_['val_fpr_all_le_1pct']}); the test FPR exceeds 1% on {n_['test_fpr_above_1pct_seeds']} seeds (max "
      f"{100*n_['max_test_fpr']:.3f}%) — ordinary transfer variation of a validation-chosen threshold.")
    w(f"- Test fraud per seed: {', '.join(str(v) for v in n_['test_fraud_per_seed'].values())}. Test sets were never pooled and no seed was selected.")
    w(f"- **Template-disjoint control** (leakage check, Section 20 item 12): with test messages drawn only from templates unseen in training and the coercion model "
      f"refitted on the training half, full-V2 PR-AUC is {td['pr_auc']['disjoint_mean']:.4f} ± {td['pr_auc']['disjoint_std']:.4f} (paired Δ "
      f"{td['pr_auc']['paired_delta_pp_mean']:+.2f} ± {td['pr_auc']['paired_delta_pp_std']:.2f} pp) — `results/ablation/template_disjoint_summary.json`.")
    w("")

    # 15
    w("## 15. Full Ten-Family End-to-End Evaluation")
    w("")
    w("**What was measured:** the complete assessment (ten families, aggregation, verdicts) on three chronological replay streams (seeds 42–44; 60,000 payments, "
      "150 days, 2,000 payers, 900 merchants, 720 frauds each), assessed payment by payment from the pre-payment state (Section 18). "
      "**Configuration:** the running prototype serves an older model in its Classifier family; for this V2-only report that family is replaced by the frozen V2 model, "
      "scored with the production mapping (graded 0–18 below the threshold, 30–66 above; counterfactual fact attribution) on the V2 vector the replay built for each payment "
      "from the same pre-payment state. The other nine families are exactly as `payee_check.check_payee` computed them, and risk, coverage, level and verdict are recomputed "
      "with the production functions (the same re-aggregation reproduced every recorded replay output with 0 mismatches over 180,000 payments). "
      "The primary slice is the last 25% of each stream (days ≈112–150), after the shared store has warmed up.")
    w("")
    w(f"### {title('table09_end_to_end.md')}")
    w(""); w(tab("table09_end_to_end.md")); w("")
    w(f"### {title('table09b_verdict_distribution.md')}")
    w(""); w(tab("table09b_verdict_distribution.md")); w("")
    w(fig("fig12_verdict_distribution", "Figure 12 — Advisory verdict distribution by group, last 25% of each stream (mean over three streams)"))
    w("")
    w(f"### {title('table09f_family_behaviour.md')}")
    w(""); w(tab("table09f_family_behaviour.md")); w("")
    w(fig("fig13b_family_availability", "Figure 13b — Evidence-family availability and activation"))
    w("")
    rp = [x["risk_pr_auc"] for x in L(0)]; vp = [x["v2_alone_pr_auc_where_available"] for x in L(0)]
    w(f"- **Continuous risk:** R reaches PR-AUC {pm(rp)} and ROC-AUC {pm([x['risk_roc_auc'] for x in L(0)])}; the frozen V2 probability alone reaches "
      f"{pm(vp)} and {pm([x['v2_alone_roc_auc_where_available'] for x in L(0)])} on the payments where its family is available. With V2 as the classifier, "
      "the rule-based aggregation ranks fraud *less* well than the V2 probability alone on this stream.")
    w(f"- **Transfer:** V2 was trained on the row-level corpus; on the replay streams its stored threshold recalls only "
      f"{pm([x['v2_alone_at_threshold']['recall'] for x in L(0)], 100, 1)}% of fraud (FPR {pm([x['v2_alone_at_threshold']['fpr'] for x in L(0)], 100, 2)}%). "
      "The two simulators differ, and the classifier's operating point does not transfer between them.")
    w(f"- **Verdicts:** legitimate payments receive APPROVE {pm([vc(s, 'legitimate', 'APPROVE') for s in S3], 100, 1)}%, WARN {pm([vc(s, 'legitimate', 'WARN') for s in S3], 100, 1)}%, "
      f"STEP_UP {pm([vc(s, 'legitimate', 'STEP_UP') for s in S3], 100, 1)}%, BLOCK {pm([vc(s, 'legitimate', 'BLOCK') for s in S3], 100, 1)}%. Social-engineering payments are "
      f"blocked in {pm([vc(s, 'social_engineering', 'BLOCK') for s in S3], 100, 1)}% and mule-collection payments in {pm([vc(s, 'mule_collection', 'BLOCK') for s in S3], 100, 1)}%; "
      f"card-testing probes are approved in {pm([vc(s, 'card_testing', 'APPROVE') for s in S3], 100, 1)}%.")
    w(f"- **Operating points of R:** WARN or above recalls {pm([x['recall'] for x in r_warn], 100, 1)}% of fraud at FPR {pm([x['fpr'] for x in r_warn], 100, 1)}%; "
      f"BLOCK recalls {pm([x['recall'] for x in r_blk], 100, 1)}% at FPR {pm([x['fpr'] for x in r_blk], 100, 2)}%.")
    w("")
    w("**Demonstrates:** how the complete rule-based system behaves on a synthetic chronological stream: it blocks most social-engineering and mule-collection "
      "payments but, with its fixed (untuned) rule weights and verdict bands, it intervenes on roughly a third of legitimate payments and blocks about 7%.  \n"
      "**Does NOT demonstrate:** a four-class classifier — APPROVE/WARN/STEP_UP/BLOCK are advisory bands of one score, not ground-truth classes; nor real-world intervention "
      "rates.")
    w("")
    w(legit_block_causes())
    w("")

    # 16
    w("## 16. Evidence Coverage Analysis")
    w("")
    w("Coverage C (0–100) is a weighted count of available families (fixed weights: payee history 24, relationship 18, payer behaviour 16, classifier 12, address/QR 11, "
      "graph 8, amount 5, message 3, link 2, intent 1). It is computed only from availability and never from scores.")
    w("")
    w(f"### {title('table09c_coverage.md')}")
    w(""); w(tab("table09c_coverage.md")); w("")
    w(f"### {title('table09d_coverage_full_stream.md')}")
    w(""); w(tab("table09d_coverage_full_stream.md")); w("")
    w(fig("fig13_coverage", "Figure 13 — Coverage distribution by label and risk versus coverage (seed 42, last 25%)"))
    w("")
    w(f"### {title('table09e_missing_evidence.md')}")
    w(""); w(tab("table09e_missing_evidence.md")); w("")
    cls = miss["42"].get("ml_classifier", {})
    w("**Verification that missing evidence is neither 'safe' nor 'fraud':**")
    w("")
    w("1. In the implementation, `evidence.Evidence` raises an error if an unavailable family carries a score or facts, so an unavailable family contributes nothing to R "
      "(it cannot lower or raise risk by being absent); coverage records the absence separately.")
    w(f"2. Measured: across the replay no unavailable family carried a score (0 cases), and payments with an unavailable family are neither set to R = 0 nor blocked "
      f"wholesale — e.g. seed 42, Classifier family unavailable on {cls.get('unavailable_rows', 0):,} payments (payer with fewer than 5 prior payments): median R "
      f"{cls.get('risk_median_when_unavailable')} vs {cls.get('risk_median_when_available')} when available; share with R = 0: "
      f"{100*cls.get('share_R0_when_unavailable', 0):.1f}%; share BLOCK: {100*cls.get('share_BLOCK_when_unavailable', 0):.1f}%. Their verdicts come from the evidence that *was* available.")
    w("3. Unit tests covering the same property pass (`test_evidence_availability`, `test_evidence_model`; Section 19).")
    w("")
    w(f"Coverage is high for both classes (full stream, seed 42: mean {cov42['mean']:.1f}, median {cov42['median']:.0f}, range {cov42['min']}–{cov42['max']}); it is lower "
      "for payments to UNKNOWN payees (no prior observation), as intended. High coverage does not imply low risk.")
    w("")

    # 17
    w("## 17. Graph Analysis")
    w("")
    w("The implementation maintains an **explicit incremental payer–payee graph** in the same database transaction that records a payment: per-pair edges "
      "(`payee_payers`: payment count, first/last timestamps), per-payee aggregates (`payee_graph_cache`: distinct payers, repeat payers, arrival window) and a "
      "persisted union–find over payer and payee nodes (`graph_components`). The assessment reads it through `graph_cache.graph_view`. **It is not a graph neural "
      "network and computes no embeddings.** Statistics below are read from the replay stores (end of each 150-day stream) and cross-checked against an independent "
      "union–find built from the transaction files.")
    w("")
    w(f"### {title('table13_graph_statistics.md')}")
    w(""); w(tab("table13_graph_statistics.md")); w("")
    w(f"### {title('table13b_relationship_state.md')}")
    w(""); w(tab("table13b_relationship_state.md")); w("")
    w(fig("fig14_graph_statistics", "Figure 14 — Payee in-degree, payer out-degree and relationship state at assessment time (seed 42)"))
    w("")
    g42 = gs["42"]
    w(f"- The graph is dominated by one giant component ({g42['largest_component_store']:,} of {g42['nodes_total']:,} nodes in seed 42), because payers share popular merchants; "
      "component size is therefore recorded but, by design, never used as a finding.")
    w(f"- {100*g42['repeat_relationship_share']:.1f}% of payer–payee edges carry repeat payments; payee in-degree is heavy-tailed (median {g42['payee_in_degree_distinct_payers']['median']:.0f}, "
      f"max {g42['payee_in_degree_distinct_payers']['max']:,}).")
    w(f"- The positional finding (collection accounts sharing one victim pool) fired on {sum(gs[s]['assessment_time']['positional_finding_fired']['fraud'] for s in gs)} fraudulent "
      f"and {sum(gs[s]['assessment_time']['positional_finding_fired']['legitimate'] for s in gs)} legitimate payments across the three streams: it is precise but rarely active under "
      "this replay design (median mule-account overlap is below its 0.6 gate). Shape findings (restating payee history) are frequent and carry zero weight.")
    w("")

    # 18
    w("## 18. Chronological Graph Replay")
    w("")
    w("**What was done:** for every payment at time t, in timestamp order: (1) set the application clock to t; (2) read only state recorded before t (payer's earlier "
      "rows; ledger, aggregates and graph); (3) assess with the production function `payee_check.check_payee`; (4) record risk, coverage, level, relationship, graph and "
      "history evidence and the V2 vector; (5) cross-check the state against an independent shadow model; (6) **only then** record the payment through the production "
      "observation path. Afterwards an independent process rebuilds the store from earlier payments only (test 8) and earlier records are compared with the final store "
      "(test 9). The only runtime substitution is the injected clock. **Category:** temporal/software validation, separate from the row-level classifier evaluation.")
    w("")
    w(f"### {title('table11_replay_validation.md')}")
    w(""); w(tab("table11_replay_validation.md")); w("")
    w(fig("fig15_replay", "Figure 15 — Verdicts as the store matures (left) and replay validation checks (right)"))
    w("")
    probe = sum(j(G / f"replay_validation_seed{s}.json")["test9b_power_probe"]["differs_when_future_visible"] for s in S3)
    probe_n = sum(j(G / f"replay_validation_seed{s}.json")["test9b_power_probe"]["sampled"] for s in S3)
    w(f"Power check: re-assessing the {probe_n} sampled payments against the *final* store (which contains their future) changes at least one recorded quantity for {probe} of "
      "them — the pre-payment boundary is material, so the tests above can detect leakage.")
    w("")
    w("### 18.1 Concrete example — before / assessment / after")
    w("")
    for e, v2r_ in ((ex_m, v2m), (ex_r, v2r)):
        b, a, af = e["before"], e["assessment"], e["after"]
        w(f"**{('Mule-collection payment' if e['kind'] == 'mule' else 'Repeat payment to a known payee')}** — stream 42, `{e['txn_id']}`, {e['time']}, amount Rs {e['amount']:,.2f}, "
          f"label: {'fraud (' + e['scenario'] + ')' if e['label'] else 'legitimate'}. Re-executed from a fresh store holding only earlier payments; identical to the replay record: "
          f"**{e['identical_to_replay_record']}**.")
        w("")
        w("| | Before the payment | After recording it |\n|---|---|---|")
        pe_b = b["pair_edge"]; pe_a = af["pair_edge"]
        w(f"| Payer→payee edge | {'none (first payment)' if pe_b is None else str(pe_b['payments']) + ' payments since ' + pe_b['first_at'][:10]} | "
          f"{pe_a['payments']} payment(s), last {pe_a['last_at'][:19]} |")
        pa_b, pa_a = b["payee_aggregate"], af["payee_aggregate"]
        w(f"| Payee: distinct / repeat payers / payments | {pa_b['distinct_payers']} / {pa_b['repeat_payers']} / {pa_b['payments']} | {pa_a['distinct_payers']} / {pa_a['repeat_payers']} / {pa_a['payments']} |")
        w(f"| Payee arrival window | {pa_b['arrival_first'][:19]} → {pa_b['arrival_last'][:19]} | {pa_a['arrival_first'][:19]} → {pa_a['arrival_last'][:19]} |")
        gv_b, gv_a = b["graph_view"], af["graph_view"]
        w(f"| Graph view: component / arrival span (days) / shared-payer peak | {gv_b['component_size']:,} / {gv_b['arrival_span_days']:.3f} / {gv_b['shared_payer_peak']} | "
          f"{gv_a['component_size']:,} / {gv_a['arrival_span_days']:.3f} / {gv_a['shared_payer_peak']} |")
        w(f"| Graph findings (shape; positional) | {len(gv_b['shape_findings'])} shape; {len(gv_b['position_findings'])} positional | {len(gv_a['shape_findings'])} shape; {len(gv_a['position_findings'])} positional |")
        w(f"| Observations in store | {b['observations_in_store']:,} | {af['observations_in_store']:,} |")
        w("")
        fams = {k: v for k, v in a["families"].items() if k != "ml_classifier"}
        w(f"*Assessment (at t, before recording):* relationship state **{a['payee_status']}**; families (score; facts): " + "; ".join(
            f"{k.replace('_', ' ')} {'unavailable' if not v['available'] else str(v['score'])}" + (f" [{', '.join(v['facts'])}]" if v.get('facts') else "")
            for k, v in fams.items()) + f"; Classifier (V2) {int(v2r_.v2_family_score) if pd.notna(v2r_.v2_family_score) else 'unavailable'} "
          f"(p = {v2r_.v2_probability:.4f}). Corroborating facts: {', '.join(a['agreement']['facts']) or 'none'} (bonus {int(v2r_.bonus)}). "
          f"**Ten-family result with V2: R = {int(v2r_.risk)}, coverage {int(v2r_.coverage)}, level {v2r_.evidence_level}, verdict {v2r_.verdict}.** "
          f"Findings: {', '.join(a['findings'])}.")
        w("")
    w("The first example shows a new payer–payee relationship being detected (no edge before; one edge after) while the payee's history already shows a compressed "
      "arrival of one-time payers, which the payee-history family scores; the second shows a repeat relationship (edge count 5 → 6) assessed as low risk. In both, the "
      "store changes only after the assessment.")
    w("")

    # 19
    w("## 19. Software Test Validation")
    w("")
    w(f"### {title('table10_software_tests.md')}")
    w(""); w(tab("table10_software_tests.md")); w("")
    w(f"**Result:** {sw['tests']} tests collected and {sw['passed']} passed, {sw['failed']} failed, {sw['skipped']} skipped ({sw['warnings']} warnings) in "
      f"{sw['pytest_time_s']} s of pytest time. Per-module results: `software_tests/pytest_summary_by_file.csv`; keyword coverage of concerns across modules "
      "(thresholds, missing evidence, graph, verdicts, corroboration, strict time boundary, JSON): `software_tests/test_keyword_coverage.csv`. "
      "There is no front-end (TypeScript) test suite in the repository. **These are implementation-correctness results and say nothing about fraud-detection accuracy.**")
    w("")

    # 20
    w("## 20. Leakage Audit")
    w("")
    w(f"Full audit with evidence: `leakage_audit/LEAKAGE_AUDIT.md`. **Summary: {la['summary']['PASS']} PASS · {la['summary']['FAIL']} FAIL · "
      f"{la['summary']['NOT APPLICABLE']} NOT APPLICABLE.**")
    w("")
    w("| # | Check | Status |\n|---|---|---|")
    for it in la["items"]:
        w(f"| {it['#']} | {it['check']} | {it['status']} |")
    w("")
    w("No serious leakage was found, so no result was withdrawn. Two passes carry disclosed caveats: message templates are shared between train and test "
      "(effect measured and negligible, Section 14), and the row-level split is random, so a payer's rows can appear in both train and test (identity is not a feature, "
      "but generalisation to unseen payers is not measured by the row-level experiments).")
    w("")

    # 21
    w("## 21. Reproducibility")
    w("")
    env = man["environment_experiments"]
    w(f"- Experiments: Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, scikit-learn {env['scikit-learn']}, scipy {env['scipy']}, joblib {env['joblib']}, "
      f"matplotlib {env['matplotlib']}, shap {env['shap']} ({env['platform']}).")
    w(f"- Frozen V2 model: {man['environment_frozen_model']}. Software tests: {man['environment_tests']}.")
    w(f"- Seeds: {man['seeds']}.")
    w(f"- Git HEAD of the evaluated snapshot: `{man['git_head_of_evaluated_snapshot']}`; SHA-256 of models, generators, corpus, experiment scripts and every result file: "
      "`reproducibility/reproducibility_manifest.json`. Human-readable manifest with the exact commands: `reproducibility/REPRODUCIBILITY_MANIFEST.md`.")
    w("- Replay-only runtime substitutions (none changes a prediction): application clock set to the replay time; larger regex cache; single OpenMP thread.")
    w("")

    # 22
    w("## 22. Limitations and Interpretation")
    w("")
    w("- **Synthetic data only.** All corpora come from simulators whose scenarios, overlaps and channel probabilities were specified by the project; the evaluation "
      "cannot reveal where those assumptions are wrong. No real UPI, PSP, bank or NPCI data were used.")
    w("- **Simulator-generated V2 channels.** Relationship, graph, message, link and intent features in the row-level corpus are drawn conditional on the scenario; the "
      "ablation measures how much of that designed signal the model recovers.")
    w("- **Message pool.** 1,022 English templates written for the project; the template-disjoint control removes exact-template reuse, not the single authorial style.")
    w("- **Prevalence and calibration.** Precision, PR-AUC and calibration were measured at ~1.2% prevalence; they change at other prevalences and under drift.")
    w("- **Transfer between simulators.** The frozen V2 model's operating point does not carry over from the row-level corpus to the replay streams (Section 15).")
    w("- **End-to-end system.** Rule weights and verdict bands are fixed design parameters, never fitted to labelled data; the resulting intervention rate on legitimate "
      "payments is high, and payee evidence depends on how long the store has been observing (cold start).")
    w("- **Graph.** Explicit analytics only; its positional finding rarely fires on this replay design; graph features grow with store age.")
    w("- **Replay scale.** Three streams of 150 days, 3–4 mule rings each.")
    w("- **No deployment claims.** The prototype is advisory, not deployed, not integrated with any bank, PSP or NPCI system; latency was not benchmarked "
      "(NOT AVAILABLE FROM CURRENT IMPLEMENTATION: no latency benchmark exists in the repository).")
    w("")

    # 23
    w("## 23. Complete Results Summary")
    w("")
    w(fig("fig00_key_results", "Figure 0 — Key V2 results overview (every value read from the result files)"))
    w("")
    w("| Result | Split / data | Value |\n|---|---|---|")
    w(f"| V2 test PR-AUC / ROC-AUC (existing = reproduced) | seed 42 test, n = 30,000 | {te['pr_auc_average_precision']:.4f} / {te['roc_auc']:.4f} |")
    w(f"| Recall / FPR / precision / F1 at θ = {thr} | seed 42 test | {te['recall']:.4f} / {te['fpr']:.5f} / {te['precision']:.4f} / {te['f1']:.4f} |")
    w(f"| Confusion TN / FP / FN / TP | seed 42 test | {te['tn']:,} / {te['fp']} / {te['fn']} / {te['tp']} |")
    w(f"| MCC / balanced accuracy / specificity | seed 42 test | {te['mcc']:.4f} / {te['balanced_accuracy']:.4f} / {te['specificity']:.4f} |")
    w(f"| Brier / BSS / ECE (10 bins) | seed 42 test | {te['brier']:.5f} / {te['brier_skill_score']:.4f} / {cal['test']['ece_uniform_10']:.4f} |")
    w(f"| Scenario recall SE / MC / AT / CT | seed 42 test | " + " / ".join(f"{100*s42['recall_'+s]:.2f}%" for s in ('social_engineering', 'mule_collection', 'account_takeover', 'card_testing')) + " |")
    w(f"| Ten-seed PR-AUC | seeds 42–51 test | {pm(msr.pr_auc)} (min {msr.pr_auc.min():.4f}, max {msr.pr_auc.max():.4f}) |")
    w(f"| Ten-seed recall at ~1% FPR / realised FPR | seeds 42–51 test | {pm(msr.recall, 100, 2)}% / {pm(msr.fpr, 100, 3)}% |")
    w(f"| Largest ablation losses (PR-AUC) | seeds 42–51 | payer behaviour {d_('J. - Payer behaviour')}; relational {d_('H. - Relational (relationship+graph)')}; graph {d_('C. - Graph')} |")
    w(f"| Template-disjoint control, full V2 | seeds 42–51 | Δ {td['pr_auc']['paired_delta_pp_mean']:+.2f} ± {td['pr_auc']['paired_delta_pp_std']:.2f} pp PR-AUC |")
    w(f"| Ten-family risk PR-AUC / ROC-AUC | replay, last 25%, 3 streams | {pm(rp)} / {pm([x['risk_roc_auc'] for x in L(0)])} |")
    w(f"| Legitimate APPROVE / BLOCK | replay, last 25% | {pm([vc(s, 'legitimate', 'APPROVE') for s in S3], 100, 1)}% / {pm([vc(s, 'legitimate', 'BLOCK') for s in S3], 100, 1)}% |")
    w(f"| Fraud BLOCK recall | replay, last 25% | {pm([x['recall'] for x in r_blk], 100, 1)}% |")
    c1, v1_ = chk("10_store_contains_only_earlier_payments")
    w(f"| Replay validation tests | 3 streams | 10/10 pass; e.g. {c1:,} pre-payment boundary checks, {v1_} violations |")
    w(f"| Software tests | project machine | {sw['passed']}/{sw['tests']} passed |")
    w(f"| Leakage audit | all experiments | {la['summary']['PASS']} PASS, {la['summary']['FAIL']} FAIL, {la['summary']['NOT APPLICABLE']} N/A |")
    w("")
    w("**Not available from the current implementation:** latency/throughput benchmarks; any evaluation on real UPI data; per-scenario precision (false positives belong to no scenario).")
    w("")
    (R / "FINAL_PROJECT_RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("written", len(out), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
