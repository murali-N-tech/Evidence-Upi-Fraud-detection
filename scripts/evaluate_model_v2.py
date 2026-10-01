"""The one evaluation of frozen V2 on the untouched test split.

V2 was chosen on validation - the model family, the feature set and the
operating threshold - and written to models/risk_model_v2.pkl before this
script ran. Nothing here selects anything. It rebuilds the same split v1 used,
scores the frozen artefact on the 25% of rows no development decision has
touched, and writes the result beside v1's rather than over it.

    evaluation/      v1, untouched
    evaluation_v2/   v2

Phases, so each stays inside a sane wall clock:

    python scripts/evaluate_model_v2.py --phase main     # the frozen model
    python scripts/evaluate_model_v2.py --phase tables   # ablation + baselines
    python scripts/evaluate_model_v2.py --phase compare  # v1 vs v2
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib                                                        # noqa: E402
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import scipy                                                         # noqa: E402
import sklearn                                                       # noqa: E402
from sklearn.calibration import calibration_curve                    # noqa: E402
from sklearn.inspection import permutation_importance                # noqa: E402
from sklearn.metrics import (                                        # noqa: E402
    average_precision_score, brier_score_loss, classification_report,
    precision_recall_curve, roc_auc_score, roc_curve,
)

# v1's evaluation module, imported read-only for its figures and metric helpers
# so the two reports are drawn by one implementation and stay comparable.
from scripts import evaluate_model as v1eval                         # noqa: E402
from backend.ml.dataset import LABEL                                 # noqa: E402
from backend.ml.dataset_v2 import (                                  # noqa: E402
    FEATURES_V1, FEATURES_V2, FEATURE_GROUPS, features_for, load_or_generate_v2,
)
from scripts.train_model_v2 import (                                 # noqa: E402
    ABLATION, candidates, recall_at_fpr, splits,
)

MODELS = ROOT / "models"
MODEL_V2 = MODELS / "risk_model_v2.pkl"
FROZEN = MODELS / "metrics_v2.json"
OUT = ROOT / "evaluation_v2"
V1_METRICS = ROOT / "evaluation" / "metrics.json"


def load_frozen():
    if not MODEL_V2.exists() or not FROZEN.exists():
        raise SystemExit("V2 is not frozen yet. Run scripts/train_model_v2.py --phase freeze")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert frozen["test_rows_used_in_any_decision"] is False
    assert frozen["validation_results"]["threshold_selected_on"] == "validation split"
    return joblib.load(MODEL_V2), frozen


def phase_main(args) -> int:
    started = datetime.now(timezone.utc)
    (OUT / "csv").mkdir(parents=True, exist_ok=True)
    model, frozen = load_frozen()
    columns = frozen["features"]
    threshold = float(frozen["validation_results"]["threshold"])

    df, fingerprint = load_or_generate_v2(seed=args.seed)
    y, idx_train, idx_val, idx_test = splits(df)
    X_test, y_test = df[columns].iloc[idx_test], y[idx_test]
    scen_test = df["scenario"].iloc[idx_test].to_numpy()

    print(f"\n  V2 on the held-out test split - {len(idx_test):,} rows, "
          f"{int(y_test.sum())} fraud, prevalence {y_test.mean():.5f}")
    print(f"  model: {frozen['model']} · {len(columns)} features · "
          f"threshold {threshold:.4f} (chosen on validation)\n")

    probs = model.predict_proba(X_test)[:, 1]
    summary = v1eval.evaluate(y_test, probs, threshold)
    base_rate = float(y_test.mean())
    reference = float(np.mean((y_test - base_rate) ** 2))
    summary["brier_reference_constant_base_rate"] = reference
    summary["brier_skill_score"] = float(1 - summary["brier"] / reference)

    prec, rec, pr_thresh = precision_recall_curve(y_test, probs)
    fpr, tpr, roc_thresh = roc_curve(y_test, probs)
    pred = (probs >= threshold).astype(int)
    report = classification_report(y_test, pred, labels=[0, 1],
                                   target_names=["legitimate", "fraud"],
                                   output_dict=True, zero_division=0)
    grid = np.unique(np.concatenate([
        np.round(np.logspace(np.log10(0.001), np.log10(0.95), 60), 6), [threshold, 0.5]]))
    df_sweep = v1eval.sweep(y_test, probs, grid)
    df_scen = v1eval.scenario_recall(scen_test, y_test, probs, threshold)

    edges = np.unique(np.quantile(probs, np.linspace(0, 1, args.bins + 1)))
    rows_cal = []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        mask = (probs >= left) & ((probs <= right) if last else (probs < right))
        if int(mask.sum()) == 0:
            continue
        rows_cal.append({"bin": len(rows_cal) + 1, "bin_lower": float(left),
                         "bin_upper": float(right), "n": int(mask.sum()),
                         "fraud_in_bin": int(y_test[mask].sum()),
                         "mean_predicted_probability": float(probs[mask].mean()),
                         "observed_fraud_rate": float(y_test[mask].mean())})
    df_cal = pd.DataFrame(rows_cal)
    uni = np.linspace(0.0, float(probs.max()), args.bins + 1)
    rows_uni = []
    for i in range(len(uni) - 1):
        left, right = uni[i], uni[i + 1]
        last = i == len(uni) - 2
        mask = (probs >= left) & ((probs <= right) if last else (probs < right))
        if int(mask.sum()) == 0:
            continue
        rows_uni.append({"bin": len(rows_uni) + 1, "bin_lower": float(left),
                         "bin_upper": float(right), "n": int(mask.sum()),
                         "fraud_in_bin": int(y_test[mask].sum()),
                         "mean_predicted_probability": float(probs[mask].mean()),
                         "observed_fraud_rate": float(y_test[mask].mean())})
    df_cal_uniform = pd.DataFrame(rows_uni)

    print("  permutation importance (average-precision drop, 5 repeats)")
    perm = permutation_importance(model, X_test, y_test, scoring="average_precision",
                                  n_repeats=5, random_state=args.seed, n_jobs=1)
    df_perm = pd.DataFrame({"feature": columns,
                            "importance_mean_drop_in_average_precision": perm.importances_mean,
                            "importance_std": perm.importances_std})

    df_shap, shap_note = None, "disabled"
    if args.shap_rows > 0:
        import shap
        rng = np.random.default_rng(args.seed)
        take = np.sort(rng.choice(len(X_test), size=min(args.shap_rows, len(X_test)),
                                  replace=False))
        sample = X_test.iloc[take]
        explainer = shap.TreeExplainer(model.calibrated_classifiers_[0].estimator)
        values = explainer.shap_values(sample)
        if isinstance(values, list):
            values = values[1]
        shap_note = "TreeExplainer on the first fold's uncalibrated booster"
        df_shap = pd.DataFrame({"feature": columns,
                                "mean_abs_shap": np.abs(values).mean(axis=0),
                                "mean_shap": values.mean(axis=0)})
        print(f"  shap on {len(sample)} test rows")

    csv = OUT / "csv"
    pd.DataFrame({"recall": rec, "precision": prec,
                  "threshold": np.append(pr_thresh, np.nan)}).to_csv(csv / "pr_curve.csv", index=False)
    pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr,
                  "threshold": roc_thresh}).to_csv(csv / "roc_curve.csv", index=False)
    pd.DataFrame({"false_positive_rate": fpr, "recall": tpr, "threshold": roc_thresh,
                  "within_1pct_budget": fpr <= args.fpr_budget}).to_csv(
        csv / "recall_vs_fpr.csv", index=False)
    df_sweep.to_csv(csv / "threshold_sweep.csv", index=False)
    df_scen.to_csv(csv / "scenario_recall.csv", index=False)
    df_cal.to_csv(csv / "calibration_curve.csv", index=False)
    df_cal_uniform.to_csv(csv / "calibration_curve_uniform_bins.csv", index=False)
    df_perm.to_csv(csv / "feature_importance_permutation.csv", index=False)
    if df_shap is not None:
        df_shap.to_csv(csv / "feature_importance_shap.csv", index=False)
    cm = summary["confusion_matrix"]
    pd.DataFrame([{"actual": "legitimate", "predicted_legitimate": cm["tn"],
                   "predicted_fraud": cm["fp"]},
                  {"actual": "fraud", "predicted_legitimate": cm["fn"],
                   "predicted_fraud": cm["tp"]}]).to_csv(csv / "confusion_matrix.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(csv / "classification_report.csv")
    pd.DataFrame({"dataset_row_index": idx_test, "scenario": scen_test, "is_fraud": y_test,
                  "predicted_probability": probs,
                  "flagged_at_operating_threshold": pred}).to_csv(
        csv / "test_predictions.csv", index=False)

    figures = [
        v1eval.fig_confusion(cm, OUT),
        v1eval.fig_pr(rec, prec, summary["average_precision"], summary["recall"],
                      summary["precision"], base_rate, OUT),
        v1eval.fig_roc(fpr, tpr, summary["roc_auc"], summary["false_positive_rate"],
                       summary["recall"], OUT),
        v1eval.fig_recall_vs_fpr(fpr, tpr, args.fpr_budget,
                                 summary["false_positive_rate"], summary["recall"], OUT),
        v1eval.fig_sweep(df_sweep, threshold, OUT),
        v1eval.fig_calibration(df_cal, df_cal_uniform, summary["brier"],
                               summary["brier_skill_score"], OUT),
        v1eval.fig_scenarios(df_scen, OUT),
        v1eval.fig_importance(df_perm, "importance_mean_drop_in_average_precision",
                              "importance_std",
                              "V2 permutation importance on the test split",
                              "feature_importance_permutation.png", OUT),
    ]
    if df_shap is not None:
        figures.append(v1eval.fig_importance(
            df_shap, "mean_abs_shap", None,
            "V2 SHAP summary - mean absolute attribution per feature",
            "shap_summary.png", OUT, colour=v1eval.TEAL))

    finished = datetime.now(timezone.utc)
    metrics = {
        "what_this_is": ("Frozen V2 measured once on the held-out test split. The model, "
                         "the feature set and the threshold were all chosen on train and "
                         "validation before this ran."),
        "version": "v2",
        "run": {"started_utc": started.isoformat(timespec="seconds"),
                "finished_utc": finished.isoformat(timespec="seconds"),
                "seed": args.seed, "command": " ".join(sys.argv)},
        "data": {"dataset_version": "v2", "source": "simulated (backend/ml/dataset_v2)",
                 "caveat": ("No public labelled UPI fraud corpus exists, and the message "
                            "channel is templated. These figures describe the method on "
                            "simulated data, not field performance."),
                 "rows": int(len(df)), "fraud_rows": int(y.sum()),
                 "fraud_prevalence": float(y.mean()),
                 "splits": {"train_rows": int(len(idx_train)),
                            "validation_rows": int(len(idx_val)),
                            "test_rows": int(len(idx_test)),
                            "test_fraud": int(y_test.sum()),
                            "test_fraud_prevalence": base_rate,
                            "identical_to_v1_split": True}},
        "model": {"family": frozen["model"], "calibration": frozen["calibration"],
                  "features": columns, "n_features": len(columns),
                  "feature_groups": frozen["feature_groups"]},
        "operating_point": {"threshold": threshold,
                            "source": "models/metrics_v2.json, selected on the validation split",
                            "fpr_budget": args.fpr_budget,
                            "test_labels_used_in_selection": False,
                            "validation_results": frozen["validation_results"]},
        "test_metrics": summary,
        "classification_report": report,
        "recall_at_fpr_budget": {"budget": args.fpr_budget,
                                 "realised_false_positive_rate": summary["false_positive_rate"],
                                 "recall": summary["recall"], "precision": summary["precision"],
                                 "within_budget": bool(summary["false_positive_rate"] <= args.fpr_budget)},
        "scenario_recall": df_scen.to_dict(orient="records"),
        "calibration": {"bins_requested": args.bins, "bins_populated": len(df_cal),
                        "strategy": "quantile", "brier": summary["brier"],
                        "brier_reference_constant_base_rate": reference,
                        "brier_skill_score": summary["brier_skill_score"],
                        "curve": df_cal.to_dict(orient="records"),
                        "curve_uniform_bins": df_cal_uniform.to_dict(orient="records")},
        "feature_importance": {"permutation": df_perm.to_dict(orient="records"),
                               "shap": (df_shap.to_dict(orient="records")
                                        if df_shap is not None else None),
                               "shap_note": shap_note},
        "artefacts": {"figures": figures, "csv": sorted(p.name for p in csv.glob("*.csv"))},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = {
        "generated_utc": finished.isoformat(timespec="seconds"),
        "version": "v2", "seed": args.seed,
        "dataset": fingerprint,
        "model_configuration": {"family": frozen["model"], "calibration": frozen["calibration"],
                                "hyperparameters": "max_iter=300, learning_rate=0.08, "
                                                   "max_leaf_nodes=31, class_weight=balanced",
                                "n_features": len(columns)},
        "features": columns,
        "feature_groups": frozen["feature_groups"],
        "threshold": threshold,
        "threshold_selection_method": ("highest recall inside a 1% false-positive budget, "
                                       "chosen on the validation split, rounded to 4 dp"),
        "splits": {"train": int(len(idx_train)), "validation": int(len(idx_val)),
                   "test": int(len(idx_test))},
        "python": platform.python_version(), "platform": platform.platform(),
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__,
                      "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                      "joblib": joblib.__version__, "matplotlib": matplotlib.__version__},
        "artefact_sha256": {
            path: v1eval.sha256(ROOT / path) for path in (
                "models/risk_model_v2.pkl", "models/metrics_v2.json",
                "backend/ml/dataset.py", "backend/ml/dataset_v2.py",
                "backend/ml/features.py", "backend/ml/features_v2.py",
                "backend/app/services/velocity.py", "scripts/train_model_v2.py",
                "scripts/evaluate_model_v2.py", "data/coercion_corpus.jsonl",
                "models/coercion_weights.json")},
        "reproduce": [
            "python scripts/train_model_v2.py --phase compare",
            "python scripts/train_model_v2.py --phase ablate",
            "python scripts/train_model_v2.py --phase freeze",
            f"python scripts/evaluate_model_v2.py --phase main --seed {args.seed}",
            "python scripts/evaluate_model_v2.py --phase tables",
            "python scripts/evaluate_model_v2.py --phase compare",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n  ROC-AUC {summary['roc_auc']:.4f}   PR-AUC {summary['average_precision']:.4f}   "
          f"Brier {summary['brier']:.5f} (skill {summary['brier_skill_score']:.4f})")
    print(f"  recall {summary['recall']:.4f}   precision {summary['precision']:.4f}   "
          f"FPR {summary['false_positive_rate']:.4f}   F1 {summary['f1']:.4f}")
    print(f"  confusion  tn {cm['tn']}  fp {cm['fp']}  fn {cm['fn']}  tp {cm['tp']}")
    for row in df_scen.to_dict(orient="records"):
        print(f"    {row['scenario']:<28} {row['recall']:>7.2%}  ({row['detected']}/{row['n']})")
    print(f"\n  written: evaluation_v2/metrics.json · manifest.json · {len(figures)} figures\n")
    return 0


def _fit_and_measure(estimator, X_tr, y_tr, X_va, y_va, X_te, y_te, scen_te, budget):
    """Fit on train, take the threshold from validation, measure once on test.

    The same protocol v1 used, applied to every row of the ablation and baseline
    tables so they can be read against each other and against v1.
    """
    estimator.fit(X_tr, y_tr)
    val_probs = estimator.predict_proba(X_va)[:, 1]
    _, threshold = recall_at_fpr(y_va, val_probs, budget)
    probs = estimator.predict_proba(X_te)[:, 1]
    flagged = probs >= threshold
    tp = int((flagged & (y_te == 1)).sum()); fp = int((flagged & (y_te == 0)).sum())
    fn = int((~flagged & (y_te == 1)).sum()); tn = int((~flagged & (y_te == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    row = {
        "threshold_from_validation": round(float(threshold), 6),
        "test_roc_auc": float(roc_auc_score(y_te, probs)),
        "test_pr_auc": float(average_precision_score(y_te, probs)),
        "test_recall": float(recall), "test_precision": float(precision),
        "test_f1": float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "test_fpr": float(fp / max(fp + tn, 1)),
        "test_brier": float(brier_score_loss(y_te, probs)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    for name in ("social_engineering", "mule_collection", "account_takeover", "card_testing"):
        mask = (scen_te == name) & (y_te == 1)
        row[f"recall_{name}"] = float(flagged[mask].mean()) if mask.sum() else float("nan")
    return row


def phase_tables(args) -> int:
    """Ablation and baselines, on the same held-out rows.

    These configurations were fixed by the validation work; measuring them here
    reports what each channel is worth on unseen data. None of these numbers
    chose anything - V2 was already frozen before this ran.
    """
    df, _ = load_or_generate_v2(seed=args.seed)
    y, idx_train, idx_val, idx_test = splits(df)
    y_tr, y_va, y_te = y[idx_train], y[idx_val], y[idx_test]
    scen_te = df["scenario"].iloc[idx_test].to_numpy()
    (OUT / "csv").mkdir(parents=True, exist_ok=True)

    def frame(columns, idx):
        return df[columns].iloc[idx]

    if args.what in ("ablation", "both"):
        rows = []
        print("\n  ablation on the held-out test split\n")
        print(f"  {'feature set':<34} {'n':>4} {'PR-AUC':>8} {'ROC':>7} {'recall':>7} "
              f"{'prec':>7} {'F1':>7} {'FPR':>7}")
        for label, groups in ABLATION.items():
            columns = features_for(groups)
            row = _fit_and_measure(candidates(args.seed)[args.model],
                                   frame(columns, idx_train), y_tr,
                                   frame(columns, idx_val), y_va,
                                   frame(columns, idx_test), y_te, scen_te, args.fpr_budget)
            row.update({"feature_set": label, "groups": ",".join(groups),
                        "n_features": len(columns), "model": args.model})
            rows.append(row)
            print(f"  {label:<34} {len(columns):>4} {row['test_pr_auc']:>8.4f} "
                  f"{row['test_roc_auc']:>7.4f} {row['test_recall']:>7.4f} "
                  f"{row['test_precision']:>7.4f} {row['test_f1']:>7.4f} {row['test_fpr']:>7.4f}")
        row = _fit_and_measure(candidates(args.seed)[args.model],
                               frame(FEATURES_V1, idx_train), y_tr,
                               frame(FEATURES_V1, idx_val), y_va,
                               frame(FEATURES_V1, idx_test), y_te, scen_te, args.fpr_budget)
        row.update({"feature_set": "v1 twelve (reference)", "groups": "v1",
                    "n_features": len(FEATURES_V1), "model": args.model})
        rows.append(row)
        print(f"  {'v1 twelve (reference)':<34} {len(FEATURES_V1):>4} {row['test_pr_auc']:>8.4f} "
              f"{row['test_roc_auc']:>7.4f} {row['test_recall']:>7.4f} "
              f"{row['test_precision']:>7.4f} {row['test_f1']:>7.4f} {row['test_fpr']:>7.4f}")
        pd.DataFrame(rows).to_csv(OUT / "ablation_results.csv", index=False)
        print(f"\n  written: evaluation_v2/ablation_results.csv")

    if args.what in ("baselines", "both"):
        rows = []
        print("\n  baseline comparison on the held-out test split "
              f"({len(FEATURES_V2)} v2 features unless stated)\n")
        print(f"  {'model':<34} {'PR-AUC':>8} {'ROC':>7} {'recall':>7} {'prec':>7} "
              f"{'F1':>7} {'FPR':>7}")
        for name, estimator in candidates(args.seed).items():
            row = _fit_and_measure(estimator, frame(FEATURES_V2, idx_train), y_tr,
                                   frame(FEATURES_V2, idx_val), y_va,
                                   frame(FEATURES_V2, idx_test), y_te, scen_te, args.fpr_budget)
            row.update({"model": name, "features": "full v2", "n_features": len(FEATURES_V2),
                        "calibrated": False})
            rows.append(row)
            print(f"  {name:<34} {row['test_pr_auc']:>8.4f} {row['test_roc_auc']:>7.4f} "
                  f"{row['test_recall']:>7.4f} {row['test_precision']:>7.4f} "
                  f"{row['test_f1']:>7.4f} {row['test_fpr']:>7.4f}")
        pd.DataFrame(rows).to_csv(OUT / "baseline_comparison.csv", index=False)
        print(f"\n  written: evaluation_v2/baseline_comparison.csv\n")
    return 0


def phase_compare(args) -> int:
    """The table the research question actually asks for."""
    if not V1_METRICS.exists():
        raise SystemExit("V1 evaluation missing - evaluation/metrics.json is the baseline")
    v1 = json.loads(V1_METRICS.read_text(encoding="utf-8"))
    v2 = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))

    v1_scen = {r["scenario"]: r for r in v1["scenario_recall"]}
    v2_scen = {r["scenario"]: r for r in v2["scenario_recall"]}
    higher_is_better = {
        "ROC-AUC": True, "PR-AUC (average precision)": True, "Recall at 1% FPR budget": True,
        "Precision at operating point": True, "F1 (fraud class)": True,
        "False-positive rate": False, "Brier score": False, "Brier skill score": True,
        "card_testing recall": True, "account_takeover recall": True,
        "mule_collection recall": True, "social_engineering recall": True,
        "False alarms (legitimate flagged)": False,
    }
    pairs = {
        "ROC-AUC": (v1["test_metrics"]["roc_auc"], v2["test_metrics"]["roc_auc"]),
        "PR-AUC (average precision)": (v1["test_metrics"]["average_precision"],
                                       v2["test_metrics"]["average_precision"]),
        "Recall at 1% FPR budget": (v1["test_metrics"]["recall"], v2["test_metrics"]["recall"]),
        "Precision at operating point": (v1["test_metrics"]["precision"],
                                         v2["test_metrics"]["precision"]),
        "F1 (fraud class)": (v1["test_metrics"]["f1"], v2["test_metrics"]["f1"]),
        "False-positive rate": (v1["test_metrics"]["false_positive_rate"],
                                v2["test_metrics"]["false_positive_rate"]),
        "Brier score": (v1["test_metrics"]["brier"], v2["test_metrics"]["brier"]),
        "Brier skill score": (v1["test_metrics"]["brier_skill_score"],
                              v2["test_metrics"]["brier_skill_score"]),
        "card_testing recall": (v1_scen["card_testing"]["recall"], v2_scen["card_testing"]["recall"]),
        "account_takeover recall": (v1_scen["account_takeover"]["recall"],
                                    v2_scen["account_takeover"]["recall"]),
        "mule_collection recall": (v1_scen["mule_collection"]["recall"],
                                   v2_scen["mule_collection"]["recall"]),
        "social_engineering recall": (v1_scen["social_engineering"]["recall"],
                                      v2_scen["social_engineering"]["recall"]),
        "False alarms (legitimate flagged)": (
            v1_scen["legitimate (false alarms)"]["detected"],
            v2_scen["legitimate (false alarms)"]["detected"]),
    }
    rows = []
    print("\n  V1 vs V2 on the same 30,000 held-out rows\n")
    print(f"  {'metric':<36} {'V1':>10} {'V2':>10} {'change':>12}  ")
    for metric, (a, b) in pairs.items():
        delta = b - a
        better = (delta > 0) == higher_is_better[metric] if delta else None
        verdict = "-" if not delta else ("better" if better else "WORSE")
        rows.append({"metric": metric, "v1": a, "v2": b, "change": delta,
                     "direction": verdict,
                     "higher_is_better": higher_is_better[metric]})
        fmt = (f"{a:>10.4f} {b:>10.4f} {delta:>+12.4f}" if abs(a) < 1000
               else f"{a:>10.0f} {b:>10.0f} {delta:>+12.0f}")
        print(f"  {metric:<36} {fmt}  {verdict}")
    pd.DataFrame(rows).to_csv(OUT / "comparison_v1_vs_v2.csv", index=False)

    v1cm, v2cm = v1["test_metrics"]["confusion_matrix"], v2["test_metrics"]["confusion_matrix"]
    print(f"\n  confusion  V1  tn {v1cm['tn']}  fp {v1cm['fp']}  fn {v1cm['fn']}  tp {v1cm['tp']}")
    print(f"             V2  tn {v2cm['tn']}  fp {v2cm['fp']}  fn {v2cm['fn']}  tp {v2cm['tp']}")
    print(f"\n  written: evaluation_v2/comparison_v1_vs_v2.csv\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("main", "tables", "compare"), required=True)
    ap.add_argument("--what", choices=("ablation", "baselines", "both"), default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fpr-budget", type=float, default=0.01)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--shap-rows", type=int, default=200)
    ap.add_argument("--model", default="HistGradientBoosting (v1 config)")
    args = ap.parse_args()
    return {"main": phase_main, "tables": phase_tables, "compare": phase_compare}[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())


def redraw_importance(out: Path = OUT) -> list[str]:
    """Re-draw the two importance charts at a height that fits 47 features.

    v1's fig_importance is sized for twelve; borrowing it unchanged produced a
    readable chart there and an unreadable one here. This draws from the CSVs
    the evaluation already wrote, so no model is scored again.
    """
    import matplotlib.pyplot as plt

    written = []
    for name, value_col, err_col, title, colour in (
        ("feature_importance_permutation", "importance_mean_drop_in_average_precision",
         "importance_std", "V2 permutation importance on the test split\n"
         "(drop in average precision when a feature is shuffled)", v1eval.BLUE),
        ("feature_importance_shap", "mean_abs_shap", None,
         "V2 SHAP summary - mean absolute attribution per feature", v1eval.TEAL),
    ):
        path = out / "csv" / f"{name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path).sort_values(value_col)
        height = max(4.4, 0.23 * len(frame))
        fig, ax = plt.subplots(figsize=(7.2, height))
        ax.barh(frame["feature"], frame[value_col], height=0.62, color=colour,
                xerr=frame[err_col] if err_col else None,
                error_kw={"ecolor": v1eval.MUTED, "elinewidth": 0.9})
        span = float(frame[value_col].max()) or 1.0
        for y, v in enumerate(frame[value_col]):
            ax.text(v + span * 0.012, y, f"{v:.4f}", va="center", fontsize=7, color=v1eval.INK)
        v1eval.frame_for(ax, title, value_col.replace("_", " "), "")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_xlim(0, span * 1.18)
        written.append(v1eval.save(fig, out / "figures" / f"{'shap_summary' if 'shap' in name else name}.png"))
    return written
