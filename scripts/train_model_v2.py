"""V2 development: model comparison, ablation, and the frozen candidate.

Everything in this script happens on TRAIN and VALIDATION. The test split is
rebuilt only so that it can be excluded, and its labels are never read. The
frozen model, the feature set and the operating threshold are all chosen here;
scripts/evaluate_model_v2.py then measures them once on the held-out rows.

Run in phases so each stays inside a sane wall clock:

    python scripts/train_model_v2.py --phase compare    # model families
    python scripts/train_model_v2.py --phase ablate     # feature groups
    python scripts/train_model_v2.py --phase freeze     # fit, calibrate, threshold

A note on calibration during comparison
---------------------------------------
Models are compared uncalibrated. Isotonic calibration is a monotone map, so
it cannot change ROC-AUC, average precision, or recall at a fixed
false-positive rate - the three numbers the comparison turns on. It does change
Brier and the meaning of the probability, so the FROZEN model is calibrated
exactly as v1's was, and its calibration is reported on validation here and on
test in the final evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib                                                        # noqa: E402
from sklearn.calibration import CalibratedClassifierCV               # noqa: E402
from sklearn.ensemble import (                                       # noqa: E402
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.impute import SimpleImputer                             # noqa: E402
from sklearn.linear_model import LogisticRegression                  # noqa: E402
from sklearn.metrics import (                                        # noqa: E402
    average_precision_score, brier_score_loss, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split                 # noqa: E402
from sklearn.pipeline import Pipeline                                # noqa: E402
from sklearn.preprocessing import StandardScaler                     # noqa: E402

from backend.ml.dataset import LABEL                                 # noqa: E402
from backend.ml.dataset_v2 import (                                  # noqa: E402
    DATASET_VERSION, FEATURES_V1, FEATURES_V2, FEATURE_GROUPS, features_for,
    load_or_generate_v2,
)
from backend.train_model import _strip_fit_only_rng                  # noqa: E402

SEED = 42
FPR_BUDGET = 0.01
MODELS = ROOT / "models"
OUT = ROOT / "evaluation_v2"
WORK = OUT / "development"

# The ablation ladder. Each rung adds one channel, so the table reads as
# "what did this information buy".
ABLATION = {
    "A. payer only":                    ["payer", "amount"],
    "B. payer + payee":                 ["payer", "amount", "payee"],
    "C. payer + payee + velocity":      ["payer", "amount", "payee", "velocity"],
    "D. C + graph":                     ["payer", "amount", "payee", "velocity", "graph"],
    "E. C + message/context":           ["payer", "amount", "payee", "velocity",
                                         "message", "link", "intent"],
    "F. full v2":                       list(FEATURE_GROUPS),
    "full v2 without graph":            [g for g in FEATURE_GROUPS if g != "graph"],
    "full v2 without message/context":  [g for g in FEATURE_GROUPS
                                         if g not in ("message", "link", "intent")],
}


def splits(df: pd.DataFrame):
    """The v1 split calls, unchanged. The test indices are returned so they can
    be held out, and for no other purpose."""
    y = df[LABEL].to_numpy()
    idx_fit, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.25, random_state=SEED, stratify=y)
    idx_train, idx_val = train_test_split(
        idx_fit, test_size=0.20, random_state=SEED, stratify=y[idx_fit])
    return y, idx_train, idx_val, idx_test


def recall_at_fpr(y_true, scores, budget=FPR_BUDGET):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    ok = fpr <= budget
    if not ok.any():
        return 0.0, 1.0
    i = int(np.argmax(tpr * ok))
    return float(tpr[i]), float(thresholds[i])


def candidates(seed: int = SEED) -> dict:
    impute = lambda: SimpleImputer(strategy="median")                # noqa: E731
    return {
        "HistGradientBoosting (v1 config)": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            class_weight="balanced", random_state=seed),
        "RandomForest": Pipeline([("impute", impute()), ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed))]),
        "ExtraTrees": Pipeline([("impute", impute()), ("clf", ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=seed))]),
        "LogisticRegression": Pipeline([
            ("impute", impute()), ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    }


def score_on_validation(model, X_tr, y_tr, X_va, y_va, scen_va) -> dict:
    started = time.time()
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_va)[:, 1]
    recall, threshold = recall_at_fpr(y_va, probs)
    flagged = probs >= threshold
    tp = int((flagged & (y_va == 1)).sum())
    fp = int((flagged & (y_va == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    row = {
        "val_pr_auc": float(average_precision_score(y_va, probs)),
        "val_roc_auc": float(roc_auc_score(y_va, probs)),
        "val_recall_at_1pct_fpr": recall,
        "val_precision_at_that_point": float(precision),
        "val_f1": float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "val_realised_fpr": float(fp / max(int((y_va == 0).sum()), 1)),
        "val_brier": float(brier_score_loss(y_va, probs)),
        "threshold_on_validation": threshold,
        "fit_seconds": round(time.time() - started, 1),
    }
    for name in ("social_engineering", "mule_collection", "account_takeover", "card_testing"):
        mask = (scen_va == name) & (y_va == 1)
        row[f"val_recall_{name}"] = float(flagged[mask].mean()) if mask.sum() else float("nan")
    return row


def load_everything(seed: int):
    df, fingerprint = load_or_generate_v2(seed=seed)
    y, idx_train, idx_val, idx_test = splits(df)
    assert not (set(idx_train) & set(idx_test)) and not (set(idx_val) & set(idx_test)), \
        "the development splits touch the test split"
    return df, y, idx_train, idx_val, idx_test, fingerprint


def phase_compare(args) -> int:
    """Which model family, on the full v2 feature set. Validation only."""
    df, y, idx_train, idx_val, _, fingerprint = load_everything(args.seed)
    X_tr, y_tr = df[FEATURES_V2].iloc[idx_train], y[idx_train]
    X_va, y_va = df[FEATURES_V2].iloc[idx_val], y[idx_val]
    scen_va = df["scenario"].iloc[idx_val].to_numpy()

    rows = []
    print(f"\n  model comparison on {len(FEATURES_V2)} v2 features "
          f"(train {len(idx_train):,} -> validation {len(idx_val):,})\n")
    print(f"  {'model':<34} {'PR-AUC':>8} {'ROC':>7} {'rec@1%':>8} {'prec':>7} "
          f"{'SE':>7} {'mule':>7}")
    for name, model in candidates(args.seed).items():
        row = score_on_validation(model, X_tr, y_tr, X_va, y_va, scen_va)
        row["model"] = name
        row["features"] = "full v2"
        rows.append(row)
        print(f"  {name:<34} {row['val_pr_auc']:>8.4f} {row['val_roc_auc']:>7.4f} "
              f"{row['val_recall_at_1pct_fpr']:>8.4f} {row['val_precision_at_that_point']:>7.4f} "
              f"{row['val_recall_social_engineering']:>7.4f} {row['val_recall_mule_collection']:>7.4f}")

    # The v1 feature set under the same protocol, as the reference point.
    reference = score_on_validation(candidates(args.seed)["HistGradientBoosting (v1 config)"],
                                    df[FEATURES_V1].iloc[idx_train], y_tr,
                                    df[FEATURES_V1].iloc[idx_val], y_va, scen_va)
    reference["model"] = "HistGradientBoosting (v1 config)"
    reference["features"] = "v1 twelve"
    rows.append(reference)
    print(f"  {'-- v1 features, same protocol':<34} {reference['val_pr_auc']:>8.4f} "
          f"{reference['val_roc_auc']:>7.4f} {reference['val_recall_at_1pct_fpr']:>8.4f} "
          f"{reference['val_precision_at_that_point']:>7.4f} "
          f"{reference['val_recall_social_engineering']:>7.4f} "
          f"{reference['val_recall_mule_collection']:>7.4f}")

    WORK.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(WORK / "validation_model_comparison.csv", index=False)
    (WORK / "dataset_fingerprint.json").write_text(json.dumps(fingerprint, indent=2),
                                                   encoding="utf-8")
    best = frame[frame["features"] == "full v2"].sort_values(
        ["val_pr_auc", "val_recall_at_1pct_fpr"], ascending=False).iloc[0]
    print(f"\n  validation prefers: {best['model']}  "
          f"(PR-AUC {best['val_pr_auc']:.4f}, recall@1%FPR {best['val_recall_at_1pct_fpr']:.4f})")
    print(f"  written: {(WORK / 'validation_model_comparison.csv').relative_to(ROOT)}\n")
    return 0


def phase_ablate(args) -> int:
    """What each channel is worth. One model family, validation only."""
    df, y, idx_train, idx_val, _, _ = load_everything(args.seed)
    y_tr, y_va = y[idx_train], y[idx_val]
    scen_va = df["scenario"].iloc[idx_val].to_numpy()

    chosen = args.model
    print(f"\n  ablation with {chosen}, validation only\n")
    print(f"  {'feature set':<34} {'n':>4} {'PR-AUC':>8} {'ROC':>7} {'rec@1%':>8} "
          f"{'prec':>7} {'SE':>7} {'mule':>7}")
    rows = []
    for label, groups in ABLATION.items():
        columns = features_for(groups)
        row = score_on_validation(candidates(args.seed)[chosen],
                                  df[columns].iloc[idx_train], y_tr,
                                  df[columns].iloc[idx_val], y_va, scen_va)
        row.update({"feature_set": label, "groups": ",".join(groups), "n_features": len(columns)})
        rows.append(row)
        print(f"  {label:<34} {len(columns):>4} {row['val_pr_auc']:>8.4f} "
              f"{row['val_roc_auc']:>7.4f} {row['val_recall_at_1pct_fpr']:>8.4f} "
              f"{row['val_precision_at_that_point']:>7.4f} "
              f"{row['val_recall_social_engineering']:>7.4f} "
              f"{row['val_recall_mule_collection']:>7.4f}")

    # The v1 twelve, for scale.
    row = score_on_validation(candidates(args.seed)[chosen],
                              df[FEATURES_V1].iloc[idx_train], y_tr,
                              df[FEATURES_V1].iloc[idx_val], y_va, scen_va)
    row.update({"feature_set": "v1 twelve (reference)", "groups": "v1", "n_features": len(FEATURES_V1)})
    rows.append(row)
    print(f"  {'v1 twelve (reference)':<34} {len(FEATURES_V1):>4} {row['val_pr_auc']:>8.4f} "
          f"{row['val_roc_auc']:>7.4f} {row['val_recall_at_1pct_fpr']:>8.4f} "
          f"{row['val_precision_at_that_point']:>7.4f} "
          f"{row['val_recall_social_engineering']:>7.4f} "
          f"{row['val_recall_mule_collection']:>7.4f}")

    WORK.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(WORK / "validation_ablation.csv", index=False)
    print(f"\n  written: {(WORK / 'validation_ablation.csv').relative_to(ROOT)}\n")
    return 0


def phase_freeze(args) -> int:
    """Fit the chosen configuration, calibrate it, take the threshold from
    validation, and write the artefact. Nothing here reads a test row."""
    df, y, idx_train, idx_val, idx_test, fingerprint = load_everything(args.seed)
    columns = features_for(ABLATION[args.feature_set]) if args.feature_set in ABLATION \
        else list(FEATURES_V2)
    X_tr, y_tr = df[columns].iloc[idx_train], y[idx_train]
    X_va, y_va = df[columns].iloc[idx_val], y[idx_val]
    scen_va = df["scenario"].iloc[idx_val].to_numpy()

    print(f"\n  freezing: {args.model} on '{args.feature_set}' ({len(columns)} features)")
    estimator = candidates(args.seed)[args.model]
    model = CalibratedClassifierCV(estimator, method="isotonic", cv=5)
    started = time.time()
    model.fit(X_tr, y_tr)
    print(f"  fitted and calibrated in {time.time() - started:.0f}s "
          f"on {len(idx_train):,} training rows")

    probs = model.predict_proba(X_va)[:, 1]
    recall, threshold = recall_at_fpr(y_va, probs, args.fpr_budget)
    flagged = probs >= threshold
    tp = int((flagged & (y_va == 1)).sum()); fp = int((flagged & (y_va == 0)).sum())
    fn = int((~flagged & (y_va == 1)).sum()); tn = int((~flagged & (y_va == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    base_rate = float(y_va.mean())
    validation = {
        "threshold": round(float(threshold), 4),
        "threshold_full_precision": float(threshold),
        "threshold_selected_on": "validation split",
        "fpr_budget": args.fpr_budget,
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "realised_fpr": float(fp / max(fp + tn, 1)),
        "pr_auc": float(average_precision_score(y_va, probs)),
        "roc_auc": float(roc_auc_score(y_va, probs)),
        "brier": float(brier_score_loss(y_va, probs)),
        "brier_reference_constant_base_rate": float(np.mean((y_va - base_rate) ** 2)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "recall_by_scenario": {
            name: float(flagged[(scen_va == name) & (y_va == 1)].mean())
            for name in ("social_engineering", "mule_collection",
                         "account_takeover", "card_testing")
        },
    }
    validation["brier_skill_score"] = float(
        1 - validation["brier"] / validation["brier_reference_constant_base_rate"])

    print(f"  validation threshold {validation['threshold']:.4f}  ->  "
          f"recall {validation['recall']:.4f}  precision {validation['precision']:.4f}  "
          f"FPR {validation['realised_fpr']:.4f}  F1 {validation['f1']:.4f}")
    print("  scenario recall on validation: " + "  ".join(
        f"{k[:4]} {v:.3f}" for k, v in validation["recall_by_scenario"].items()))

    MODELS.mkdir(exist_ok=True)
    stripped = _strip_fit_only_rng(model)     # portability across numpy majors, as v1
    joblib.dump(model, MODELS / "risk_model_v2.pkl")
    frozen = {
        "version": "v2",
        "frozen_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "model": args.model,
        "calibration": "isotonic, 5-fold",
        "feature_set": args.feature_set,
        "features": columns,
        "n_features": len(columns),
        "feature_groups": {g: [c for c in FEATURE_GROUPS[g] if c in columns]
                           for g in FEATURE_GROUPS},
        "dataset_version": DATASET_VERSION,
        "dataset_fingerprint": fingerprint,
        "seed": args.seed,
        "splits": {"train": int(len(idx_train)), "validation": int(len(idx_val)),
                   "test_held_out": int(len(idx_test))},
        "validation_results": validation,
        "test_rows_used_in_any_decision": False,
        "rng_states_stripped": stripped,
        "artefact": "models/risk_model_v2.pkl",
    }
    (MODELS / "metrics_v2.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "frozen_configuration.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"\n  written: models/risk_model_v2.pkl · models/metrics_v2.json")
    print("  v2 is now frozen. The test split has still never been read.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("compare", "ablate", "freeze"), required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--model", default="HistGradientBoosting (v1 config)")
    ap.add_argument("--feature-set", default="F. full v2")
    ap.add_argument("--fpr-budget", type=float, default=FPR_BUDGET)
    args = ap.parse_args()
    return {"compare": phase_compare, "ablate": phase_ablate, "freeze": phase_freeze}[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
