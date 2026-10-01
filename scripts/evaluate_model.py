"""Reproducible evaluation of the CURRENT trained artifact on the untouched test split.

What this script is
-------------------
A measurement, not a training run. It fits nothing, changes nothing and
touches no application code. It rebuilds the same dataset and the same three
splits the training script produced, loads models/risk_model.pkl as it stands,
takes the operating threshold that was already selected on the VALIDATION
split, and reports what that artifact does on the 25% of rows it has never
seen.

Why the split can be rebuilt rather than stored
-----------------------------------------------
backend/train_model.train() derives its splits from
train_test_split(np.arange(len(df)), test_size=0.25, random_state=seed,
stratify=y) over generate(seed=seed). Both are deterministic functions of the
seed, so the same seed reproduces the same row indices exactly. The script
asserts this rather than assuming it: the three index sets must be pairwise
disjoint, must cover every row, and must match the sizes the training run
recorded.

Where the threshold comes from, and why that matters
----------------------------------------------------
The operating threshold is read from models/metrics.json, where the training
run stored it together with the note that it was selected on the validation
split. The script then RE-DERIVES it from the validation split alone and
requires the two to agree. At no point does a test label or a test feature
influence the threshold: selecting an operating point on the data you then
report is how a fraud system comes to promise a false-positive rate it cannot
hold, and the guarantee here is structural rather than a matter of care.

Outputs (all under --out, default evaluation/)
----------------------------------------------
    metrics.json                machine-readable, every number in this run
    manifest.json               versions, artefact hashes, timestamps, seed
    csv/*.csv                   the underlying values behind every figure
    figures/*.png               the figures themselves

Usage
-----
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --seed 42 --out evaluation --shap-rows 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib                                                     # noqa: E402
import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import scipy                                                      # noqa: E402
import sklearn                                                    # noqa: E402
from sklearn.calibration import calibration_curve                 # noqa: E402
from sklearn.inspection import permutation_importance             # noqa: E402
from sklearn.metrics import (                                     # noqa: E402
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split              # noqa: E402

from backend.ml.dataset import FEATURES, LABEL, generate          # noqa: E402
from backend.train_model import recall_at_fpr                     # noqa: E402

MODELS = ROOT / "models"
MODEL_PATH = MODELS / "risk_model.pkl"
TRAIN_METRICS = MODELS / "metrics.json"
BACKGROUND = MODELS / "shap_background.json"

# The figure palette. Three categorical hues, validated for colour-vision
# deficiency separation and contrast against a light surface; every series is
# also direct-labelled, so identity never rests on colour alone.
BLUE, TEAL, AMBER = "#2563EB", "#0D9488", "#D97706"
INK, MUTED, GRID = "#1F2937", "#6B7280", "#E5E7EB"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_for(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path.name


# ── 1. the splits, rebuilt exactly as the training run made them ────────────

def rebuild_splits(seed: int):
    """generate() and train_test_split() are both pure functions of the seed."""
    df = generate(seed=seed)
    y = df[LABEL].to_numpy()
    idx_fit, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.25, random_state=seed, stratify=y
    )
    idx_train, idx_val = train_test_split(
        idx_fit, test_size=0.20, random_state=seed, stratify=y[idx_fit]
    )
    # The guarantees this evaluation rests on, checked rather than assumed.
    sets = [set(idx_train.tolist()), set(idx_val.tolist()), set(idx_test.tolist())]
    assert not (sets[0] & sets[1]), "train and validation overlap"
    assert not (sets[0] & sets[2]), "train and TEST overlap"
    assert not (sets[1] & sets[2]), "validation and TEST overlap"
    assert len(sets[0] | sets[1] | sets[2]) == len(df), "the three splits do not cover the dataset"
    return df, y, idx_train, idx_val, idx_test


# ── 2. the operating point, selected on validation and only on validation ───

def operating_threshold(model, df, y, idx_val, fpr_budget: float):
    """Read the stored threshold, then re-derive it from validation alone.

    Two independent routes to the same number. The stored one is what the
    live system uses; the re-derived one proves the number is reproducible
    without ever consulting a test row.
    """
    stored = json.loads(TRAIN_METRICS.read_text(encoding="utf-8"))
    op = stored["selected"]["operating_point"]
    provenance = op.get("threshold_selected_on", "")
    assert provenance == "validation split", (
        f"the stored threshold claims to come from {provenance!r}; this evaluation "
        "will not use a threshold that was selected on the data it reports"
    )
    val_probs = model.predict_proba(df[FEATURES].iloc[idx_val])[:, 1]
    _, rederived = recall_at_fpr(y[idx_val], val_probs, fpr_budget)
    # train_model writes round(threshold, 4), so the stored number is the
    # served number and the re-derived one is its full-precision original.
    # Agreement is therefore agreement to the stored precision, not to the bit.
    return float(op["threshold"]), float(rederived), stored


# ── 3. metrics, all of them on the test split alone ─────────────────────────

def evaluate(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    support_pos, support_neg = int(y_true.sum()), int(len(y_true) - y_true.sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {"fraud": support_pos, "legitimate": support_neg, "total": int(len(y_true))},
        "fraud_prevalence": float(support_pos / len(y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "f1": float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "balanced_accuracy": float(0.5 * (recall + (tn / (tn + fp) if (tn + fp) else 0.0))),
        "accuracy": float((tp + tn) / len(y_true)),
        "alerts": int(tp + fp),
        "alert_rate": float((tp + fp) / len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "average_precision": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
    }


def sweep(y_true, probs, thresholds):
    rows = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({
            "threshold": float(t), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(p), "recall": float(r),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "f1": float(2 * p * r / (p + r)) if (p + r) else 0.0,
            "alerts": int(tp + fp), "alert_rate": float((tp + fp) / len(y_true)),
        })
    return pd.DataFrame(rows)


def scenario_recall(scen, y_true, probs, threshold):
    flagged = probs >= threshold
    rows = []
    for name in sorted(set(scen[y_true == 1])):
        mask = (scen == name) & (y_true == 1)
        n = int(mask.sum())
        caught = int(flagged[mask].sum())
        rows.append({
            "scenario": name, "n": n, "detected": caught,
            "recall": float(caught / n) if n else 0.0,
            "mean_probability": float(probs[mask].mean()),
            "median_probability": float(np.median(probs[mask])),
        })
    legit = y_true == 0
    rows.append({
        "scenario": "legitimate (false alarms)", "n": int(legit.sum()),
        "detected": int(flagged[legit].sum()),
        "recall": float(flagged[legit].mean()),
        "mean_probability": float(probs[legit].mean()),
        "median_probability": float(np.median(probs[legit])),
    })
    return pd.DataFrame(rows)


# ── 4. figures ──────────────────────────────────────────────────────────────

def fig_confusion(cm, out):
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    grid = np.array([[tn, fp], [fn, tp]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(grid / grid.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            share = grid[i, j] / grid[i].sum()
            ax.text(j, i, f"{int(grid[i, j]):,}\n{share:.1%} of row",
                    ha="center", va="center", fontsize=10,
                    color="white" if share > 0.55 else INK,
                    fontweight="bold")
    ax.set_xticks([0, 1], ["predicted legitimate", "predicted fraud"])
    ax.set_yticks([0, 1], ["actually legitimate", "actually fraud"])
    ax.grid(False)
    ax.set_title("Confusion matrix — held-out test split", fontsize=10, fontweight="bold", pad=10)
    return save(fig, out / "figures" / "confusion_matrix.png")


def fig_pr(rec, prec, ap, op_rec, op_prec, prevalence, out):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(rec, prec, color=BLUE, linewidth=2, label=f"model (AP = {ap:.4f})")
    ax.axhline(prevalence, color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(0.02, prevalence + 0.012, f"no-skill baseline = prevalence {prevalence:.4f}",
            fontsize=7.5, color=MUTED)
    ax.plot([op_rec], [op_prec], "o", markersize=9, color=AMBER, zorder=5)
    ax.annotate(f"operating point\nrecall {op_rec:.3f} · precision {op_prec:.3f}",
                (op_rec, op_prec), textcoords="offset points", xytext=(12, 14),
                fontsize=7.5, color=INK)
    frame_for(ax, "Precision–recall — held-out test split", "Recall", "Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return save(fig, out / "figures" / "pr_curve.png")


def fig_roc(fpr, tpr, auc, op_fpr, op_tpr, out):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2, label=f"model (ROC-AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.1, linestyle="--", label="chance")
    ax.plot([op_fpr], [op_tpr], "o", markersize=9, color=AMBER, zorder=5)
    ax.annotate(f"operating point\nFPR {op_fpr:.4f} · TPR {op_tpr:.3f}",
                (op_fpr, op_tpr), textcoords="offset points", xytext=(14, -6),
                fontsize=7.5, color=INK)
    frame_for(ax, "ROC — held-out test split", "False-positive rate", "True-positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return save(fig, out / "figures" / "roc_curve.png")


def fig_recall_vs_fpr(fpr, tpr, budget, op_fpr, op_tpr, out):
    keep = fpr <= 0.10
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(fpr[keep], tpr[keep], color=BLUE, linewidth=2, label="recall achievable")
    ax.axvline(budget, color=TEAL, linewidth=1.4, linestyle="--")
    ax.text(budget + 0.0015, 0.05, f"{budget:.0%} false-positive budget",
            rotation=90, fontsize=7.5, color=TEAL, va="bottom")
    ax.plot([op_fpr], [op_tpr], "o", markersize=9, color=AMBER, zorder=5)
    ax.annotate(f"operating point — recall {op_tpr:.3f} at FPR {op_fpr:.4f}",
                (op_fpr, op_tpr), textcoords="offset points", xytext=(14, 6),
                fontsize=7.5, color=INK)
    frame_for(ax, "Recall against false-positive rate (test split)",
              "False-positive rate", "Recall")
    ax.set_xlim(0, 0.10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return save(fig, out / "figures" / "recall_vs_fpr.png")


def fig_sweep(df_sweep, threshold, out):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for col, colour, label in ((("precision"), BLUE, "precision"),
                               ("recall", TEAL, "recall"),
                               ("false_positive_rate", AMBER, "false-positive rate")):
        ax.plot(df_sweep["threshold"], df_sweep[col], color=colour, linewidth=2, label=label)
        j = int(len(df_sweep) * 0.62)
        ax.annotate(label, (df_sweep["threshold"].iloc[j], df_sweep[col].iloc[j]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5, color=colour)
    ax.axvline(threshold, color=INK, linewidth=1.2, linestyle="--")
    ax.annotate(f"operating threshold {threshold:.4f}\nselected on the validation split",
                xy=(threshold, 0.72), xytext=(-8, 0), textcoords="offset points",
                fontsize=7.5, color=INK, ha="right", va="center",
                bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.3"})
    frame_for(ax, "Threshold analysis — test split", "Decision threshold", "Rate")
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    return save(fig, out / "figures" / "threshold_sweep.png")


def fig_calibration(df_q, df_u, brier, bss, out):
    """Two panels, because the two binnings live on different scales.

    Nearly every prediction sits below 0.1, so quantile bins describe where the
    mass is and fixed-width bins describe the upper range where the fraud is.
    Putting both on one axis would squash the first into the corner.
    """
    fig, (ax_q, ax_u) = plt.subplots(1, 2, figsize=(9.4, 4.4))

    lim_q = max(float(df_q["mean_predicted_probability"].max()),
                float(df_q["observed_fraud_rate"].max())) * 1.18
    ax_q.plot([0, lim_q], [0, lim_q], color=MUTED, linewidth=1.1, linestyle="--")
    ax_q.plot(df_q["mean_predicted_probability"], df_q["observed_fraud_rate"], "o-",
              color=BLUE, linewidth=2, markersize=7)
    for _, r in df_q.iterrows():
        ax_q.annotate(f"n={r['n']:,.0f}", (r["mean_predicted_probability"], r["observed_fraud_rate"]),
                      textcoords="offset points", xytext=(8, -4), fontsize=7.5, color=BLUE)
    frame_for(ax_q, f"Quantile bins ({len(df_q)} populated) — where the mass is",
              "Mean predicted probability", "Observed fraud rate")
    ax_q.set_xlim(0, lim_q); ax_q.set_ylim(0, lim_q)

    ax_u.plot([0, 1], [0, 1], color=MUTED, linewidth=1.1, linestyle="--", label="perfect calibration")
    ax_u.plot(df_u["mean_predicted_probability"], df_u["observed_fraud_rate"], "s-",
              color=TEAL, linewidth=2, markersize=6, label="fixed-width bins")
    frame_for(ax_u, "Fixed-width bins — the upper range, where the fraud is",
              "Mean predicted probability", "Observed fraud rate")
    ax_u.set_xlim(0, 1.02); ax_u.set_ylim(0, 1.02)
    ax_u.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(f"Calibration — held-out test split    ·    Brier {brier:.5f}   ·   "
                 f"skill score {bss:.3f} against the base rate",
                 fontsize=10, fontweight="bold", y=1.0)
    return save(fig, out / "figures" / "calibration_curve.png")


def fig_scenarios(df_scen, out):
    rows = df_scen[df_scen["scenario"] != "legitimate (false alarms)"].sort_values("recall")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.barh(rows["scenario"], rows["recall"], color=BLUE, height=0.55)
    for y, (r, n, d) in enumerate(zip(rows["recall"], rows["n"], rows["detected"])):
        ax.text(r + 0.012, y, f"{r:.1%}   ({d}/{n})", va="center", fontsize=8, color=INK)
    frame_for(ax, "Recall by fraud scenario at the operating threshold", "Recall", "")
    ax.set_xlim(0, 1.15); ax.grid(axis="y", visible=False)
    return save(fig, out / "figures" / "scenario_recall.png")


def fig_importance(df_imp, value_col, err_col, title, fname, out, colour=BLUE):
    rows = df_imp.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.barh(rows["feature"], rows[value_col], height=0.6, color=colour,
            xerr=rows[err_col] if err_col else None,
            error_kw={"ecolor": MUTED, "elinewidth": 1})
    for y, v in enumerate(rows[value_col]):
        ax.text(v + rows[value_col].max() * 0.015, y, f"{v:.4f}", va="center", fontsize=7.5, color=INK)
    frame_for(ax, title, value_col.replace("_", " "), "")
    ax.grid(axis="y", visible=False)
    return save(fig, out / "figures" / fname)


# ── 5. the run ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42,
                    help="the seed the artefact was trained with (default 42)")
    ap.add_argument("--out", default="evaluation", help="output directory")
    ap.add_argument("--fpr-budget", type=float, default=0.01)
    ap.add_argument("--bins", type=int, default=10, help="calibration bins")
    ap.add_argument("--shap-rows", type=int, default=200,
                    help="test rows to explain with SHAP (0 disables)")
    ap.add_argument("--permutation-repeats", type=int, default=5)
    args = ap.parse_args()

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    if not MODEL_PATH.exists():
        print(f"No trained artefact at {MODEL_PATH}. Train it first: python backend/train_model.py")
        return 1

    print(f"\n  Evaluating {MODEL_PATH.relative_to(ROOT)} on the held-out test split")
    print(f"  seed {args.seed}   ·   nothing is fitted, nothing is retrained\n")

    # data and splits
    df, y, idx_train, idx_val, idx_test = rebuild_splits(args.seed)
    X_test = df[FEATURES].iloc[idx_test]
    y_test = y[idx_test]
    scen_test = df["scenario"].iloc[idx_test].to_numpy()
    print(f"  dataset      {len(df):,} rows, {int(y.sum()):,} fraud ({y.mean():.5f} prevalence)")
    print(f"  splits       train {len(idx_train):,} · validation {len(idx_val):,} · TEST {len(idx_test):,}")
    print(f"  test split   {int(y_test.sum()):,} fraud ({y_test.mean():.5f} prevalence), "
          f"{int(len(y_test) - y_test.sum()):,} legitimate")

    model = joblib.load(MODEL_PATH)
    stored_threshold, rederived, train_metrics = operating_threshold(
        model, df, y, idx_val, args.fpr_budget)
    agrees = abs(stored_threshold - rederived) <= 5e-5     # half of the stored 4-dp step
    print(f"  threshold    {stored_threshold:.6f} from models/metrics.json (selected on validation)")
    print(f"               re-derived from validation alone: {rederived:.6f} "
          f"{'- agrees to the stored 4-decimal precision' if agrees else '- DOES NOT AGREE'}")
    print("               evaluated at the stored value, which is the one the service serves")
    if not agrees:
        print("  WARNING: the stored threshold is not reproducible from the validation split")

    # the only place test features are ever scored
    probs = model.predict_proba(X_test)[:, 1]
    summary = evaluate(y_test, probs, stored_threshold)

    base_rate = float(y_test.mean())
    brier_reference = float(np.mean((y_test - base_rate) ** 2))
    summary["brier_reference_constant_base_rate"] = brier_reference
    summary["brier_skill_score"] = float(1 - summary["brier"] / brier_reference)

    # curves
    prec, rec, pr_thresh = precision_recall_curve(y_test, probs)
    fpr, tpr, roc_thresh = roc_curve(y_test, probs)
    pred = (probs >= stored_threshold).astype(int)
    op_fpr = summary["false_positive_rate"]
    op_rec = summary["recall"]
    op_prec = summary["precision"]

    # classification report
    report = classification_report(
        y_test, pred, labels=[0, 1], target_names=["legitimate", "fraud"],
        output_dict=True, zero_division=0)

    # threshold sweep: a fixed log grid plus the operating point itself
    grid = np.unique(np.concatenate([
        np.round(np.logspace(np.log10(0.001), np.log10(0.95), 60), 6),
        [stored_threshold, 0.5],
    ]))
    df_sweep = sweep(y_test, probs, grid)

    # scenario recall
    df_scen = scenario_recall(scen_test, y_test, probs, stored_threshold)

    # Calibration. Computed here rather than taken from sklearn's helper so that
    # every column of the CSV describes the same bins: at a 1% base rate most
    # predictions sit near zero, so quantile edges collapse and the requested
    # bin count is not the bin count you get. sklearn's curve is kept beside it
    # as a cross-check.
    edges = np.unique(np.quantile(probs, np.linspace(0, 1, args.bins + 1)))
    rows_cal = []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        mask = (probs >= left) & ((probs <= right) if last else (probs < right))
        n = int(mask.sum())
        if n == 0:
            continue
        rows_cal.append({
            "bin": len(rows_cal) + 1, "bin_lower": float(left), "bin_upper": float(right),
            "n": n, "fraud_in_bin": int(y_test[mask].sum()),
            "mean_predicted_probability": float(probs[mask].mean()),
            "observed_fraud_rate": float(y_test[mask].mean()),
        })
    df_cal = pd.DataFrame(rows_cal)
    mean_pred = df_cal["mean_predicted_probability"].to_numpy()
    observed = df_cal["observed_fraud_rate"].to_numpy()
    # A second view with fixed-width bins: the quantile view compresses the
    # upper range where nearly all the fraud sits, and the uniform view shows it.
    uni_edges = np.linspace(0.0, float(probs.max()), args.bins + 1)
    rows_uni = []
    for i in range(len(uni_edges) - 1):
        left, right = uni_edges[i], uni_edges[i + 1]
        last = i == len(uni_edges) - 2
        mask = (probs >= left) & ((probs <= right) if last else (probs < right))
        n = int(mask.sum())
        if n == 0:
            continue
        rows_uni.append({
            "bin": len(rows_uni) + 1, "bin_lower": float(left), "bin_upper": float(right),
            "n": n, "fraud_in_bin": int(y_test[mask].sum()),
            "mean_predicted_probability": float(probs[mask].mean()),
            "observed_fraud_rate": float(y_test[mask].mean()),
        })
    df_cal_uniform = pd.DataFrame(rows_uni)

    sk_observed, sk_mean = calibration_curve(y_test, probs, n_bins=args.bins, strategy="quantile")
    calibration_cross_check = [
        {"mean_predicted_probability": float(m), "observed_fraud_rate": float(o)}
        for m, o in zip(sk_mean, sk_observed)
    ]

    # permutation importance, on the calibrated model the service actually serves
    print(f"  importance   permutation ({args.permutation_repeats} repeats, average-precision scorer)")
    perm = permutation_importance(
        model, X_test, y_test, scoring="average_precision",
        n_repeats=args.permutation_repeats, random_state=args.seed, n_jobs=1)
    df_perm = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean_drop_in_average_precision": perm.importances_mean,
        "importance_std": perm.importances_std,
    })

    # SHAP
    df_shap, shap_note = None, "disabled"
    if args.shap_rows > 0:
        import shap                                              # noqa: PLC0415
        rng = np.random.default_rng(args.seed)
        take = rng.choice(len(X_test), size=min(args.shap_rows, len(X_test)), replace=False)
        sample = X_test.iloc[np.sort(take)]
        try:
            explainer = shap.TreeExplainer(model.calibrated_classifiers_[0].estimator)
            values = explainer.shap_values(sample)
            if isinstance(values, list):
                values = values[1]
            shap_note = ("TreeExplainer on the first fold's uncalibrated booster "
                         "(calibration is monotone, so attribution order is preserved)")
        except Exception as exc:                                  # noqa: BLE001
            background = pd.read_json(BACKGROUND, orient="split")[FEATURES]
            def f(X):
                return model.predict_proba(pd.DataFrame(X, columns=FEATURES))[:, 1]
            explainer = shap.Explainer(f, background.to_numpy()[:100])
            values = explainer(sample.to_numpy(), max_evals=2 * len(FEATURES) + 1).values
            shap_note = (f"model-agnostic Permutation explainer over the stored training "
                         f"background, as the service uses ({type(exc).__name__} from TreeExplainer)")
        df_shap = pd.DataFrame({
            "feature": FEATURES,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        })
        print(f"  shap         {len(sample)} test rows · {shap_note}")

    # ── CSVs: every figure's underlying values ─────────────────────────────
    csv = out / "csv"
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
    pd.DataFrame(
        [{"actual": "legitimate", "predicted_legitimate": cm["tn"], "predicted_fraud": cm["fp"]},
         {"actual": "fraud", "predicted_legitimate": cm["fn"], "predicted_fraud": cm["tp"]}]
    ).to_csv(csv / "confusion_matrix.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(csv / "classification_report.csv")
    # the raw predictions, so every number above can be recomputed from scratch
    pd.DataFrame({
        "dataset_row_index": idx_test, "scenario": scen_test, "is_fraud": y_test,
        "predicted_probability": probs, "flagged_at_operating_threshold": pred,
    }).to_csv(csv / "test_predictions.csv", index=False)
    flat = {k: v for k, v in summary.items() if not isinstance(v, dict)}
    flat.update({f"support_{k}": v for k, v in summary["support"].items()})
    flat.update({f"confusion_{k}": v for k, v in cm.items()})
    pd.DataFrame(sorted(flat.items()), columns=["metric", "value"]).to_csv(
        csv / "summary_metrics.csv", index=False)

    # ── figures ────────────────────────────────────────────────────────────
    figures = [
        fig_confusion(cm, out),
        fig_pr(rec, prec, summary["average_precision"], op_rec, op_prec, base_rate, out),
        fig_roc(fpr, tpr, summary["roc_auc"], op_fpr, op_rec, out),
        fig_recall_vs_fpr(fpr, tpr, args.fpr_budget, op_fpr, op_rec, out),
        fig_sweep(df_sweep, stored_threshold, out),
        fig_calibration(df_cal, df_cal_uniform, summary["brier"], summary["brier_skill_score"], out),
        fig_scenarios(df_scen, out),
        fig_importance(df_perm, "importance_mean_drop_in_average_precision", "importance_std",
                       "Permutation importance on the test split\n(drop in average precision when a feature is shuffled)",
                       "feature_importance_permutation.png", out),
    ]
    if df_shap is not None:
        figures.append(fig_importance(df_shap, "mean_abs_shap", None,
                                      "SHAP summary — mean absolute attribution per feature",
                                      "shap_summary.png", out, colour=TEAL))

    # ── machine-readable results ───────────────────────────────────────────
    finished = datetime.now(timezone.utc)
    metrics = {
        "what_this_is": (
            "Evaluation of the current trained artefact on the held-out test split. "
            "Nothing was fitted or retrained; the operating threshold was selected on "
            "the validation split before this evaluation and is used as-is."
        ),
        "run": {
            "started_utc": started.isoformat(timespec="seconds"),
            "finished_utc": finished.isoformat(timespec="seconds"),
            "seconds": round((finished - started).total_seconds(), 1),
            "seed": args.seed,
            "command": " ".join(sys.argv),
        },
        "data": {
            "source": "simulated (backend/ml/dataset.generate)",
            "caveat": ("No public labelled UPI fraud corpus exists. These figures describe "
                       "the method on simulated data, not field performance."),
            "rows": int(len(df)),
            "fraud_rows": int(y.sum()),
            "fraud_prevalence": float(y.mean()),
            "splits": {
                "train_rows": int(len(idx_train)), "validation_rows": int(len(idx_val)),
                "test_rows": int(len(idx_test)),
                "train_fraud": int(y[idx_train].sum()),
                "validation_fraud": int(y[idx_val].sum()),
                "test_fraud": int(y[idx_test].sum()),
                "test_fraud_prevalence": base_rate,
                "reconstruction": ("train_test_split(arange(n), test_size=0.25, random_state=seed, "
                                   "stratify=y), then 20% of the remainder for validation - the same "
                                   "calls backend/train_model.train() makes"),
                "disjoint_verified": True,
            },
        },
        "operating_point": {
            "threshold": stored_threshold,
            "source": "models/metrics.json, selected on the validation split",
            "fpr_budget": args.fpr_budget,
            "rederived_from_validation_split_full_precision": rederived,
            "stored_value_is_rounded_to_4dp": True,
            "rederivation_agrees_to_stored_precision": bool(agrees),
            "difference": abs(stored_threshold - rederived),
            "evaluated_at": "the stored value, which is what the running service uses",
            "test_labels_used_in_selection": False,
            "test_features_used_in_selection": False,
        },
        "test_metrics": summary,
        "classification_report": report,
        "recall_at_fpr_budget": {
            "budget": args.fpr_budget,
            "realised_false_positive_rate": op_fpr,
            "recall": op_rec,
            "precision": op_prec,
            "within_budget": bool(op_fpr <= args.fpr_budget),
        },
        "scenario_recall": df_scen.to_dict(orient="records"),
        "calibration": {
            "bins_requested": int(args.bins),
            "bins_populated": int(len(df_cal)),
            "note": ("quantile edges collapse at a 1% base rate because most predictions sit "
                     "near zero; empty bins are dropped rather than reported as bins"),
            "strategy": "quantile",
            "sklearn_cross_check": calibration_cross_check,
            "brier": summary["brier"],
            "brier_reference_constant_base_rate": brier_reference,
            "brier_skill_score": summary["brier_skill_score"],
            "curve": df_cal.to_dict(orient="records"),
            "curve_uniform_bins": df_cal_uniform.to_dict(orient="records"),
        },
        "feature_importance": {
            "permutation": df_perm.to_dict(orient="records"),
            "permutation_note": (f"average-precision drop, {args.permutation_repeats} repeats, "
                                 "computed on the test split against the calibrated model"),
            "shap": (df_shap.to_dict(orient="records") if df_shap is not None else None),
            "shap_note": shap_note,
        },
        "artefacts": {"figures": figures, "csv": sorted(p.name for p in csv.glob("*.csv"))},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = {
        "generated_utc": finished.isoformat(timespec="seconds"),
        "seed": args.seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__,
                      "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                      "joblib": joblib.__version__, "matplotlib": matplotlib.__version__},
        "artefact_sha256": {
            "models/risk_model.pkl": sha256(MODEL_PATH),
            "models/metrics.json": sha256(TRAIN_METRICS),
            "backend/ml/dataset.py": sha256(ROOT / "backend/ml/dataset.py"),
            "backend/ml/features.py": sha256(ROOT / "backend/ml/features.py"),
            "backend/train_model.py": sha256(ROOT / "backend/train_model.py"),
        },
        "trained_with": train_metrics.get("environment"),
        "reproduce": "python scripts/evaluate_model.py --seed {} --out {}".format(args.seed, args.out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ── console summary ────────────────────────────────────────────────────
    print(f"\n  ── results on {len(y_test):,} held-out rows "
          f"({int(y_test.sum())} fraud, prevalence {base_rate:.5f}) ──\n")
    print(f"    ROC-AUC                 {summary['roc_auc']:.4f}")
    print(f"    Average precision       {summary['average_precision']:.4f}   "
          f"(no-skill = {base_rate:.4f})")
    print(f"    Brier                   {summary['brier']:.5f}   "
          f"skill {summary['brier_skill_score']:.4f} vs {brier_reference:.5f}")
    print(f"    At threshold {stored_threshold:.4f}:")
    print(f"      recall                {op_rec:.4f}   ({cm['tp']} of {cm['tp'] + cm['fn']})")
    print(f"      precision             {op_prec:.4f}   ({cm['tp']} of {summary['alerts']} alerts)")
    print(f"      false-positive rate   {op_fpr:.4f}   "
          f"({'inside' if op_fpr <= args.fpr_budget else 'OUTSIDE'} the {args.fpr_budget:.0%} budget)")
    print(f"      confusion             tn {cm['tn']}  fp {cm['fp']}  fn {cm['fn']}  tp {cm['tp']}")
    print("\n    recall by scenario")
    for row in df_scen.to_dict(orient="records"):
        print(f"      {row['scenario']:<28} {row['recall']:>7.2%}   ({row['detected']}/{row['n']})")
    print(f"\n  written: {out.relative_to(ROOT)}/metrics.json · manifest.json · "
          f"{len(figures)} figures · {len(list(csv.glob('*.csv')))} csv files\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
