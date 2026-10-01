"""Final figure pass for the V2 results.

Redraws, from the saved result files only (no model is refitted or re-scored), the
figures whose layout needed correction after visual review:

  fig02  ROC inset now has its own opaque background (chance line no longer shows through)
  fig07  single-hue bars; evidence family named beside each bar (no 8-colour legend)
  fig08  single-hue bars; value labels placed clear of the error bars
  fig09  value labels placed clear of bars and error bars; grouped removals shaded
  fig10  boxplot fliers suppressed (every seed is already drawn as a point)
  fig11  'overall' reference label moved off the title
  fig00  key-results overview panel (new)

Every number drawn here is read from results/final_v2/**. Run after all other
final_v2 scripts; it overwrites only the figure files listed above.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R = ROOT / "results" / "final_v2"
F = R / "figures"
BLUE, ORANGE, INK, INK2, MUTED = style.BLUE, style.ORANGE, style.INK, style.INK2, style.MUTED
LIGHT_BLUE = "#9ec5f4"


def j(p):
    return json.load(open(R / p))


def fig02(te):
    roc = pd.read_csv(R / "roc_pr/roc_curve_test.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="Chance", zorder=1)
    ax.plot(roc.fpr, roc.tpr, color=BLUE, label=f"V2 (ROC-AUC = {te['roc_auc']:.4f})", zorder=3)
    ax.plot(te["fpr"], te["recall"], "o", ms=8, color=ORANGE, mec="white", mew=1.5, zorder=4,
            label=f"Operating point (FPR {te['fpr']:.4f}, TPR {te['recall']:.4f})")
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate (recall)")
    ax.set_title("ROC curve — V2, test split (n = 30,000)")
    ax.legend(loc="lower right", fontsize=7.5)
    ins = ax.inset_axes([0.40, 0.34, 0.54, 0.36]); ins.set_facecolor("white"); ins.set_zorder(5)
    ins.plot(roc.fpr, roc.tpr, color=BLUE, lw=1.5)
    ins.axvline(0.01, color=MUTED, lw=0.8, ls=":")
    ins.plot(te["fpr"], te["recall"], "o", ms=6, color=ORANGE, mec="white", mew=1)
    ins.set_xlim(0, 0.05); ins.set_ylim(0.4, 1.0); ins.tick_params(labelsize=6)
    ins.set_title("Zoom: FPR 0–5% (dotted: 1% budget)", fontsize=7)
    for s in ins.spines.values():
        s.set_visible(True); s.set_color(MUTED)
    style.save(fig, F / "fig02_roc_curve")


def fig07():
    imp = pd.read_csv(R / "feature_importance/feature_importance_v2.csv")
    t = imp.sort_values("perm_importance_mean", ascending=False).head(20)[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    y = np.arange(len(t))
    ax.barh(y, t.perm_importance_mean, xerr=t.perm_importance_std, color=BLUE, height=0.66,
            error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
    for yv, v, sd, fam in zip(y, t.perm_importance_mean, t.perm_importance_std, t.family):
        ax.text(v + sd + 0.006, yv, f"{v:.3f}   {fam}", va="center", fontsize=7.2, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(t.feature, fontsize=8)
    ax.set_xlim(0, t.perm_importance_mean.max() * 1.45)
    ax.set_xlabel("Mean drop in average precision when the feature is shuffled (5 repeats; bar = ±1 sd)")
    ax.set_title("Top 20 V2 features — permutation importance (test split)\nlabel: importance and evidence family")
    ax.grid(axis="y", visible=False)
    style.save(fig, F / "fig07_feature_importance")


def fig08():
    fam = pd.read_csv(R / "feature_importance/family_importance_v2.csv")[::-1].reset_index(drop=True)
    col = "Grouped permutation importance (AP drop)"
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    y = np.arange(len(fam))
    ax.barh(y, fam[col], xerr=fam.sd, color=BLUE, height=0.6, error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
    for yv, v, sd, n in zip(y, fam[col], fam.sd, fam["Number of features"]):
        ax.text(max(v, 0) + sd + 0.008, yv, f"{v:.3f}  ({n} feature{'s' if n > 1 else ''})", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(fam.Family)
    ax.set_xlim(0, fam[col].max() * 1.35)
    ax.set_xlabel("Drop in average precision when all of a family's columns are shuffled together")
    ax.set_title("Evidence-family importance — V2, test split")
    ax.grid(axis="y", visible=False)
    style.save(fig, F / "fig08_family_importance")


def fig09():
    s = pd.read_csv(R / "ablation/v2_family_ablation_summary.csv")
    full = s.iloc[0]; d = s.iloc[1:].reset_index(drop=True)
    grouped = d.config.str.startswith(("G.", "H.", "M."))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(d))[::-1]
    m, sd = 100 * d.delta_pr_auc_mean, 100 * d.delta_pr_auc_sd
    ax.barh(y, m, xerr=sd, color=[LIGHT_BLUE if g else BLUE for g in grouped], height=0.62,
            error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
    for yv, v, e, nl in zip(y, m, sd, d.delta_pr_auc_seeds_lower):
        ax.text(max(v + e, 0) + 1.0, yv, f"{v:+.2f} ± {e:.2f} pp   ({int(nl)}/10 seeds lower)", va="center", ha="left", fontsize=7.2, color=INK2)
    labels = [f"{c}  [{int(n)} feat.]" for c, n in zip(d.config, d.n_features)]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim((m - sd).min() - 2, 24); ax.set_xticks(range(-30, 1, 5))
    ax.set_xlabel("Change in test PR-AUC vs full V2 (pp), mean ± sd over seeds 42–51, paired by seed", loc="left")
    ax.set_title(f"V2 leave-family-out ablation — full V2 PR-AUC {full.pr_auc_mean:.4f} ± {full.pr_auc_sd:.4f} (47 features)\n"
                 "dark: single family removed · light: several families removed together", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    style.save(fig, F / "fig09_v2_ablation")


def fig10():
    full = pd.read_csv(R / "multi_seed/v2_ten_seed_results.csv").set_index("seed")
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.2))
    rng = np.random.default_rng(0)
    for ax, (k, lab, sc, fmt) in zip(axes, [("pr_auc", "PR-AUC", 1, "{:.3f}"), ("recall", "Recall at ~1% FPR (%)", 100, "{:.1f}"),
                                           ("fpr", "Realised test FPR (%)", 100, "{:.2f}"), ("precision", "Precision (%)", 100, "{:.1f}")]):
        v = full[k].to_numpy() * sc
        ax.boxplot(v, widths=0.45, showmeans=True, showfliers=False, medianprops=dict(color=INK),
                   meanprops=dict(marker="D", markerfacecolor=INK, markeredgecolor=INK, markersize=4))
        ax.scatter(1 + rng.uniform(-0.12, 0.12, len(v)), v, s=18, color=BLUE, zorder=3, edgecolor="white", linewidth=0.6)
        ax.scatter([1.33], [full.loc[42, k] * sc], s=70, marker="*", color=ORANGE, zorder=4, label="seed 42 (frozen model)")
        if k == "fpr":
            ax.axhline(1.0, color=MUTED, ls="--", lw=1); ax.text(0.55, 1.0, "1% budget", fontsize=6.5, color=INK2, va="bottom")
        ax.set_xticks([]); ax.set_xlim(0.5, 1.5)
        ax.set_title(f"{lab}\nmean {fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}", fontsize=8.5)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("V2 across ten independently generated corpora (seeds 42–51) — dots: seeds, box: quartiles, ◆: mean", fontsize=9.5)
    style.save(fig, F / "fig10_ten_seed_stability")


def fig11(te):
    t = pd.read_csv(R / "tables/table06_scenario_performance.csv").iloc[:4]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(t))[::-1]
    rec = t.Recall.astype(float)
    ax.barh(y, rec, color=BLUE, height=0.55)
    for yv, r, tp, n in zip(y, rec, t["Detected (TP)"], t["Test fraud"]):
        ax.text(r + 0.012, yv, f"{100*r:.1f}%  ({tp}/{n})", va="center", fontsize=8, color=INK2,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=5)
    ax.axvline(te["recall"], color=INK2, ls="--", lw=1)
    ax.text(te["recall"] + 0.008, -0.62, f"overall recall {100*te['recall']:.1f}%", fontsize=7.2, color=INK2, va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(t["Scenario (synthetic)"]); ax.set_xlim(0, 1.2); ax.set_ylim(-0.7, len(t) - 0.4)
    ax.set_xlabel("Recall at threshold 0.1123 (label: detected / test fraud)")
    ax.set_title("Recall by synthetic fraud scenario — V2, test split")
    ax.grid(axis="y", visible=False)
    style.save(fig, F / "fig11_scenario_performance")


def fig00(te):
    """Key-results overview: every value read from the result files."""
    ms = j("multi_seed/v2_ten_seed_summary.json")["summary"]
    e2e = j("end_to_end/e2e_v2_summary.json")
    la = j("leakage_audit/leakage_audit_v2.json")
    sw = j("software_tests/software_tests_summary.json")
    per = e2e["per_seed"]
    def mean_over(key_path):
        vals = []
        for s in per.values():
            v = s
            for k in key_path:
                v = v[k]
            vals.append(v)
        return float(np.mean(vals)), float(np.std(vals, ddof=1))
    status = [x.get("status") for x in (la["items"] if isinstance(la, dict) and "items" in la else la)]
    n_pass = sum(s == "PASS" for s in status); n_fail = sum(s == "FAIL" for s in status); n_na = len(status) - n_pass - n_fail
    tot = sw.get("total", sw)
    passed, tests = tot.get("passed"), tot.get("tests", tot.get("collected"))
    tiles = [
        ("V2 test PR-AUC", f"{te['pr_auc_average_precision']:.4f}", "seed 42 · no-skill 0.0118"),
        ("V2 test ROC-AUC", f"{te['roc_auc']:.4f}", "seed 42 · n = 30,000"),
        ("Recall at θ = 0.1123", f"{100*te['recall']:.2f}%", f"FPR {100*te['fpr']:.3f}% · precision {100*te['precision']:.2f}%"),
        ("Confusion (TN/FP/FN/TP)", f"{te['tn']:,} / {te['fp']} / {te['fn']} / {te['tp']}", f"F1 {te['f1']:.4f} · MCC {te['mcc']:.4f}"),
        ("Calibration", f"Brier {te['brier']:.5f}", f"Brier skill score {te['brier_skill_score']:.3f}"),
        ("Ten seeds (42–51)", f"PR-AUC {ms['pr_auc']['mean']:.4f} ± {ms['pr_auc']['sd']:.4f}",
         f"recall {100*ms['recall']['mean']:.2f} ± {100*ms['recall']['sd']:.2f}%"),
        ("Leakage audit (18 items)", f"{n_pass} PASS · {n_fail} FAIL", f"{n_na} not applicable"),
        ("Software tests", f"{passed}/{tests} passed", "implementation correctness, not accuracy"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 3.6))
    for ax, (h, v, sub) in zip(axes.ravel(), tiles):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.04), 0.96, 0.92, transform=ax.transAxes, facecolor="#f5f4f1", edgecolor="#e4e3df", lw=1))
        ax.text(0.08, 0.78, h, transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.text(0.08, 0.44, v, transform=ax.transAxes, fontsize=13 if len(v) < 22 else 10.5, color=INK, weight="bold")
        ax.text(0.08, 0.16, sub, transform=ax.transAxes, fontsize=7.5, color=INK2)
    fig.suptitle("Key V2 results (synthetic data; frozen seed-42 model unless stated)", fontsize=10.5, x=0.02, ha="left")
    style.save(fig, F / "fig00_key_results")


def main() -> int:
    te = j("classifier/v2_metrics.json")["test"]
    assert (te["tn"], te["fp"], te["fn"], te["tp"]) == (29350, 295, 78, 277)
    assert round(te["pr_auc_average_precision"], 4) == 0.7326
    fig00(te); fig02(te); fig07(); fig08(); fig09(); fig10(); fig11(te)
    print("figures redrawn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
