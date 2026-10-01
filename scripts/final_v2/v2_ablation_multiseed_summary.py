"""Summaries of the V2 leave-family-out ablation and the V2 ten-seed stability run.

Input: results/final_v2/ablation/v2_family_ablation_results.csv (130 fits, seeds
42-51, written by v2_family_ablation.py). The 'A. Full V2' rows of that file ARE
the ten-seed stability experiment: same protocol, one fit per seed, threshold
from validation only, one test measurement.

Statistics are across the ten seed-level values (sample sd, ddof = 1); test
sets are never pooled; no seed is selected. Deltas are paired by seed
(configuration minus full V2 on the same seed).
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

R = ROOT / "results" / "final_v2"
SC = ["social_engineering", "mule_collection", "account_takeover", "card_testing"]
SCL = {"social_engineering": "SE", "mule_collection": "MC", "account_takeover": "AT", "card_testing": "CT"}


def d(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}


def md(df, path, title, note=""):
    path.write_text(f"**{title}**\n\n" + df.to_markdown(index=False) + (f"\n\n{note}\n" if note else "\n"), encoding="utf-8")
    df.to_csv(path.with_suffix(".csv"), index=False)


def main() -> int:
    a = pd.read_csv(R / "ablation/v2_family_ablation_results.csv")
    assert a.groupby("config").size().eq(10).all() and a.seed.nunique() == 10
    full = a[a.config == "A. Full V2"].set_index("seed").sort_index()

    # ── ten-seed stability (full V2) ─────────────────────────────────────
    (R / "multi_seed").mkdir(parents=True, exist_ok=True)
    keep = ["n", "n_fraud", "threshold", "pr_auc", "roc_auc", "recall", "fpr", "precision", "f1", "brier", "bss",
            "tp", "fp", "fn", "tn", "val_fpr", "val_recall"] + [f"recall_{s}" for s in SC] + [f"n_{s}" for s in SC]
    full[keep].reset_index().to_csv(R / "multi_seed/v2_ten_seed_results.csv", index=False)
    metrics = [("pr_auc", "PR-AUC", 1, 4), ("roc_auc", "ROC-AUC", 1, 4), ("recall", "Recall at ~1% FPR (%)", 100, 2),
               ("fpr", "Realised test FPR (%)", 100, 3), ("precision", "Precision (%)", 100, 2), ("f1", "F1", 1, 4),
               ("brier", "Brier score", 1, 5), ("bss", "Brier skill score", 1, 4), ("threshold", "Validation threshold", 1, 4)] + \
              [(f"recall_{s}", f"Recall — {s.replace('_', ' ')} (%)", 100, 2) for s in SC]
    rows, summ = [], {}
    for k, lab, sc, dec in metrics:
        s = d(full[k] * sc); summ[k] = d(full[k])
        rows.append({"Metric": lab, "Mean": f"{s['mean']:.{dec}f}", "SD": f"{s['sd']:.{dec}f}", "Min": f"{s['min']:.{dec}f}",
                     "Max": f"{s['max']:.{dec}f}", "Seed 42 (original run)": f"{sc*full.loc[42, k]:.{dec}f}"})
    t8 = pd.DataFrame(rows)
    md(t8, R / "tables/table08_ten_seed_stability.md", "Table 8 — V2 ten-seed stability (seeds 42–51; test split of 30,000 per seed)",
       "Each seed regenerates the corpus, re-splits (stratified, random_state = seed), refits the calibrated V2 model, selects the threshold "
       "on its own validation split (FPR ≤ 1%) and measures the test split once. Statistics are over the ten seed-level values (sd with ddof = 1). "
       "Seed 42 reproduces the frozen V2 artefact exactly.")
    over = {"test_fpr_above_1pct_seeds": int((full.fpr > 0.01).sum()), "max_test_fpr": float(full.fpr.max()),
            "val_fpr_all_le_1pct": bool((full.val_fpr <= 0.01 + 1e-12).all()),
            "seed42_rank_by_pr_auc_desc": int(full.pr_auc.rank(ascending=False)[42]),
            "test_fraud_per_seed": full.n_fraud.astype(int).to_dict()}
    json.dump({"summary": summ, "notes": over}, open(R / "multi_seed/v2_ten_seed_summary.json", "w"), indent=2, default=float)

    fig, axes = plt_subplots(1, 4, (10.2, 3.1))
    for ax, (k, lab) in zip(axes, [("pr_auc", "PR-AUC"), ("recall", "Recall at ~1% FPR"), ("fpr", "Realised test FPR"),
                                  ("precision", "Precision")]):
        v = full[k].to_numpy()
        ax.boxplot(v, widths=0.45, showmeans=True, showfliers=False, medianprops=dict(color=style.INK),
                   meanprops=dict(marker="D", markerfacecolor=style.INK, markeredgecolor=style.INK, markersize=4))
        ax.scatter(1 + np.linspace(-0.14, 0.14, len(v)), v, s=16, color=style.BLUE, zorder=3)
        ax.scatter([1.28], [full.loc[42, k]], s=40, marker="*", color=style.ORANGE, zorder=4, label="seed 42")
        if k == "fpr":
            ax.axhline(0.01, color=style.MUTED, ls="--", lw=1)
        ax.set_xticks([]); ax.set_title(lab)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("V2 across ten independently generated corpora (seeds 42–51)", fontsize=10)
    style.save(fig, R / "figures/fig10_ten_seed_stability")

    # ── ablation ─────────────────────────────────────────────────────────
    cfgs = list(dict.fromkeys(a.config))
    mcols = ["roc_auc", "pr_auc", "recall", "fpr", "precision", "f1", "brier"] + [f"recall_{s}" for s in SC]
    rows, deltas = [], []
    for c in cfgs:
        g = a[a.config == c].set_index("seed").sort_index()
        dd = {m: d(g[m] - full[m]) for m in mcols}
        pos = {m: int(((g[m] - full[m]) > 0).sum()) for m in mcols}
        neg = {m: int(((g[m] - full[m]) < 0).sum()) for m in mcols}
        deltas.append({"config": c, "n_features": int(g.n_features.iloc[0]),
                       **{f"{m}_mean": float(g[m].mean()) for m in mcols}, **{f"{m}_sd": float(g[m].std(ddof=1)) for m in mcols},
                       **{f"delta_{m}_mean": dd[m]["mean"] for m in mcols}, **{f"delta_{m}_sd": dd[m]["sd"] for m in mcols},
                       **{f"delta_{m}_seeds_lower": neg[m] for m in mcols}, **{f"delta_{m}_seeds_higher": pos[m] for m in mcols},
                       **{f"seed42_{m}": float(g.loc[42, m]) for m in mcols}})
        full_row = c == "A. Full V2"
        rows.append({"Configuration": c, "F": int(g.n_features.iloc[0]),
                     "PR-AUC": f"{g.pr_auc.mean():.4f} ± {g.pr_auc.std(ddof=1):.4f}",
                     "ΔPR-AUC (pp)": "—" if full_row else f"{100*dd['pr_auc']['mean']:+.2f} ± {100*dd['pr_auc']['sd']:.2f}",
                     "ROC-AUC": f"{g.roc_auc.mean():.4f}", "ΔROC-AUC (pp)": "—" if full_row else f"{100*dd['roc_auc']['mean']:+.2f}",
                     "Recall %": f"{100*g.recall.mean():.2f}", "ΔRecall (pp)": "—" if full_row else f"{100*dd['recall']['mean']:+.2f}",
                     "FPR %": f"{100*g.fpr.mean():.3f}", "Precision %": f"{100*g.precision.mean():.2f}",
                     "F1": f"{g.f1.mean():.4f}", "Brier": f"{g.brier.mean():.5f}",
                     **{f"{SCL[s]} recall %": f"{100*g['recall_'+s].mean():.1f}" for s in SC},
                     "Seeds lower PR-AUC": "—" if full_row else f"{neg['pr_auc']}/10"})
    t7 = pd.DataFrame(rows)
    (R / "ablation").mkdir(exist_ok=True)
    pd.DataFrame(deltas).to_csv(R / "ablation/v2_family_ablation_summary.csv", index=False)
    md(t7, R / "tables/table07_v2_family_ablation.md",
       "Table 7 — V2 feature-family ablation (leave-family-out; mean over seeds 42–51)",
       "Every configuration is refitted with the identical protocol (same splits, learner, calibration and validation-only threshold rule). "
       "Δ = configuration minus full V2 on the same seed (paired), mean ± sd over seeds, in percentage points. Families follow "
       "scripts/final_v2/families.py. Contextual = message + link + intent; Relational = relationship + graph. SE, MC, AT, CT: synthetic "
       "scenario recall. Differences are associations under the simulator, not causal effects.")
    # seed-42-only table for the original split
    s42 = a[a.seed == 42].set_index("config").loc[cfgs]
    t7b = pd.DataFrame([{"Configuration": c, "F": int(r.n_features), "PR-AUC": f"{r.pr_auc:.4f}", "ROC-AUC": f"{r.roc_auc:.4f}",
                         "Recall": f"{r.recall:.4f}", "FPR": f"{r.fpr:.5f}", "Precision": f"{r.precision:.4f}", "F1": f"{r.f1:.4f}",
                         "Brier": f"{r.brier:.5f}", **{f"{SCL[s]} recall": f"{r['recall_'+s]:.4f}" for s in SC}}
                        for c, r in s42.iterrows()])
    md(t7b, R / "tables/table07b_v2_family_ablation_seed42.md", "Table 7b — V2 feature-family ablation on the original seed-42 split",
       "Single split; see Table 7 for the ten-seed version.")

    dfd = pd.DataFrame(deltas).iloc[1:]
    fig, ax = plt_subplots(1, 1, (7.2, 4.4))
    yy = np.arange(len(dfd))[::-1]
    ax.barh(yy, 100 * dfd.delta_pr_auc_mean, xerr=100 * dfd.delta_pr_auc_sd, color=style.BLUE, height=0.6,
            error_kw=dict(lw=0.8, capsize=2))
    for yv, v, nf in zip(yy, 100 * dfd.delta_pr_auc_mean, dfd.n_features):
        ax.text(v - 0.6, yv, f"{v:+.1f} pp  (F={nf})", va="center", ha="right", fontsize=7.5, color=style.INK)
    ax.set_yticks(yy); ax.set_yticklabels(dfd.config, fontsize=8); ax.axvline(0, color=style.INK, lw=0.8)
    ax.set_xlabel("Change in test PR-AUC vs full V2 (pp), mean ± sd over 10 seeds")
    ax.set_title(f"V2 leave-family-out ablation (full V2 PR-AUC {full.pr_auc.mean():.4f} ± {full.pr_auc.std(ddof=1):.4f})")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(100 * dfd.delta_pr_auc_mean - 100 * dfd.delta_pr_auc_sd) - 9, 3)
    style.save(fig, R / "figures/fig09_v2_ablation")
    print(t8.to_string(index=False)); print(t7.to_string(index=False)); print(json.dumps(over, indent=1))
    return 0


def plt_subplots(r, c, size):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(r, c, figsize=size)
    return fig, axes


if __name__ == "__main__":
    raise SystemExit(main())
