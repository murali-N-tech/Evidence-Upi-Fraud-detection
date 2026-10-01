"""Experiments 1 and 2 - analysis of results/ablation/ablation_results.csv.

The seed is the unit of replication. Nothing here pools test sets: every
statistic is computed per seed first, then summarised ACROSS seeds (mean,
sample std with ddof=1, min, max, and a 95% t-interval over the 10 seed-level
values). The t-interval describes run-to-run variation of the whole
generate-split-train-select pipeline; it is not a confidence interval for
deployment performance.

Experiment 1 (stability): V1 = configuration A, V2 = configuration K.
Experiment 2 (ablation): all configurations A-K; differences are reported in
percentage points, paired by seed (config minus reference on the same seed).

Outputs
  results/multiseed/multiseed_results.csv, multiseed_summary.json
  results/ablation/ablation_summary.json, ablation_paired_deltas.csv
  results/tables/table_A_multiseed.{csv,md}, table_B_ablation.{csv,md},
  results/tables/table_C_ablation_scenarios.{csv,md}
  results/figures/fig1_multiseed_prauc.png, fig2_multiseed_operating.png,
  results/figures/fig3_ablation_prauc.png, fig4_ablation_scenarios.png
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, SCENARIOS, environment, sha256_file, source_state, write_json  # noqa: E402

ABL = RESULTS / "ablation"
MS = RESULTS / "multiseed"
TAB = RESULTS / "tables"
FIG = RESULTS / "figures"

METRICS = ["pr_auc", "roc_auc", "recall", "fpr", "precision", "f1", "brier", "bss", "threshold",
           "tp", "fp", "fn", "tn"] + [f"recall_{s}" for s in SCENARIOS]
PP = {"pr_auc", "roc_auc", "recall", "fpr", "precision", "f1", "bss"} | {f"recall_{s}" for s in SCENARIOS}


def describe(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    out = {"n_seeds": n, "mean": float(v.mean()), "std": float(v.std(ddof=1)) if n > 1 else 0.0,
           "min": float(v.min()), "max": float(v.max()), "median": float(np.median(v))}
    if n > 1:
        half = stats.t.ppf(0.975, n - 1) * out["std"] / np.sqrt(n)
        out["t95_low"], out["t95_high"] = out["mean"] - half, out["mean"] + half
    return out


def fmt(d: dict, pp: bool, dec=4) -> str:
    if pp:
        return f"{100*d['mean']:.2f} ± {100*d['std']:.2f}"
    return f"{d['mean']:.{dec}f} ± {d['std']:.{dec}f}"


def main() -> int:
    df = pd.read_csv(ABL / "ablation_results.csv")
    seeds = sorted(df.seed.unique())
    configs = sorted(df.config.unique())
    # completeness: every (seed, config) exactly once
    counts = df.groupby(["seed", "config"]).size()
    assert (counts == 1).all(), "duplicate (seed, config) rows"
    assert len(counts) == len(seeds) * len(configs), "missing (seed, config) rows"

    # ── Experiment 1 ────────────────────────────────────────────────────
    v1 = df[df.config.str.startswith("A_")].set_index("seed").sort_index()
    v2 = df[df.config.str.startswith("K_")].set_index("seed").sort_index()
    keep = ["n", "n_fraud", "n_legit"] + METRICS + [f"n_{s}" for s in SCENARIOS]
    ms = pd.concat([v1[keep].assign(model="V1"), v2[keep].assign(model="V2")]).reset_index()
    ms = ms[["seed", "model"] + keep].sort_values(["seed", "model"])
    MS.mkdir(parents=True, exist_ok=True)
    ms.to_csv(MS / "multiseed_results.csv", index=False)

    summ = {"V1": {}, "V2": {}, "paired_V2_minus_V1": {}}
    for m in METRICS:
        summ["V1"][m] = describe(v1[m].to_numpy())
        summ["V2"][m] = describe(v2[m].to_numpy())
        d = (v2[m] - v1[m]).to_numpy()
        pd_ = describe(d)
        pd_["seeds_V2_better"] = int(((d > 0) if m not in ("fpr", "brier", "fp", "fn") else (d < 0)).sum())
        if len(d) > 1 and np.nanstd(d) > 0:
            pd_["paired_t_p"] = float(stats.ttest_1samp(d[~np.isnan(d)], 0.0).pvalue)
            try:
                pd_["wilcoxon_p"] = float(stats.wilcoxon(d[~np.isnan(d)]).pvalue)
            except ValueError:
                pass
        summ["paired_V2_minus_V1"][m] = pd_
    # threshold behaviour: realised test FPR vs the 1% validation budget
    summ["fpr_budget_check"] = {
        "budget": 0.01,
        "V1_seeds_test_fpr_above_budget": int((v1.fpr > 0.01).sum()),
        "V2_seeds_test_fpr_above_budget": int((v2.fpr > 0.01).sum()),
        "V1_max_test_fpr": float(v1.fpr.max()), "V2_max_test_fpr": float(v2.fpr.max()),
        "note": "threshold chosen on validation only; test FPR can exceed 1% by sampling variation",
    }
    summ["seed42_reproduction"] = {
        "V1": {k: float(v1.loc[42, k]) for k in ("pr_auc", "tp", "fp", "fn", "tn")},
        "V2_config_K": {k: float(v2.loc[42, k]) for k in ("pr_auc", "roc_auc", "tp", "fp", "fn", "tn", "brier")},
        "frozen_reference": {"V1": {"pr_auc": 0.4860, "tp": 175, "fp": 280, "fn": 180, "tn": 29365},
                             "V2": {"pr_auc": 0.7326, "tp": 277, "fp": 295, "fn": 78, "tn": 29350}},
    }
    summ["dataset_per_seed"] = {int(s): {"n_test": int(v1.loc[s, "n"]), "test_fraud": int(v1.loc[s, "n_fraud"])}
                                for s in seeds}
    write_json(MS / "multiseed_summary.json", {
        "what": "Experiment 1: 10-seed stability of V1 (config A) and V2 (config K). "
                "Each seed regenerates the corpus, re-splits, refits and re-selects the "
                "threshold on its own validation split; statistics are across seed-level values.",
        "seeds": [int(s) for s in seeds],
        "interval_note": "t95 = mean ± t(0.975, n-1)·sd/√n over seeds; describes pipeline "
                         "run-to-run variation on this simulator, not deployment uncertainty.",
        "summary": summ,
        "input_sha256": sha256_file(ABL / "ablation_results.csv"),
        "environment": environment(), "source": source_state(),
    })

    # ── Experiment 2 ────────────────────────────────────────────────────
    per_cfg = {}
    for c in configs:
        sub = df[df.config == c].set_index("seed").sort_index()
        per_cfg[c] = {"n_features": int(sub.n_features.iloc[0]), "groups": sub.groups.iloc[0],
                      "metrics": {m: describe(sub[m].to_numpy()) for m in METRICS}}
    refs = {"vs_A_V1": next(c for c in configs if c.startswith("A_")),
            "vs_B_core": next(c for c in configs if c.startswith("B_")),
            "vs_K_full": next(c for c in configs if c.startswith("K_"))}
    delta_rows = []
    for c in configs:
        sub = df[df.config == c].set_index("seed").sort_index()
        per_cfg[c]["paired_deltas_pp"] = {}
        for rname, r in refs.items():
            ref = df[df.config == r].set_index("seed").sort_index()
            block = {}
            for m in [x for x in METRICS if x in PP] + ["brier"]:
                d = (sub[m] - ref[m]).to_numpy()
                scale = 100.0 if m in PP else 1.0
                desc = describe(d * scale)
                desc["seeds_positive"] = int((d > 0).sum()); desc["seeds_negative"] = int((d < 0).sum())
                block[m] = desc
                delta_rows.append({"config": c, "reference": r, "metric": m,
                                   "unit": "pp" if m in PP else "raw",
                                   **{k: desc[k] for k in ("mean", "std", "min", "max")},
                                   "seeds_positive": desc["seeds_positive"],
                                   "seeds_negative": desc["seeds_negative"]})
            per_cfg[c]["paired_deltas_pp"][rname] = block
    pd.DataFrame(delta_rows).to_csv(ABL / "ablation_paired_deltas.csv", index=False)
    write_json(ABL / "ablation_summary.json", {
        "what": "Experiment 2: feature-family ablation. Same seeds, same splits, same learner and "
                "threshold rule for every configuration; only the feature columns differ.",
        "language_note": "Differences are associations under this simulator and protocol; "
                         "they are not causal effects of the feature families.",
        "seeds": [int(s) for s in seeds], "configs": per_cfg,
        "input_sha256": sha256_file(ABL / "ablation_results.csv"),
    })

    # ── Tables ──────────────────────────────────────────────────────────
    TAB.mkdir(parents=True, exist_ok=True)
    rows_a = []
    labels = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "recall": "Recall @ val. thr.",
              "fpr": "FPR", "precision": "Precision", "f1": "F1", "brier": "Brier", "bss": "BSS",
              **{f"recall_{s}": f"Recall – {s.replace('_', ' ')}" for s in SCENARIOS}}
    for m, lab in labels.items():
        a, b, d = summ["V1"][m], summ["V2"][m], summ["paired_V2_minus_V1"][m]
        pp = m in PP; dec = 5 if m == "brier" else 4
        conv = (lambda x: 100 * x) if pp else (lambda x: x)
        unit = "%" if pp else ""
        rows_a.append({
            "Metric": lab,
            "V1 mean ± sd": fmt(a, pp, dec) + unit, "V1 [min, max]": f"[{conv(a['min']):.{2 if pp else dec}f}, {conv(a['max']):.{2 if pp else dec}f}]",
            "V2 mean ± sd": fmt(b, pp, dec) + unit, "V2 [min, max]": f"[{conv(b['min']):.{2 if pp else dec}f}, {conv(b['max']):.{2 if pp else dec}f}]",
            "Δ (V2−V1) mean ± sd": (f"{100*d['mean']:+.2f} ± {100*d['std']:.2f} pp" if pp
                                   else f"{d['mean']:+.{dec}f} ± {d['std']:.{dec}f}"),
            "Seeds V2 better": f"{d['seeds_V2_better']}/{d['n_seeds']}",
        })
    ta = pd.DataFrame(rows_a)
    ta.to_csv(TAB / "table_A_multiseed.csv", index=False)
    (TAB / "table_A_multiseed.md").write_text(ta.to_markdown(index=False), encoding="utf-8")

    rows_b, rows_c = [], []
    for c in configs:
        mt = per_cfg[c]["metrics"]; dB = per_cfg[c]["paired_deltas_pp"]["vs_B_core"]
        dA = per_cfg[c]["paired_deltas_pp"]["vs_A_V1"]
        rows_b.append({
            "Config": c, "Features": per_cfg[c]["n_features"],
            "PR-AUC": fmt(mt["pr_auc"], False), "ROC-AUC": fmt(mt["roc_auc"], False),
            "Recall %": fmt(mt["recall"], True), "FPR %": fmt(mt["fpr"], True),
            "Precision %": fmt(mt["precision"], True), "F1 %": fmt(mt["f1"], True),
            "Brier": fmt(mt["brier"], False, 5),
            "ΔPR-AUC vs B (pp)": f"{dB['pr_auc']['mean']:+.2f} ± {dB['pr_auc']['std']:.2f}",
            "ΔPR-AUC vs A (pp)": f"{dA['pr_auc']['mean']:+.2f} ± {dA['pr_auc']['std']:.2f}",
            "Seeds > B": f"{dB['pr_auc']['seeds_positive']}/{dB['pr_auc']['n_seeds']}",
        })
        rc = {"Config": c}
        for s in SCENARIOS:
            rc[f"{s} %"] = fmt(mt[f"recall_{s}"], True)
            rc[f"Δ{s} vs B (pp)"] = f"{dB[f'recall_{s}']['mean']:+.2f}"
        rows_c.append(rc)
    tb, tc = pd.DataFrame(rows_b), pd.DataFrame(rows_c)
    tb.to_csv(TAB / "table_B_ablation.csv", index=False)
    (TAB / "table_B_ablation.md").write_text(tb.to_markdown(index=False), encoding="utf-8")
    tc.to_csv(TAB / "table_C_ablation_scenarios.csv", index=False)
    (TAB / "table_C_ablation_scenarios.md").write_text(tc.to_markdown(index=False), encoding="utf-8")

    # ── Figures ─────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})

    # Fig 1 - paired seed-level PR-AUC
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for s in seeds:
        ax.plot([0, 1], [v1.loc[s, "pr_auc"], v2.loc[s, "pr_auc"]], color="0.6", lw=0.7, zorder=1)
    ax.scatter(np.zeros(len(seeds)), v1.pr_auc, s=14, color="#4C72B0", zorder=2, label="V1 (12 features)")
    ax.scatter(np.ones(len(seeds)), v2.pr_auc, s=14, color="#C44E52", zorder=2, label="V2 (47 features)")
    for x, m in ((0, v1.pr_auc), (1, v2.pr_auc)):
        ax.hlines(m.mean(), x - 0.18, x + 0.18, color="k", lw=1.2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["V1", "V2"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Test PR-AUC (per seed)"); ax.grid(axis="y", lw=0.3, alpha=0.6)
    ax.set_title(f"{len(seeds)} seeds ({seeds[0]}–{seeds[-1]}); bar = mean", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig1_multiseed_prauc.png", dpi=300); plt.close(fig)

    # Fig 2 - operating-point metrics across seeds
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1))
    for ax, (m, lab) in zip(axes, [("recall", "Recall"), ("fpr", "FPR"), ("precision", "Precision"),
                                   ("recall_social_engineering", "SE recall")]):
        ax.boxplot([100 * v1[m], 100 * v2[m]], widths=0.5, showmeans=True,
                   meanprops=dict(marker="^", markerfacecolor="k", markeredgecolor="k", markersize=3),
                   medianprops=dict(color="k", lw=0.8))
        for x, series, col in ((1, v1[m], "#4C72B0"), (2, v2[m], "#C44E52")):
            ax.scatter(np.full(len(series), x) + np.linspace(-0.12, 0.12, len(series)), 100 * series,
                       s=6, color=col, zorder=3)
        if m == "fpr":
            ax.axhline(1.0, color="k", ls="--", lw=0.6)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["V1", "V2"]); ax.set_title(lab + " (%)", fontsize=8)
        ax.grid(axis="y", lw=0.3, alpha=0.6)
    fig.tight_layout(); fig.savefig(FIG / "fig2_multiseed_operating.png", dpi=300); plt.close(fig)

    # Fig 3 - ablation PR-AUC mean ± sd
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    means = [per_cfg[c]["metrics"]["pr_auc"]["mean"] for c in configs]
    sds = [per_cfg[c]["metrics"]["pr_auc"]["std"] for c in configs]
    y = np.arange(len(configs))[::-1]
    cols = ["#4C72B0" if c.startswith("A_") else "#C44E52" if c.startswith("K_") else "0.55" for c in configs]
    ax.barh(y, means, xerr=sds, color=cols, height=0.65, error_kw=dict(lw=0.7, capsize=2))
    ax.set_yticks(y); ax.set_yticklabels([c.replace("_", " ", 1) for c in configs], fontsize=6.5)
    ax.set_xlabel("Test PR-AUC, mean ± sd over seeds"); ax.grid(axis="x", lw=0.3, alpha=0.6)
    ax.axvline(per_cfg[refs["vs_B_core"]]["metrics"]["pr_auc"]["mean"], color="k", ls=":", lw=0.7)
    fig.tight_layout(); fig.savefig(FIG / "fig3_ablation_prauc.png", dpi=300); plt.close(fig)

    # Fig 4 - scenario recall heatmap
    mat = np.array([[100 * per_cfg[c]["metrics"][f"recall_{s}"]["mean"] for s in SCENARIOS] for c in configs])
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    im = ax.imshow(mat, cmap="Greys", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels(["Social\neng.", "Mule\ncoll.", "Account\ntakeover", "Card\ntesting"], fontsize=6.5)
    ax.set_yticks(range(len(configs))); ax.set_yticklabels([c.replace("_", " ", 1) for c in configs], fontsize=6.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="w" if mat[i, j] > 55 else "k")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Recall % (mean over seeds)", fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG / "fig4_ablation_scenarios.png", dpi=300); plt.close(fig)

    print(ta.to_string(index=False)); print(); print(tb.to_string(index=False)); print(); print(tc.to_string(index=False))
    print(json.dumps(summ["fpr_budget_check"], indent=1)); print(json.dumps(summ["seed42_reproduction"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
