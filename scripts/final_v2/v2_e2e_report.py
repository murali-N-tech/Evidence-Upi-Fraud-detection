"""Tables and figures for the ten-family end-to-end evaluation (V2 in the Classifier family),
evidence coverage, and the chronological replay results.

Inputs (all produced earlier, none recomputed here):
  results/final_v2/end_to_end/e2e_v2_summary.json, e2e_v2_rows_seed{42,43,44}.csv.gz,
  family_behaviour_last25.csv                             (v2_end_to_end.py)
  results/graph_replay/replay_validation_seed*.json,
  replay_run_seed*.json                                    (exp3_4_replay.py / exp4_rebuild_validation.py)
  results/final_v2/graph/graph_statistics.json,
  results/final_v2/replay/replay_examples_seed42.json     (v2_graph_and_replay_example.py)
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
G = ROOT / "results" / "graph_replay"
SEEDS = (42, 43, 44)
VERD = ["APPROVE", "WARN", "STEP_UP", "BLOCK"]
SC = ["social_engineering", "mule_collection", "account_takeover", "card_testing"]
SCL = {"social_engineering": "Social engineering", "mule_collection": "Mule collection",
       "account_takeover": "Account takeover", "card_testing": "Card testing"}
FAM_LABEL = {"payee_history": "Payee history", "relationship": "Relationship", "payer_behaviour": "Payer behaviour",
             "ml_classifier": "Classifier (V2)", "address_and_qr": "Address and QR", "graph": "Graph",
             "amount_context": "Amount context", "message_pressure": "Message pressure", "link_safety": "Link safety",
             "stated_intent": "Stated intent"}


def ms(vals, k=1.0, dec=3):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float) * k
    if len(v) == 0:
        return "n/a"
    return f"{v.mean():.{dec}f} ± {v.std(ddof=1):.{dec}f}" if len(v) > 1 else f"{v[0]:.{dec}f}"


def md(df, path, title, note=""):
    path.write_text(f"**{title}**\n\n" + df.to_markdown(index=False) + (f"\n\n{note}\n" if note else "\n"), encoding="utf-8")
    df.to_csv(path.with_suffix(".csv"), index=False)


def main() -> int:
    S = json.load(open(R / "end_to_end/e2e_v2_summary.json"))
    P = {int(k): v for k, v in S["per_seed"].items()}
    rows = []
    for sl, lab in (("last25pct", "Last 25% of each stream (primary)"), ("full_stream", "Full stream")):
        get = lambda f: [f(P[s][sl]) for s in SEEDS]
        rows += [
            {"Slice": lab, "Quantity": "Payments / fraud per stream", "Value (mean ± sd over 3 streams)":
                f"{P[42][sl]['n']:,} / {ms(get(lambda d: d['fraud']), dec=0)}"},
            {"Slice": lab, "Quantity": "Fraud prevalence (%)", "Value (mean ± sd over 3 streams)": ms(get(lambda d: d["prevalence"]), 100, 2)},
            {"Slice": lab, "Quantity": "Ten-family risk R — PR-AUC", "Value (mean ± sd over 3 streams)": ms(get(lambda d: d["risk_pr_auc"]))},
            {"Slice": lab, "Quantity": "Ten-family risk R — ROC-AUC", "Value (mean ± sd over 3 streams)": ms(get(lambda d: d["risk_roc_auc"]))},
            {"Slice": lab, "Quantity": "Frozen V2 probability alone — PR-AUC (rows where the family is available)",
             "Value (mean ± sd over 3 streams)": ms(get(lambda d: d["v2_alone_pr_auc_where_available"]))},
            {"Slice": lab, "Quantity": "Frozen V2 probability alone — ROC-AUC (same rows)",
             "Value (mean ± sd over 3 streams)": ms(get(lambda d: d["v2_alone_roc_auc_where_available"]))},
            {"Slice": lab, "Quantity": "Frozen V2 alone at θ = 0.1123 — recall / FPR (%)",
             "Value (mean ± sd over 3 streams)": f"{ms(get(lambda d: d['v2_alone_at_threshold']['recall']), 100, 1)} / "
                                                 f"{ms(get(lambda d: d['v2_alone_at_threshold']['fpr']), 100, 2)}"},
            {"Slice": lab, "Quantity": "Verdict WARN or above — recall / FPR (%)",
             "Value (mean ± sd over 3 streams)": f"{ms(get(lambda d: d['warn_plus']['recall']), 100, 1)} / {ms(get(lambda d: d['warn_plus']['fpr']), 100, 1)}"},
            {"Slice": lab, "Quantity": "Verdict STEP_UP or above — recall / FPR (%)",
             "Value (mean ± sd over 3 streams)": f"{ms(get(lambda d: d['step_up_plus']['recall']), 100, 1)} / {ms(get(lambda d: d['step_up_plus']['fpr']), 100, 1)}"},
            {"Slice": lab, "Quantity": "Verdict BLOCK — recall / FPR (%)",
             "Value (mean ± sd over 3 streams)": f"{ms(get(lambda d: d['block']['recall']), 100, 1)} / {ms(get(lambda d: d['block']['fpr']), 100, 2)}"},
            {"Slice": lab, "Quantity": "Intervention rate, all payments (non-APPROVE, %)",
             "Value (mean ± sd over 3 streams)": ms(get(lambda d: d["intervention_rate"]), 100, 1)},
        ]
    t9 = pd.DataFrame(rows)
    md(t9, R / "tables/table09_end_to_end.md", "Table 9 — Ten-family end-to-end evaluation (chronological replay, V2 as the Classifier family)",
       "Verdicts are advisory states of the risk score, not ground-truth classes; recall/FPR at a verdict level are operating points of R. "
       "Three replay streams (seeds 42–44), 60,000 payments each; the last 25% (days ≈112–150) follows a warm-up of the shared store.")

    # verdict counts and rates table (per group, pooled counts shown per seed then mean rate)
    vr = []
    for grp, lab in [("all", "All payments"), ("legitimate", "Legitimate"), ("fraud", "Fraud (all scenarios)")] + [(s, SCL[s]) for s in SC]:
        row = {"Group (last 25%)": lab}
        tot = [sum(P[s]["last25pct"]["verdict_counts"][grp].values()) for s in SEEDS]
        row["n per stream"] = " / ".join(str(t) for t in tot)
        for v in VERD:
            rates = [P[s]["last25pct"]["verdict_counts"][grp][v] / max(t, 1) for s, t in zip(SEEDS, tot)]
            cnts = [P[s]["last25pct"]["verdict_counts"][grp][v] for s in SEEDS]
            row[f"{v} % (counts)"] = f"{ms(rates, 100, 1)} ({'/'.join(map(str, cnts))})"
        vr.append(row)
    t9b = pd.DataFrame(vr)
    md(t9b, R / "tables/table09b_verdict_distribution.md", "Table 9b — Verdict distribution by group (last 25% of each stream)",
       "Rates are mean ± sd over the three streams; counts are per stream (seed 42 / 43 / 44). Scenario groups contain fraudulent payments only.")

    # evidence level + coverage
    cov_rows = []
    for s in SEEDS:
        c = P[s]["last25pct"]["coverage"]
        cov_rows.append({"Stream": s, "Coverage mean": round(c["mean"], 2), "Median": c["median"], "Min": c["min"], "Max": c["max"],
                         "Legitimate mean": round(c["by_label"]["legitimate"]["mean"], 2), "Fraud mean": round(c["by_label"]["fraud"]["mean"], 2),
                         **{f"Level {k} (n)": v for k, v in P[s]["last25pct"]["evidence_level_counts"].items()},
                         **{f"Payee status {k} mean": round(v["mean"], 1) for k, v in c["by_payee_status"].items()}})
    t_cov = pd.DataFrame(cov_rows)
    md(t_cov, R / "tables/table09c_coverage.md", "Table 9c — Evidence coverage C (last 25% of each stream)",
       "Coverage (0–100) depends only on which families were available (fixed weights, evidence.CONFIDENCE_WEIGHT); it is not a risk score.")
    full_cov = []
    for s in SEEDS:
        c = P[s]["full_stream"]["coverage"]
        full_cov.append({"Stream": s, "Coverage mean": round(c["mean"], 2), "Median": c["median"], "Min": c["min"], "Max": c["max"],
                         **{f"Level {k} (n)": v for k, v in P[s]["full_stream"]["evidence_level_counts"].items()},
                         **{f"{k}: mean / median / min / max": f"{v['mean']:.1f} / {v['median']:.0f} / {v['min']} / {v['max']}"
                            for k, v in c["by_level"].items()}})
    md(pd.DataFrame(full_cov), R / "tables/table09d_coverage_full_stream.md", "Table 9d — Evidence coverage C by evidence level (full streams)")
    mc = S["missing_evidence_checks"]
    mrows = []
    for s in SEEDS:
        for fam, v in mc[str(s)].items():
            mrows.append({"Stream": s, "Family": FAM_LABEL[fam], "Unavailable payments": v["unavailable_rows"],
                          "Median R when unavailable": v["risk_median_when_unavailable"],
                          "Median R when available": v["risk_median_when_available"],
                          "Share R = 0 when unavailable": round(v["share_R0_when_unavailable"], 4),
                          "Share BLOCK when unavailable": round(v["share_BLOCK_when_unavailable"], 4),
                          "Fraud rate when unavailable": round(v["fraud_rate_when_unavailable"], 4)})
    md(pd.DataFrame(mrows), R / "tables/table09e_missing_evidence.md", "Table 9e — Missing evidence is neither 'safe' nor 'fraud' (full streams)",
       "Families that were available on every payment (payee history, relationship, address and QR, amount context, link safety) are not listed.")

    fam = pd.read_csv(R / "end_to_end/family_behaviour_last25.csv")
    frow = []
    for f in FAM_LABEL:
        g = fam[fam.family == f]
        frow.append({"Family": FAM_LABEL[f], "Available %": ms(g.available_rate, 100, 1), "Active on fraud %": ms(g.active_rate_fraud, 100, 1),
                     "Active on legitimate %": ms(g.active_rate_legit, 100, 1), "AUC of family score (where available)": ms(g.auc_within_available, 1, 2),
                     "Leads fraud assessments %": ms(g.leading_rate_fraud, 100, 1)})
    md(pd.DataFrame(frow), R / "tables/table09f_family_behaviour.md", "Table 9f — Evidence-family behaviour in the complete system (last 25%, mean ± sd over 3 streams)",
       "Active = family available and scored above zero. Leads = the family has the highest score in a fraudulent assessment.")

    # ── figures ───────────────────────────────────────────────────────────
    rows_all = {s: pd.read_csv(R / f"end_to_end/e2e_v2_rows_seed{s}.csv.gz") for s in SEEDS}
    # fig12 verdict distribution
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    groups = ["legitimate"] + SC
    labels = ["Legitimate"] + [SCL[s] for s in SC]
    left = np.zeros(len(groups))
    cols = ["#cfe0f5", "#eda100", "#eb6834", "#b3261e"]
    for v, c in zip(VERD, cols):
        vals = np.array([np.mean([P[s]["last25pct"]["verdict_counts"][g][v] / max(sum(P[s]["last25pct"]["verdict_counts"][g].values()), 1)
                                  for s in SEEDS]) for g in groups]) * 100
        ax.barh(range(len(groups)), vals, left=left, color=c, label=v.replace("_", "-"), height=0.62)
        for i, (l, w) in enumerate(zip(left, vals)):
            if w >= 6:
                ax.text(l + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=7.5, color="#0b0b0b" if v in ("APPROVE", "WARN") else "white")
        left += vals
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("Share of payments (%), mean over three replay streams")
    ax.set_title("Advisory verdict distribution — ten-family system, last 25% of each stream")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.36), fontsize=8); ax.grid(axis="y", visible=False)
    style.save(fig, R / "figures/fig12_verdict_distribution")

    # fig13 coverage distribution + risk vs coverage
    d42 = rows_all[42]; n = len(d42); d = d42.iloc[int(n * 0.75):]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    bins = np.arange(60, 102, 2)
    ax.hist(d.coverage[d.is_fraud == 0], bins=bins, color=style.LEGIT, alpha=0.6, density=True, label="Legitimate")
    ax.hist(d.coverage[d.is_fraud == 1], bins=bins, color=style.FRAUD, histtype="step", lw=2, density=True, label="Fraud")
    ax.set_xlabel("Evidence coverage C"); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    ax.set_title("Coverage distribution (seed 42, last 25%)")
    ax = axes[1]
    hb = ax.hexbin(d.coverage, d.risk, gridsize=30, bins="log", cmap="Blues", mincnt=1)
    ax.scatter(d.coverage[d.is_fraud == 1], d.risk[d.is_fraud == 1], s=6, color=style.FRAUD, label="Fraud")
    for t in (22, 45, 70):
        ax.axhline(t, color=style.MUTED, ls="--", lw=0.8)
    ax.set_xlabel("Evidence coverage C"); ax.set_ylabel("Risk score R"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Risk versus coverage (hexbin: all payments)")
    fig.colorbar(hb, ax=ax, fraction=0.04).set_label("payments (log)", fontsize=7)
    style.save(fig, R / "figures/fig13_coverage")

    # fig: family availability & activation
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    order = list(FAM_LABEL)
    y = np.arange(len(order))[::-1]
    av = [fam[fam.family == f].available_rate.mean() * 100 for f in order]
    af = [fam[fam.family == f].active_rate_fraud.mean() * 100 for f in order]
    al = [fam[fam.family == f].active_rate_legit.mean() * 100 for f in order]
    ax.barh(y + 0.27, av, height=0.26, color="#9bbfe9", label="Available")
    ax.barh(y, af, height=0.26, color=style.FRAUD, label="Active on fraud")
    ax.barh(y - 0.27, al, height=0.26, color=style.LEGIT, label="Active on legitimate")
    ax.set_yticks(y); ax.set_yticklabels([FAM_LABEL[f] for f in order]); ax.set_xlabel("Share of payments (%)")
    ax.set_title("Evidence-family availability and activation (last 25%, mean over 3 streams)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", visible=False)
    style.save(fig, R / "figures/fig13b_family_availability")

    # fig15 replay: store maturity + validation
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for s in SEEDS:
        b = P[s]["blocks"]; x = [int(z["days"].split("-")[0]) + 15 for z in b]
        lw, al_ = (2.0, 1) if s == 42 else (1.0, 0.5)
        ax.plot(x, [100 * z["legit_block_rate"] for z in b], "s--", color=style.LEGIT, lw=lw, alpha=al_, ms=3)
        ax.plot(x, [100 * z["legit_warn_plus_rate"] for z in b], "o-", color=style.LEGIT, lw=lw, alpha=al_, ms=3)
        ax.plot(x, [100 * z["fraud_block_recall"] for z in b], "s--", color=style.FRAUD, lw=lw, alpha=al_, ms=3)
        ax.plot(x, [100 * z["fraud_warn_plus_recall"] for z in b], "o-", color=style.FRAUD, lw=lw, alpha=al_, ms=3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=style.FRAUD, ls="-", label="Fraud flagged (WARN+)"),
                       Line2D([], [], color=style.FRAUD, ls="--", label="Fraud blocked"),
                       Line2D([], [], color=style.LEGIT, ls="-", label="Legitimate flagged (WARN+)"),
                       Line2D([], [], color=style.LEGIT, ls="--", label="Legitimate blocked")], fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_xlabel("Day of stream (30-day blocks)"); ax.set_ylabel("Share (%)"); ax.set_ylim(0, 100)
    ax.set_title("Chronological replay: verdicts as the store matures\n(seed 42 bold; seeds 43–44 thin)")
    ax = axes[1]
    val = {s: json.load(open(G / f"replay_validation_seed{s}.json"))["tests"] for s in SEEDS}
    names = sorted(val[42], key=lambda k: int(k.split("_")[0].rstrip("ab")) + (0.5 if k.startswith("9") else 0))
    checked = [sum(val[s][k]["checked"] for s in SEEDS) for k in names]
    viol = [sum(val[s][k]["violations"] for s in SEEDS) for k in names]
    yy = np.arange(len(names))[::-1]
    ax.barh(yy, checked, color=style.BLUE, height=0.6)
    for yv, c, v in zip(yy, checked, viol):
        ax.text(c * 1.25, yv, f"{c:,} · {v} violations", va="center", fontsize=7, color=style.INK2)
    short = {"1": "first payment to a payee is 'new'", "2": "prior pair count = earlier rows", "3": "distinct payers from earlier rows only",
             "4": "arrival window from earlier rows only", "5": "graph component is pre-payment", "6": "position finding only when gates hold",
             "7": "unobserved payee: graph unavailable, not 0", "8": "independent rebuild from earlier rows identical",
             "9a": "earlier observations never rewritten", "10": "store holds only earlier payments"}
    ax.set_xscale("log"); ax.set_yticks(yy); ax.set_yticklabels([f"{k.split('_')[0]}. {short.get(k.split('_')[0], k)}" for k in names], fontsize=7)
    ax.set_xlim(10, 5e6); ax.set_xlabel("Checks performed (log scale), three streams combined")
    ax.set_title("Replay validation tests (all passed)"); ax.grid(axis="y", visible=False)
    style.save(fig, R / "figures/fig15_replay")

    # table 11: replay validation
    t11 = pd.DataFrame([{"Test": k, **{f"Seed {s} checked": val[s][k]["checked"] for s in SEEDS},
                         "Violations (all seeds)": sum(val[s][k]["violations"] for s in SEEDS),
                         "Result": "PASS" if all(val[s][k]["pass"] for s in SEEDS) else "FAIL"} for k in names])
    md(t11, R / "tables/table11_replay_validation.md", "Table 11 — Chronological graph-replay validation (three streams)",
       "Tests 1–7 and 10 are checked online at every assessment against an independent shadow model of earlier payments; "
       "8 by an independent rebuild of the store from earlier payments only; 9a by comparing observation rows written before the "
       "stream midpoint with the final store.")
    print(t9.to_string(index=False)); print(t9b.to_string(index=False)); print(t11.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
