"""Paper tables and result figures, generated ONLY from experiment artefacts.

Every number printed in Tables III-VI and Figs. 4-5 comes from files under
/home/claude/exp/results; nothing is typed by hand.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/home/claude/exp/results")
OUT = Path(__file__).parent / "tabs"; OUT.mkdir(exist_ok=True)
SCEN = ["social_engineering", "mule_collection", "account_takeover", "card_testing"]


def j(p):
    return json.loads((R / p).read_text())


def pm(d, pct=False, dec=3):
    k = 100 if pct else 1
    return f"{k*d['mean']:.{dec}f} ± {k*d['std']:.{dec}f}"


def rng_(d, pct=False, dec=3):
    k = 100 if pct else 1
    return f"{k*d['min']:.{dec}f}–{k*d['max']:.{dec}f}"


# ---------------- Table III: ten-seed stability ----------------
ms = j("multiseed/multiseed_summary.json")["summary"]
rows = [("PR-AUC", "pr_auc", False, 3), ("ROC-AUC", "roc_auc", False, 3), ("Recall (%)", "recall", True, 1),
        ("FPR (%)", "fpr", True, 2), ("Precision (%)", "precision", True, 1), ("F1 (%)", "f1", True, 1),
        ("Brier (×10⁻³)", "brier", "milli", 2), ("BSS", "bss", False, 3),
        ("SE recall (%)", "recall_social_engineering", True, 1), ("MC recall (%)", "recall_mule_collection", True, 1),
        ("AT recall (%)", "recall_account_takeover", True, 1), ("CT recall (%)", "recall_card_testing", True, 1)]
lines = ["| Metric | V1 mean ± sd | V1 min–max | V2 mean ± sd | V2 min–max | Δ (V2 − V1) | Seeds V2 better |",
         "|---|---|---|---|---|---|---|"]
for lab, m, pct, dec in rows:
    a, b, d = ms["V1"][m], ms["V2"][m], ms["paired_V2_minus_V1"][m]
    if pct == "milli":
        f = lambda x: {k: 1000 * v if k in ("mean", "std", "min", "max") else v for k, v in x.items()}
        a, b, d = f(a), f(b), f(d); pct = False
    unit = " pp" if pct else ""
    lines.append(f"| {lab} | {pm(a, pct, dec)} | {rng_(a, pct, dec)} | {pm(b, pct, dec)} | {rng_(b, pct, dec)} | "
                 f"{(100 if pct else 1)*d['mean']:+.{dec}f} ± {(100 if pct else 1)*d['std']:.{dec}f}{unit} | "
                 f"{d['seeds_V2_better']}/10 |")
(OUT / "tab3.md").write_text("\n".join(lines) + "\n")

# ---------------- Table IV: ablation (+ template-disjoint) ----------------
ab = j("ablation/ablation_summary.json")["configs"]
names = {"A_v1_baseline": ("A", "V1 baseline"), "B_core": ("B", "V2 core"), "C_core+relationship": ("C", "Core + relationship"),
         "D_core+graph": ("D", "Core + graph"), "E_core+message": ("E", "Core + message"), "F_core+link": ("F", "Core + link"),
         "G_core+intent": ("G", "Core + intent"), "H_core+rel+graph": ("H", "Core + rel. + graph"),
         "I_core+rel+graph+msg": ("I", "H + message"), "J_core+rel+graph+msg+link": ("J", "I + link"),
         "K_full_v2": ("K", "Full V2 (I + link + intent)")}
lines = ["| Configuration | *F* | PR-AUC | ΔPR-AUC vs B (pp) | ROC-AUC | Recall | FPR | Prec. | F1 | SE | MC | AT | CT |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for key, (tag, lab) in names.items():
    c = ab[key]; mt = c["metrics"]; dB = c["paired_deltas_pp"]["vs_B_core"]["pr_auc"]
    delta = "—" if tag == "B" else f"{dB['mean']:+.1f} ± {dB['std']:.1f}"
    lines.append(f"| {tag}. {lab} | {c['n_features']} | {pm(mt['pr_auc'])} | {delta} | {mt['roc_auc']['mean']:.3f} | "
                 f"{100*mt['recall']['mean']:.1f} | {100*mt['fpr']['mean']:.2f} | {100*mt['precision']['mean']:.1f} | "
                 f"{100*mt['f1']['mean']:.1f} | " + " | ".join(f"{100*mt['recall_'+s]['mean']:.1f}" for s in SCEN) + " |")
tdp = R / "ablation/template_disjoint_results.csv"
if tdp.exists():
    td = pd.read_csv(tdp)
    abl = pd.read_csv(R / "ablation/ablation_results.csv")
    core = abl[abl.config == "B_core"].set_index("seed").pr_auc
    for key, tag in (("E_core+message", "E†"), ("I_core+rel+graph+msg", "I†"), ("K_full_v2", "K†")):
        s = td[td.config == key].set_index("seed").sort_index()
        if len(s) < 10:
            continue
        d = 100 * (s.pr_auc - core.loc[s.index])
        lines.append(f"| {tag} {names[key][1].split(' (')[0]}, template-disjoint | {int(s.n_features.iloc[0])} | "
                     f"{s.pr_auc.mean():.3f} ± {s.pr_auc.std(ddof=1):.3f} | {d.mean():+.1f} ± {d.std(ddof=1):.1f} | "
                     f"{s.roc_auc.mean():.3f} | {100*s.recall.mean():.1f} | {100*s.fpr.mean():.2f} | "
                     f"{100*s.precision.mean():.1f} | {100*s.f1.mean():.1f} | "
                     + " | ".join(f"{100*s['recall_'+x].mean():.1f}" for x in SCEN) + " |")
(OUT / "tab4.md").write_text("\n".join(lines) + "\n")
print((OUT / "tab3.md").read_text()); print((OUT / "tab4.md").read_text())

# ---------------- Fig. 4: robustness (paired seeds + ablation gains) ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7, "axes.linewidth": 0.6})
abl = pd.read_csv(R / "ablation/ablation_results.csv")
pv = abl.pivot(index="seed", columns="config", values="pr_auc")
fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.35), gridspec_kw={"width_ratios": [0.8, 1.45]})
ax = axes[0]
for s in pv.index:
    ax.plot([0, 1], [pv.loc[s, "A_v1_baseline"], pv.loc[s, "K_full_v2"]], color="0.65", lw=0.6)
ax.scatter(np.zeros(len(pv)), pv["A_v1_baseline"], s=9, color="#4C72B0", zorder=3)
ax.scatter(np.ones(len(pv)), pv["K_full_v2"], s=9, color="#C44E52", zorder=3)
for x, c in ((0, "A_v1_baseline"), (1, "K_full_v2")):
    ax.hlines(pv[c].mean(), x - 0.2, x + 0.2, color="k", lw=1.1)
ax.set_xticks([0, 1]); ax.set_xticklabels(["V1", "V2"]); ax.set_xlim(-0.45, 1.45)
ax.set_ylabel("Test PR-AUC"); ax.set_title("(a) Ten seeds", fontsize=7)
ax.grid(axis="y", lw=0.3, alpha=0.5)
ax = axes[1]
core = pv["B_core"]
keys = [("C_core+relationship", "C +rel."), ("D_core+graph", "D +graph"), ("E_core+message", "E +msg."),
        ("F_core+link", "F +link"), ("G_core+intent", "G +intent"), ("H_core+rel+graph", "H +rel.+graph"),
        ("I_core+rel+graph+msg", "I  H+msg."), ("J_core+rel+graph+msg+link", "J  I+link"), ("K_full_v2", "K  full V2")]
labs, means, sds, cols = [], [], [], []
for k, lab in keys:
    d = 100 * (pv[k] - core); labs.append(lab); means.append(d.mean()); sds.append(d.std(ddof=1))
    cols.append("#C44E52" if k == "K_full_v2" else "0.55")
if tdp.exists():
    td = pd.read_csv(tdp)
    for k, lab in (("E_core+message", "E† disjoint"), ("I_core+rel+graph+msg", "I† disjoint"), ("K_full_v2", "K† disjoint")):
        s = td[td.config == k].set_index("seed").pr_auc
        if len(s) == 10:
            d = 100 * (s - core.loc[s.index]); labs.append(lab); means.append(d.mean()); sds.append(d.std(ddof=1))
            cols.append("white")
yy = np.arange(len(labs))[::-1]
ax.barh(yy, means, xerr=sds, color=cols, edgecolor="k", lw=0.4, height=0.65, error_kw=dict(lw=0.6, capsize=1.5))
ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=6)
ax.set_xlabel("ΔPR-AUC vs. core B (pp)"); ax.set_title("(b) Ablation gains", fontsize=7)
ax.grid(axis="x", lw=0.3, alpha=0.5); ax.axvline(0, color="k", lw=0.5)
fig.tight_layout(w_pad=0.6); fig.savefig(Path(__file__).parent / "fig/fig4_robustness.png", dpi=400); plt.close(fig)

# ---------------- Tables V and VI: replay stream (Experiments 3 and 4) ----------------
E = R / "end_to_end"; G = R / "graph_replay"
seeds = sorted(int(p.stem.split("seed")[1]) for p in E.glob("e2e_summary_seed*.json"))
seeds4 = sorted(int(p.stem.split("seed")[1]) for p in G.glob("replay_metrics_seed*.json"))
S3 = {s: json.loads((E / f"e2e_summary_seed{s}.json").read_text()) for s in seeds}
S4 = {s: json.loads((G / f"replay_metrics_seed{s}.json").read_text()) for s in seeds4}
FAM = pd.concat([pd.read_csv(E / f"evidence_family_metrics_seed{s}.csv").assign(seed=s) for s in seeds])


def ms_(vals, pct=False, dec=3):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    k = 100 if pct else 1
    if len(v) == 0:
        return "—"
    if len(v) == 1:
        return f"{k*v[0]:.{dec}f}"
    return f"{k*v.mean():.{dec}f} ± {k*v.std(ddof=1):.{dec}f}"


def sl(s, name):
    return next(x for x in S3[s]["risk_and_verdicts"] if x["slice"] == name)


# Table V: per-family behaviour in the complete system (last 25% of each stream)
fam_names = {"payee_history": "Payee history", "relationship": "Relationship", "payer_behaviour": "Payer behaviour",
             "ml_classifier": "Classifier (V1)", "address_and_qr": "Address and QR", "graph": "Graph",
             "amount_context": "Amount context", "message_pressure": "Message pressure", "link_safety": "Link safety",
             "stated_intent": "Stated intent"}
lines = ["| Family | Avail. | Active: fraud | Active: legit. | AUC | Leads (fraud) | LOO ↓ fraud | LOO ↓ legit. |",
         "|---|---|---|---|---|---|---|---|"]
F = FAM[FAM.slice == "test_last25pct"]
for key, lab in fam_names.items():
    g = F[F.family == key]
    lines.append(f"| {lab} | {ms_(g.available_rate, True, 1)} | {ms_(g.active_rate_fraud, True, 1)} | "
                 f"{ms_(g.active_rate_legit, True, 1)} | {ms_(g.auc_within_available, False, 2)} | "
                 f"{ms_(g.leading_rate_fraud, True, 1)} | {ms_(g.loo_verdict_down_fraud, True, 1)} | "
                 f"{ms_(g.loo_verdict_down_legit, True, 1)} |")
(OUT / "tab5.md").write_text("\n".join(lines) + "\n")

# Table VI: system vs classifier variants on the replay test period
lines = ["| Replay stream, last 25% | PR-AUC | ROC-AUC | Recall | FPR | Precision | SE | MC | AT | CT |",
         "|---|---|---|---|---|---|---|---|---|---|"]
t = lambda s: sl(s, "test_last25pct")
for op, lab in (("warn_plus", "System: WARN or above"), ("step_up_plus", "System: STEP_UP or above"),
                ("block", "System: BLOCK")):
    pr = ms_([t(s)["risk_pr_auc"] for s in seeds]) if op == "warn_plus" else "″"
    roc = ms_([t(s)["risk_roc_auc"] for s in seeds]) if op == "warn_plus" else "″"
    sc = [S3[s]["one_percent_budget_on_last25pct"][f"scenario_recall_{op}_last25"] for s in seeds]
    lines.append(f"| {lab} | {pr} | {roc} | {ms_([t(s)[op]['recall'] for s in seeds], True, 1)} | "
                 f"{ms_([t(s)[op]['fpr'] for s in seeds], True, 1)} | {ms_([t(s)[op]['precision'] for s in seeds], True, 1)} | "
                 + " | ".join(ms_([c['recall_' + x] for c in sc], True, 0) for x in SCEN) + " |")
lv = [S3[s]["one_percent_budget_on_last25pct"]["live_v1_at_stored_threshold_last25"] for s in seeds]
lines.append(f"| Live V1 alone, $\\theta$ = {lv[0]['threshold']} | {ms_([x['pr_auc'] for x in lv])} | "
             f"{ms_([x['roc_auc'] for x in lv])} | {ms_([x['recall'] for x in lv], True, 1)} | {ms_([x['fpr'] for x in lv], True, 2)} | "
             f"{ms_([x['precision'] for x in lv], True, 1)} | " + " | ".join(ms_([x['recall_' + c] for x in lv], True, 0) for c in SCEN) + " |")
vn = {"A_v1_replay_state": "A. V1, refitted on replay", "B_v2_simulator_graph_channel": "B. V2 + simulator graph*",
      "C_v2_replayed_graph": "C. V2 + replayed graph", "D_v2_without_graph": "D. V2 without graph"}
for v, lab in vn.items():
    rr = [S4[s]["results"][v] for s in seeds4]
    lines.append(f"| {lab} | {ms_([x['pr_auc'] for x in rr])} | {ms_([x['roc_auc'] for x in rr])} | "
                 f"{ms_([x['recall'] for x in rr], True, 1)} | {ms_([x['fpr'] for x in rr], True, 2)} | "
                 f"{ms_([x['precision'] for x in rr], True, 1)} | " + " | ".join(ms_([x['recall_' + c] for x in rr], True, 0) for c in SCEN) + " |")
(OUT / "tab6.md").write_text("\n".join(lines) + "\n")
print("seeds E2E", seeds, "seeds replay models", seeds4)
print((OUT / "tab5.md").read_text()); print((OUT / "tab6.md").read_text())

# ---------------- facts for the running text (Sections IV-D to IV-F) ----------------
from collections import Counter
V = {s: json.loads((G / f"replay_validation_seed{s}.json").read_text()) for s in seeds
     if (G / f"replay_validation_seed{s}.json").exists()}
RUN = {s: json.loads((G / f"replay_run_seed{s}.json").read_text()) for s in seeds}
TX = {s: pd.read_csv(G / f"replay_transactions_seed{s}.csv", usecols=["timestamp", "is_fraud", "scenario", "group_id"])
      for s in seeds}


def pmv(vals, pct=False, dec=1):
    return ms_(vals, pct, dec)


def vs(s, grp, v):
    return S3[s]["verdict_share_by_group_test_last25pct"][grp][v]


facts = {}
facts["n_streams"] = {1: "one", 2: "two", 3: "three", 4: "four"}.get(len(seeds), str(len(seeds)))
facts["stream_seeds"] = ", ".join(map(str, seeds))
facts["rows_total"] = f"{60000*len(seeds):,}"
facts["consistency_mismatches"] = str(sum(sum(S3[s]["consistency_audit"]["mismatches"].values()) for s in seeds))
facts["risk_pr"] = pmv([t(s)["risk_pr_auc"] for s in seeds], dec=3)
facts["risk_roc"] = pmv([t(s)["risk_roc_auc"] for s in seeds], dec=3)
facts["prev"] = f"{100*np.mean([t(s)['prevalence'] for s in seeds]):.2f}"
facts["lv_pr"] = pmv([S3[s]["one_percent_budget_on_last25pct"]["live_v1_at_stored_threshold_last25"]["pr_auc"] for s in seeds], dec=3)
facts["lv_roc"] = pmv([S3[s]["one_percent_budget_on_last25pct"]["live_v1_at_stored_threshold_last25"]["roc_auc"] for s in seeds], dec=3)
facts["lv_rec"] = pmv([S3[s]["one_percent_budget_on_last25pct"]["live_v1_at_stored_threshold_last25"]["recall"] for s in seeds], True)
for k, v in (("legit_approve", "APPROVE"), ("legit_warn", "WARN"), ("legit_stepup", "STEP_UP"), ("legit_block", "BLOCK")):
    facts[k] = pmv([vs(s, "legitimate", v) for s in seeds], True)
facts["se_block"] = pmv([vs(s, "social_engineering", "BLOCK") for s in seeds], True)
facts["mc_block"] = pmv([vs(s, "mule_collection", "BLOCK") for s in seeds], True)
facts["at_block"] = pmv([vs(s, "account_takeover", "BLOCK") for s in seeds], True)
facts["ct_approve"] = pmv([vs(s, "card_testing", "APPROVE") for s in seeds], True)
facts["warn_rec"] = pmv([t(s)["warn_plus"]["recall"] for s in seeds], True)
facts["warn_fpr"] = pmv([t(s)["warn_plus"]["fpr"] for s in seeds], True)
facts["block_rec"] = pmv([t(s)["block"]["recall"] for s in seeds], True)
facts["block_fpr"] = pmv([t(s)["block"]["fpr"] for s in seeds], True)
sat = [S3[s]["one_percent_budget_on_last25pct"]["risk_saturation"] for s in seeds]
facts["r99"] = pmv([x["legit_share_R99_selection_window"] for x in sat], True)
tot = [sum(x["leading_family_counts_legit_BLOCK_last25"].values()) for x in sat]
# a blocked payment can have tied leaders; shares are of leader mentions
for k, fam in (("lead_intent", "stated_intent"), ("lead_msg", "message_pressure"), ("lead_payee", "payee_history")):
    facts[k] = pmv([x["leading_family_counts_legit_BLOCK_last25"].get(fam, 0) / tt for x, tt in zip(sat, tot)], True, 0)
blocks = {s: [x for x in S3[s]["risk_and_verdicts"] if x["slice"].startswith("days_")] for s in seeds}
facts["block_legit_first"] = pmv([b[0]["block"]["fpr"] for b in blocks.values()], True)
facts["block_legit_last"] = pmv([b[-1]["block"]["fpr"] for b in blocks.values()], True)
allfr = [x["block"]["recall"] for b in blocks.values() for x in b]
facts["block_fraud_min"] = f"{100*min(allfr):.0f}"; facts["block_fraud_max"] = f"{100*max(allfr):.0f}"
Ft = FAM[FAM.slice == "test_last25pct"]
g_ = lambda fam, col: Ft[Ft.family == fam][col].tolist()
facts["pb_loo_fraud"] = pmv(g_("payer_behaviour", "loo_verdict_down_fraud"), True, 0)
facts["pb_loo_legit"] = pmv(g_("payer_behaviour", "loo_verdict_down_legit"), True, 0)
facts["ph_act_fraud"] = pmv(g_("payee_history", "active_rate_fraud"), True, 0)
facts["ph_act_legit"] = pmv(g_("payee_history", "active_rate_legit"), True, 0)
ev_cols = ["is_fraud", "graph_position_fired", "coverage", "risk", "facts"]
gp_f = gp_l = 0
for s in seeds:
    e_ = pd.read_csv(G / f"replay_evidence_seed{s}.csv", usecols=["is_fraud", "graph_position_fired"])
    gp_f += int(e_.graph_position_fired[e_.is_fraud == 1].sum()); gp_l += int(e_.graph_position_fired[e_.is_fraud == 0].sum())
facts["gp_fraud"] = str(gp_f); facts["gp_legit"] = str(gp_l)
cor = [S3[s]["corroboration"] for s in seeds]
facts["bonus_fraud"] = pmv([c["bonus_rate_fraud"] for c in cor], True, 0)
facts["bonus_legit"] = pmv([c["bonus_rate_legit"] for c in cor], True, 0)
facts["dedup_rows"] = str(sum(c["rows_where_family_count_rule_exceeds_fact_rule"] for c in cor))
rc = [S3[s]["risk_vs_coverage"] for s in seeds]
facts["cov_fraud"] = f"{np.median([r['coverage_median_fraud'] for r in rc]):.0f}"
facts["cov_legit"] = f"{np.median([r['coverage_median_legit'] for r in rc]):.0f}"
facts["cov_rho"] = pmv([r["spearman_risk_coverage_all"] for r in rc], False, 2)
oc = Counter(); occ = Counter()
for s in seeds:
    for k, v in RUN[s]["online_checks"].items():
        oc[k] += v["checked"]; occ[k] += v["violations"]
assert sum(occ.values()) == 0, occ
facts["reads_1"] = f"{oc['1_first_pair_is_new']:,}"; facts["reads_4"] = f"{oc['4_arrival_window_only_prior']:,}"
facts["reads_7"] = f"{oc['7_unobserved_payee_graph_unavailable_not_zero']:,}"
if V:
    facts["t8"] = f"{sum(v['tests']['8_independent_rebuild_prior_only_identical']['checked'] for v in V.values())}"
    assert all(v["tests"]["8_independent_rebuild_prior_only_identical"]["pass"] for v in V.values())
    facts["t9a"] = f"{sum(v['tests']['9a_earlier_observations_not_rewritten']['checked'] for v in V.values()):,}"
    assert all(v["tests"]["9a_earlier_observations_not_rewritten"]["pass"] for v in V.values())
    d9 = sum(v["test9b_power_probe"]["differs_when_future_visible"] for v in V.values())
    n9 = sum(v["test9b_power_probe"]["sampled"] for v in V.values())
    facts["t9b"] = "all" if d9 == n9 else f"{d9} of {n9}"
    facts["n_val_streams"] = str(len(V))
for v, k in (("A_v1_replay_state", "A_pr"), ("B_v2_simulator_graph_channel", "B_pr"), ("C_v2_replayed_graph", "C_pr"),
             ("D_v2_without_graph", "D_pr")):
    facts[k] = pmv([S4[s]["results"][v]["pr_auc"] for s in seeds4], False, 3)
cd = [100 * (S4[s]["results"]["C_v2_replayed_graph"]["pr_auc"] - S4[s]["results"]["D_v2_without_graph"]["pr_auc"]) for s in seeds4]
bc = [100 * (S4[s]["results"]["B_v2_simulator_graph_channel"]["pr_auc"] - S4[s]["results"]["C_v2_replayed_graph"]["pr_auc"]) for s in seeds4]
fmt_signed = lambda xs: (f"{np.mean(xs):+.1f} ± {np.std(xs, ddof=1):.1f}" if len(xs) > 1 else f"{xs[0]:+.1f}")
facts["CD"] = fmt_signed(cd); facts["BC"] = fmt_signed(bc).lstrip("+")
facts["comp_legit"] = f"{np.median([S4[s]['graph_signal']['graph_feature_medians_by_scenario']['component_size']['legitimate'] for s in seeds4]):,.0f}"
facts["mc_overlap"] = pmv([S4[s]["graph_signal"]["graph_feature_medians_by_scenario"]["overlap_ratio"]["mule_collection"] for s in seeds4], False, 2)
rings, zero = [], 0
for s in seeds:
    tx = TX[s]; n_ = len(tx)
    rings.append(tx[(tx.is_fraud == 1) & (tx.scenario == "mule_collection")].group_id.nunique())
    tr_ = tx.iloc[:int(n_ * 0.6)]
    zero += int(((tr_.is_fraud == 1) & (tr_.scenario == "mule_collection")).sum() == 0)
facts["rings"] = "–".join(sorted({str(min(rings)), str(max(rings))}, key=int)) if min(rings) != max(rings) else str(rings[0])
facts["mc_train_zero"] = {0: "none", 1: "one", 2: "two", 3: "three"}.get(zero, str(zero))
tv = [S4[s]["graph_signal"]["train_vs_test_median_legit"]["arrival_span_days"] for s in seeds4]
facts["arr_train"] = f"{min(x['train'] for x in tv):.0f}–{max(x['train'] for x in tv):.0f}"
facts["arr_test"] = f"{min(x['test'] for x in tv):.0f}–{max(x['test'] for x in tv):.0f}"
facts["CD_min"] = f"{min(cd):+.1f}".replace("-", "−"); facts["CD_max"] = f"{max(cd):+.1f}".replace("-", "−")
facts["fraud_total"] = f"{sum(S3[s]['fraud'] for s in seeds):,}"
facts["_detail"] = {"cd_per_seed": cd, "bc_per_seed": bc, "rings_per_seed": rings, "seeds": seeds, "seeds4": seeds4}
(OUT / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
print(json.dumps(facts, indent=1, default=float))

# ---------------- Fig. 5: end-to-end evaluation ----------------
VERD = ["APPROVE", "WARN", "STEP_UP", "BLOCK"]
e42 = pd.read_csv(G / "replay_evidence_seed42.csv", usecols=["timestamp", "is_fraud", "scenario", "risk", "verdict"])
e42 = e42.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
T = e42.iloc[int(len(e42) * 0.75):]; yT = T.is_fraud.to_numpy()
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), gridspec_kw={"width_ratios": [1, 1.1, 1]})
ax = axes[0]; bins = np.arange(0, 101, 4)
ax.hist(T.risk[yT == 0], bins=bins, density=True, histtype="stepfilled", alpha=0.35, color="#4C72B0", ec="#4C72B0", lw=0.7, label="Legitimate")
ax.hist(T.risk[yT == 1], bins=bins, density=True, histtype="step", color="#C44E52", lw=1.0, label="Fraud")
for x in (22, 45, 70):
    ax.axvline(x, color="k", ls="--", lw=0.5)
ax.set_yscale("log"); ax.set_xlabel("Risk score R"); ax.set_ylabel("Density (log)")
ax.legend(frameon=False, fontsize=6, loc="upper center"); ax.set_title("(a) Risk by label, last 25%", fontsize=7, pad=15)
ax = axes[1]
grp = np.where(T.is_fraud == 1, T.scenario, "legitimate")
mix = pd.crosstab(grp, T.verdict, normalize="index").reindex(columns=VERD, fill_value=0)
order = ["legitimate", "social_engineering", "mule_collection", "account_takeover", "card_testing"]
order = [o for o in order if o in mix.index]
left = np.zeros(len(order))
for v, col in zip(VERD, ["0.92", "0.7", "0.45", "0.12"]):
    vals = 100 * mix.loc[order, v].to_numpy()
    ax.barh(range(len(order)), vals, left=left, color=col, label=v.replace("_", "-"), height=0.62, edgecolor="k", lw=0.3)
    left += vals
nm = {"legitimate": "Legitimate", "social_engineering": "Social eng.", "mule_collection": "Mule coll.",
      "account_takeover": "Acct. takeover", "card_testing": "Card testing"}
ax.set_yticks(range(len(order))); ax.set_yticklabels([nm[o] for o in order], fontsize=6.3)
ax.invert_yaxis(); ax.set_xlabel("Share of payments (%)"); ax.set_xlim(0, 100)
ax.legend(ncol=4, fontsize=5.3, frameon=False, loc="lower center", bbox_to_anchor=(0.42, 0.99), handlelength=0.9,
          columnspacing=0.5, handletextpad=0.3, borderaxespad=0.1)
ax.set_title("(b) Verdict mix, last 25%", fontsize=7, pad=15)
ax = axes[2]
for s in seeds:
    b = blocks[s]; xs = [int(x["slice"].split("_")[1].split("-")[0]) + 15 for x in b]
    lw = 1.1 if s == 42 else 0.6; al = 1 if s == 42 else 0.55
    ax.plot(xs, [100 * x["warn_plus"]["fpr"] for x in b], "-", color="#4C72B0", lw=lw, alpha=al, marker="o", ms=2)
    ax.plot(xs, [100 * x["block"]["fpr"] for x in b], "--", color="#4C72B0", lw=lw, alpha=al, marker="s", ms=2)
    ax.plot(xs, [100 * x["warn_plus"]["recall"] for x in b], "-", color="#C44E52", lw=lw, alpha=al, marker="o", ms=2)
    ax.plot(xs, [100 * x["block"]["recall"] for x in b], "--", color="#C44E52", lw=lw, alpha=al, marker="s", ms=2)
from matplotlib.lines import Line2D
h = [Line2D([], [], color="#C44E52", ls="-", label="Fraud, WARN+"), Line2D([], [], color="#C44E52", ls="--", label="Fraud, BLOCK"),
     Line2D([], [], color="#4C72B0", ls="-", label="Legit., WARN+"), Line2D([], [], color="#4C72B0", ls="--", label="Legit., BLOCK")]
ax.legend(handles=h, fontsize=5.2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2,
          handlelength=1.6, columnspacing=0.6, borderaxespad=0.1)
ax.set_xlabel("Day of stream (30-day blocks)"); ax.set_ylabel("Flagged (%)"); ax.set_ylim(0, 100); ax.set_xticks([15, 45, 75, 105, 135])
ax.grid(lw=0.3, alpha=0.5); ax.set_title("(c) Store maturity", fontsize=7, pad=22)
fig.tight_layout(w_pad=0.7); fig.savefig(Path(__file__).parent / "fig/fig5_e2e.png", dpi=400); plt.close(fig)
