"""Experiment 3 - end-to-end analysis of the ten-family assessment.

Input: results/graph_replay/replay_evidence_seed<S>.csv, written by
exp3_4_replay.py, which called payee_check.check_payee (the function behind
/payee/check and /payee/confirm) for every replayed payment from the
pre-payment state only.

This script does NOT re-score anything with a different rule. It
  1. rebuilds each row's Evidence objects from the recorded per-family
     availability / score / facts and re-runs evidence.combine_scores,
     evidence.agreement_bonus, evidence.confidence_score, evidence.evidence_level
     and core.verdict.verdict_from_score (plus the payee_check HARD_BLOCK rule)
     to confirm the recorded outputs are reproduced exactly (consistency audit);
  2. measures continuous risk discrimination (risk score vs synthetic label)
     separately from the advisory verdict distribution;
  3. per family: availability, activation, P(fraud | active), within-available
     AUC of the family score, leading-family rate, and a leave-one-family-out
     counterfactual (the same aggregation with that family removed);
  4. corroboration: bonus frequency, fact sharing between families and how
     much a family-count rule would have added (double-counting audit).

The four verdicts are advisory bands of one score, not ground-truth classes;
"detection at WARN+" etc. are therefore reported as operating points of the
risk score, and the verdict mix is reported as an intervention profile.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import RESULTS, SCENARIOS, environment, sha256_file, source_state, write_json  # noqa: E402

from backend.app.core.verdict import BLOCK_AT, STEP_UP_AT, WARN_AT, verdict_from_score   # noqa: E402
from backend.app.services.evidence import (AGREEMENT_BONUS, AGREEMENT_THRESHOLD,          # noqa: E402
                                           Evidence, Fact, Family, agreement_bonus,
                                           combine_scores, confidence_score,
                                           evidence_level)
from backend.app.services.payee_check import HARD_BLOCK                                  # noqa: E402

SRC = RESULTS / "graph_replay"
OUT = RESULTS / "end_to_end"
TAB = RESULTS / "tables"
FIG = RESULTS / "figures"
VERDICTS = ["APPROVE", "WARN", "STEP_UP", "BLOCK"]


def rebuild(row, families, drop=None) -> list[Evidence]:
    out = []
    for fam in families:
        if fam == drop:
            continue
        if not row[f"avail__{fam}"]:
            out.append(Evidence(family=Family(fam), available=False))
            continue
        facts = row[f"facts__{fam}"]
        facts = [Fact(f) for f in str(facts).split("|")] if isinstance(facts, str) and facts else []
        out.append(Evidence(family=Family(fam), available=True,
                            score=int(row[f"score__{fam}"]), facts=frozenset(facts)))
    return out


def recompute(evs, hard_block: bool):
    base = combine_scores([e.score for e in evs if e.available])
    bonus, facts = agreement_bonus(evs)
    risk = min(99, base + bonus)
    verdict = "BLOCK" if hard_block else verdict_from_score(risk)
    return risk, bonus, verdict, base


def family_count_bonus(evs) -> int:
    """What the bonus would be if it counted voting FAMILIES (the pre-audit rule)."""
    n = sum(1 for e in evs if e.counts_toward_agreement and e.score >= AGREEMENT_THRESHOLD and e.facts)
    return 0 if n < 2 else AGREEMENT_BONUS.get(n, max(AGREEMENT_BONUS.values()))


def detection(y, flagged) -> dict:
    tp = int((flagged & (y == 1)).sum()); fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum()); tn = int((~flagged & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": r, "precision": p,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "fpr": fp / max(fp + tn, 1),
            "flag_rate": float(flagged.mean())}


def risk_block(d: pd.DataFrame, label: str) -> dict:
    y = d.is_fraud.to_numpy(); risk = d.risk.to_numpy()
    out = {"slice": label, "n": int(len(d)), "n_fraud": int(y.sum()), "prevalence": float(y.mean()),
           "risk_pr_auc": float(average_precision_score(y, risk)),
           "risk_roc_auc": float(roc_auc_score(y, risk))}
    ml = d.ml_probability
    m = ml.notna().to_numpy()
    out["ml_available_rate"] = float(m.mean())
    if m.sum() and len(np.unique(y[m])) == 2:
        out["ml_only_pr_auc_where_available"] = float(average_precision_score(y[m], ml[m]))
        out["ml_only_roc_auc_where_available"] = float(roc_auc_score(y[m], ml[m]))
        out["risk_pr_auc_same_rows"] = float(average_precision_score(y[m], risk[m]))
        out["risk_roc_auc_same_rows"] = float(roc_auc_score(y[m], risk[m]))
    for name, thr in (("warn_plus", WARN_AT), ("step_up_plus", STEP_UP_AT), ("block", BLOCK_AT)):
        flagged = d.verdict.isin(VERDICTS[VERDICTS.index({"warn_plus": "WARN", "step_up_plus": "STEP_UP",
                                                          "block": "BLOCK"}[name]):]).to_numpy()
        out[name] = detection(y, flagged)
    out["verdict_share_fraud"] = {v: float((d.loc[d.is_fraud == 1, "verdict"] == v).mean()) for v in VERDICTS}
    out["verdict_share_legit"] = {v: float((d.loc[d.is_fraud == 0, "verdict"] == v).mean()) for v in VERDICTS}
    out["intervention_rate_all"] = float((d.verdict != "APPROVE").mean())
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input", default=None, help="evidence CSV (default: the seed's replay output)")
    args = ap.parse_args()
    S = f"seed{args.seed}"
    src = (Path(args.input) if args.input else SRC / f"replay_evidence_{S}.csv").resolve()
    ev = pd.read_csv(src, low_memory=False)
    ev = ev.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    families = sorted(c.split("__", 1)[1] for c in ev.columns if c.startswith("avail__"))
    assert len(families) == 10, families
    n = len(ev)
    test_start = int(n * 0.75)
    ev["period"] = np.where(np.arange(n) >= test_start, "test_last25", "first75")

    # ── 1. consistency audit + counterfactuals ─────────────────────────
    mism = Counter(); cf_rows = []
    fam_count_bonus = np.zeros(n, dtype=int); fact_share = np.zeros(n, dtype=int)
    leading = []
    codes = ev.finding_codes.fillna("").astype(str)
    hard = codes.apply(lambda s: any(c in HARD_BLOCK for c in s.split("|") if c)).to_numpy()
    for i, row in enumerate(ev.to_dict("records")):
        evs = rebuild(row, families)
        risk, bonus, verdict, base = recompute(evs, hard[i])
        if risk != row["risk"]: mism["risk"] += 1
        if bonus != row["bonus"]: mism["bonus"] += 1
        if verdict != row["verdict"]: mism["verdict"] += 1
        if confidence_score(evs) != row["coverage"]: mism["coverage"] += 1
        if evidence_level(evs).value != row["evidence_level"]: mism["evidence_level"] += 1
        fam_count_bonus[i] = family_count_bonus(evs)
        by_fact = Counter(f for e in evs if e.counts_toward_agreement and e.score >= AGREEMENT_THRESHOLD
                          for f in e.facts)
        fact_share[i] = sum(1 for c in by_fact.values() if c > 1)
        scored = [(e.score, e.family.value) for e in evs if e.available and e.score > 0]
        top = max(s for s, _ in scored) if scored else 0
        leading.append([f for s, f in scored if s == top] if scored else [])
        cf = {}
        for fam in families:
            r2, _, v2, _ = recompute(rebuild(row, families, drop=fam), hard[i])
            cf[f"drisk__{fam}"] = risk - r2
            cf[f"vchg__{fam}"] = int(v2 != verdict)
            cf[f"vdown__{fam}"] = int(VERDICTS.index(v2) < VERDICTS.index(verdict))
        cf_rows.append(cf)
    cf = pd.DataFrame(cf_rows)
    consistency = {"rows": n, "mismatches": dict(mism), "hard_block_rows": int(hard.sum()),
                   "rule": "evidence.combine_scores + agreement_bonus (distinct facts, >=2 families, "
                           "score>=20) -> min(99, .); verdict_from_score bands 22/45/70; HARD_BLOCK findings"}
    print("consistency:", consistency, flush=True)

    # ── 2. risk and verdicts ────────────────────────────────────────────
    slices = [risk_block(ev, "full_stream"), risk_block(ev[ev.period == "test_last25"], "test_last25pct")]
    for s in SCENARIOS:
        d = ev[(ev.scenario == s) | (ev.is_fraud == 0)]
        if d.is_fraud.nunique() < 2:
            continue
        slices.append(risk_block(d, f"scenario={s}_vs_all_legit"))
    # maturity of the shared store: 30-day blocks of the replay window
    day = (pd.to_datetime(ev.timestamp) - pd.to_datetime(ev.timestamp).min()).dt.total_seconds() / 86400
    ev["day_block"] = (day // 30).astype(int)
    for b in sorted(ev.day_block.unique()):
        d = ev[ev.day_block == b]
        if d.is_fraud.nunique() == 2:
            slices.append(risk_block(d, f"days_{30*b}-{30*b+30}"))
    for lvl in ("FULL", "PARTIAL", "MINIMAL"):
        d = ev[ev.evidence_level == lvl]
        if d.is_fraud.nunique() == 2:
            slices.append(risk_block(d, f"evidence_level={lvl}"))
    # Operating point at a 1% FPR budget, chosen on the 60-75% window of the stream (the window that
    # precedes the evaluation slice, as in Experiment 4's validation split) - never on the last 25%
    first = ev.iloc[int(n * 0.60):test_start]; last = ev[ev.period == "test_last25"]
    budget = {}
    for name, col in (("risk", "risk"), ("ml_probability", "ml_probability")):
        f_ = first[first[col].notna()]; l_ = last[last[col].notna()]
        cand = np.unique(f_[col].to_numpy())
        tau = None
        for c in cand:                                   # smallest threshold whose FPR on the first 75% <= 1%
            if (f_.loc[f_.is_fraud == 0, col] >= c).mean() <= 0.01:
                tau = float(c); break
        if tau is None:
            budget[name] = {"threshold": None, "note": "no threshold meets FPR<=1% on the selection window"}
            continue
        d = detection(l_.is_fraud.to_numpy(), (l_[col] >= tau).to_numpy())
        d.update({"threshold": tau, "rows": int(len(l_)), "chosen_on": "60-75% window of the stream",
                  "selection_window_fpr": float((f_.loc[f_.is_fraud == 0, col] >= tau).mean())})
        budget[name] = d
    # the live classifier on its own, at its stored operating threshold (models/metrics.json)
    from backend.models.ensemble_model import load_metrics
    theta = float(load_metrics()["selected"]["operating_point"]["threshold"])
    l_ = last[last.ml_probability.notna()]
    d = detection(l_.is_fraud.to_numpy(), (l_.ml_probability >= theta).to_numpy())
    d.update({"threshold": theta, "rows": int(len(l_)), "pr_auc": float(average_precision_score(l_.is_fraud, l_.ml_probability)),
              "roc_auc": float(roc_auc_score(l_.is_fraud, l_.ml_probability))})
    for sc_ in SCENARIOS:
        m_ = (l_.scenario == sc_) & (l_.is_fraud == 1)
        d[f"recall_{sc_}"] = float((l_.loc[m_, "ml_probability"] >= theta).mean()) if m_.any() else float("nan")
    budget["live_v1_at_stored_threshold_last25"] = d
    for op, first_v in (("warn_plus", "WARN"), ("step_up_plus", "STEP_UP"), ("block", "BLOCK")):
        fl = last.verdict.isin(VERDICTS[VERDICTS.index(first_v):])
        rec = {}
        for sc_ in SCENARIOS:
            m_ = (last.scenario == sc_) & (last.is_fraud == 1)
            rec[f"recall_{sc_}"] = float(fl[m_].mean()) if m_.any() else float("nan")
        budget[f"scenario_recall_{op}_last25"] = rec
    lg = last[last.is_fraud == 0]
    sel = ev.iloc[int(n * 0.60):test_start]; sel_lg = sel[sel.is_fraud == 0]
    lead_last = [leading[i] for i in last.index]
    block_legit_lead = Counter(f for i, l in zip(last.index, lead_last)
                               if ev.at[i, "is_fraud"] == 0 and ev.at[i, "verdict"] == "BLOCK" for f in l)
    codes_block_legit = Counter(c for c in last.loc[(last.is_fraud == 0) & (last.verdict == "BLOCK"), "finding_codes"]
                                .fillna("").astype(str).str.split("|").sum() if c)
    budget["risk_saturation"] = {
        "legit_share_R99_selection_window": float((sel_lg.risk >= 99).mean()),
        "legit_share_R99_last25": float((lg.risk >= 99).mean()),
        "legit_share_R_ge_70_last25": float((lg.risk >= 70).mean()),
        "leading_family_counts_legit_BLOCK_last25": dict(block_legit_lead.most_common()),
        "finding_codes_legit_BLOCK_last25_top": dict(codes_block_legit.most_common(12)),
        "n_legit_BLOCK_last25": int(((last.is_fraud == 0) & (last.verdict == "BLOCK")).sum()),
    }
    print("1% budget:", budget, flush=True)
    flat = []
    for s in slices:
        row = {k: v for k, v in s.items() if not isinstance(v, dict)}
        for op in ("warn_plus", "step_up_plus", "block"):
            for k, v in s[op].items():
                row[f"{op}_{k}"] = v
        for v in VERDICTS:
            row[f"share_fraud_{v}"] = s["verdict_share_fraud"][v]
            row[f"share_legit_{v}"] = s["verdict_share_legit"][v]
        flat.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(OUT / f"e2e_results_{S}.csv", index=False)

    def vmix(d):
        g = d.assign(group=np.where(d.is_fraud == 1, d.scenario, "legitimate"))
        return (g.groupby("group").verdict.value_counts(normalize=True).unstack(fill_value=0)
                .reindex(columns=VERDICTS, fill_value=0)), g.groupby("group").size()
    verdict_by_scenario, counts_by_scenario = vmix(ev)
    verdict_by_scenario_test, counts_by_scenario_test = vmix(ev[ev.period == "test_last25"])

    # ── 3. family metrics ────────────────────────────────────────────────
    fam_rows = []
    for slice_name, msk in (("full_stream", np.ones(n, bool)), ("test_last25pct", (ev.period == "test_last25").to_numpy())):
        E = ev[msk]; CF = cf[msk]; y = E.is_fraud.to_numpy()
        LEAD = [l for l, m_ in zip(leading, msk) if m_]
        for fam in families:
            av = E[f"avail__{fam}"].to_numpy().astype(bool)
            sc = E[f"score__{fam}"].fillna(0).to_numpy()
            act = av & (sc > 0)
            voting = act & (sc >= AGREEMENT_THRESHOLD)
            lead = np.array([fam in l for l in LEAD])
            r = {"slice": slice_name, "family": fam,
                 "available_rate": float(av.mean()),
                 "available_rate_fraud": float(av[y == 1].mean()), "available_rate_legit": float(av[y == 0].mean()),
                 "active_rate_fraud": float(act[y == 1].mean()), "active_rate_legit": float(act[y == 0].mean()),
                 "p_fraud_given_active": float(y[act].mean()) if act.sum() else float("nan"),
                 "lift_given_active": float(y[act].mean() / y.mean()) if act.sum() else float("nan"),
                 "n_active": int(act.sum()),
                 "mean_score_active_fraud": float(sc[act & (y == 1)].mean()) if (act & (y == 1)).sum() else float("nan"),
                 "mean_score_active_legit": float(sc[act & (y == 0)].mean()) if (act & (y == 0)).sum() else float("nan"),
                 "auc_within_available": (float(roc_auc_score(y[av], sc[av]))
                                          if av.sum() and len(np.unique(y[av])) == 2 and np.unique(sc[av]).size > 1
                                          else float("nan")),
                 "voting_rate_fraud": float(voting[y == 1].mean()), "voting_rate_legit": float(voting[y == 0].mean()),
                 "leading_rate_fraud": float(lead[y == 1].mean()), "leading_rate_legit": float(lead[y == 0].mean()),
                 "loo_mean_drisk_fraud": float(CF[f"drisk__{fam}"][y == 1].mean()),
                 "loo_mean_drisk_legit": float(CF[f"drisk__{fam}"][y == 0].mean()),
                 "loo_verdict_change_fraud": float(CF[f"vchg__{fam}"][y == 1].mean()),
                 "loo_verdict_change_legit": float(CF[f"vchg__{fam}"][y == 0].mean()),
                 "loo_verdict_down_fraud": float(CF[f"vdown__{fam}"][y == 1].mean()),
                 "loo_verdict_down_legit": float(CF[f"vdown__{fam}"][y == 0].mean())}
            for s in SCENARIOS:
                m = (E.scenario == s).to_numpy() & (y == 1)
                r[f"active_rate_{s}"] = float(act[m].mean()) if m.sum() else float("nan")
            # risk PR-AUC without this family (leave-one-out), full stream
            r["loo_risk_pr_auc"] = float(average_precision_score(y, E.risk.to_numpy() - CF[f"drisk__{fam}"].to_numpy()))
            fam_rows.append(r)
    fam_df = pd.DataFrame(fam_rows)
    y = ev.is_fraud.to_numpy()
    fam_df.to_csv(OUT / f"evidence_family_metrics_{S}.csv", index=False)

    # ── 4. corroboration / double counting ──────────────────────────────
    bon = ev.bonus.to_numpy()
    corro = {
        "bonus_rate_fraud": float((bon[y == 1] > 0).mean()), "bonus_rate_legit": float((bon[y == 0] > 0).mean()),
        "bonus_distribution_fraud": {int(k): int(v) for k, v in Counter(bon[y == 1]).items()},
        "bonus_distribution_legit": {int(k): int(v) for k, v in Counter(bon[y == 0]).items()},
        "rows_with_fact_observed_by_2plus_voting_families": int((fact_share > 0).sum()),
        "rows_where_family_count_rule_exceeds_fact_rule": int((fam_count_bonus > bon).sum()),
        "rows_where_family_count_rule_exceeds_fact_rule_fraud": int(((fam_count_bonus > bon) & (y == 1)).sum()),
        "mean_bonus_removed_by_fact_dedup_when_positive": (float((fam_count_bonus - bon)[fam_count_bonus > bon].mean())
                                                          if (fam_count_bonus > bon).any() else 0.0),
        "ml_classifier_rows_with_facts": int(ev["facts__ml_classifier"].fillna("").astype(str).str.len().gt(0).sum()),
        "unavailable_rows_with_nonzero_score": int(sum(((ev[f"avail__{f}"] == 0) & ev[f"score__{f}"].fillna(0).ne(0)).sum()
                                                   for f in families)),
    }
    risk_cov = {
        "spearman_risk_coverage_all": float(ev[["risk", "coverage"]].corr(method="spearman").iloc[0, 1]),
        "spearman_risk_coverage_legit": float(ev.loc[ev.is_fraud == 0, ["risk", "coverage"]].corr(method="spearman").iloc[0, 1]),
        "coverage_median_fraud": float(ev.coverage[y == 1].median()), "coverage_median_legit": float(ev.coverage[y == 0].median()),
        "evidence_level_share_fraud": ev.loc[y == 1, "evidence_level"].value_counts(normalize=True).to_dict(),
        "evidence_level_share_legit": ev.loc[y == 0, "evidence_level"].value_counts(normalize=True).to_dict(),
        "risk_by_coverage_bin": (ev.assign(cov_bin=pd.cut(ev.coverage, [0, 40, 60, 80, 100], include_lowest=True))
                                 .groupby(["cov_bin", "is_fraud"], observed=True).risk.median().unstack()
                                 .rename(columns={0: "legit_median_risk", 1: "fraud_median_risk"})
                                 .reset_index().astype({"cov_bin": str}).to_dict("records")),
        "payee_status_share_fraud": ev.loc[y == 1, "payee_status"].value_counts(normalize=True).to_dict(),
        "payee_status_share_legit": ev.loc[y == 0, "payee_status"].value_counts(normalize=True).to_dict(),
    }

    # audit sample: 20 fraud + 20 legitimate, stratified over verdicts, from the full audit log
    full = [json.loads(l) for l in open(OUT / f"e2e_audit_full_{S}.jsonl")]
    rng = np.random.default_rng(42)
    sample = []
    for lab in (1, 0):
        pool = [r for r in full if r["label"] == lab]
        for v in VERDICTS:
            sub = [r for r in pool if r["verdict"] == v]
            k = min(5, len(sub))
            sample += [sub[j] for j in rng.choice(len(sub), size=k, replace=False)] if k else []
    with open(OUT / f"e2e_audit_sample_{S}.jsonl", "w") as fh:
        for r in sample:
            fh.write(json.dumps(r, default=str) + "\n")

    summary = {
        "what": "Experiment 3: the running ten-family assessment (payee_check.check_payee) evaluated on the "
                "chronological replay stream. Seed: " + str(args.seed) + ". Continuous risk is evaluated separately from the "
                "advisory verdict mix. Labels are synthetic.",
        "rows": n, "fraud": int(y.sum()), "families": families,
        "consistency_audit": consistency,
        "risk_and_verdicts": slices,
        "verdict_share_by_group": verdict_by_scenario.round(6).to_dict("index"),
        "rows_by_group": counts_by_scenario.to_dict(),
        "verdict_share_by_group_test_last25pct": verdict_by_scenario_test.round(6).to_dict("index"),
        "rows_by_group_test_last25pct": counts_by_scenario_test.to_dict(),
        "corroboration": corro, "risk_vs_coverage": risk_cov,
        "one_percent_budget_on_last25pct": budget,
        "notes": [
            "ml_probability is the RUNNING V1 model (models/risk_model.pkl), trained on the row-level "
            "simulator, applied here to a different (replay) corpus without refitting.",
            "Verdict thresholds are the application's fixed bands, not tuned on this stream.",
            "Leave-one-family-out re-applies the same aggregation without that family; it is a "
            "counterfactual of the aggregation rule, not a retrained system.",
        ],
        "audit_sample": {"file": f"e2e_audit_sample_{S}.jsonl", "n": len(sample),
                         "design": "up to 5 rows per (label, verdict), rng seed 42"},
        "input_sha256": sha256_file(src), "input": str(src.relative_to(RESULTS.parent)),
        "environment": environment(), "source": source_state(),
    }
    write_json(OUT / f"e2e_summary_{S}.json", summary)

    # ── tables ──────────────────────────────────────────────────────────
    TAB.mkdir(parents=True, exist_ok=True)
    s0, s1 = slices[0], slices[1]
    td = []
    for s in (s0, s1):
        td.append({"Slice": s["slice"], "n": s["n"], "fraud": s["n_fraud"],
                   "Risk PR-AUC": f"{s['risk_pr_auc']:.4f}", "Risk ROC-AUC": f"{s['risk_roc_auc']:.4f}",
                   "Classifier-only PR-AUC": f"{s.get('ml_only_pr_auc_where_available', float('nan')):.4f}",
                   "WARN+ recall %": f"{100*s['warn_plus']['recall']:.2f}", "WARN+ FPR %": f"{100*s['warn_plus']['fpr']:.2f}",
                   "STEP_UP+ recall %": f"{100*s['step_up_plus']['recall']:.2f}", "STEP_UP+ FPR %": f"{100*s['step_up_plus']['fpr']:.2f}",
                   "BLOCK recall %": f"{100*s['block']['recall']:.2f}", "BLOCK FPR %": f"{100*s['block']['fpr']:.3f}",
                   "Intervention %": f"{100*s['intervention_rate_all']:.2f}"})
    tdf = pd.DataFrame(td)
    tdf.to_csv(TAB / f"table_D_e2e_{S}.csv", index=False)
    vb = (100 * verdict_by_scenario).round(2); vb.insert(0, "n", counts_by_scenario); vb.insert(0, "slice", "full_stream")
    vt = (100 * verdict_by_scenario_test).round(2); vt.insert(0, "n", counts_by_scenario_test); vt.insert(0, "slice", "test_last25pct")
    vb = pd.concat([vb, vt])
    vb.to_csv(TAB / f"table_D2_verdict_by_group_{S}.csv")
    (TAB / f"table_D_e2e_{S}.md").write_text(tdf.to_markdown(index=False) + "\n\n" + vb.to_markdown(), encoding="utf-8")
    te = fam_df[["slice", "family", "available_rate", "active_rate_fraud", "active_rate_legit", "p_fraud_given_active",
                 "lift_given_active", "auc_within_available", "leading_rate_fraud", "loo_mean_drisk_fraud",
                 "loo_verdict_down_fraud", "loo_verdict_down_legit"]].copy()
    te.to_csv(TAB / f"table_E_families_{S}.csv", index=False)
    (TAB / f"table_E_families_{S}.md").write_text(te.round(4).to_markdown(index=False), encoding="utf-8")

    # ── figures ─────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})
    T = ev[ev.period == "test_last25"]; yT = T.is_fraud.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), gridspec_kw={"width_ratios": [1, 1.05, 1]})
    ax = axes[0]
    bins = np.arange(0, 101, 4)
    ax.hist(T.risk[yT == 0], bins=bins, density=True, histtype="stepfilled", alpha=0.35, color="#4C72B0", ec="#4C72B0", lw=0.8, label="Legitimate")
    ax.hist(T.risk[yT == 1], bins=bins, density=True, histtype="step", color="#C44E52", lw=1.1, label="Fraud")
    for t in (WARN_AT, STEP_UP_AT, BLOCK_AT):
        ax.axvline(t, color="k", ls="--", lw=0.6)
    ax.set_yscale("log"); ax.set_xlabel("Risk score $R$"); ax.set_ylabel("Density (log scale)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center"); ax.set_title("(a) Risk by label, last 25%", fontsize=7.5)
    ax = axes[1]
    order = ["legitimate"] + [s_ for s_ in SCENARIOS if s_ in verdict_by_scenario_test.index]
    left = np.zeros(len(order)); greys = ["0.9", "0.68", "0.42", "0.12"]
    for v, col in zip(VERDICTS, greys):
        vals = 100 * verdict_by_scenario_test.loc[order, v].to_numpy()
        ax.barh(range(len(order)), vals, left=left, color=col, label=v.replace("_", "-"), height=0.62,
                edgecolor="k", lw=0.3)
        left += vals
    names = {"legitimate": "Legitimate", "social_engineering": "Social eng.", "mule_collection": "Mule coll.",
             "account_takeover": "Acct. takeover", "card_testing": "Card testing"}
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[o] for o in order], fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlabel("Share of payments (%)"); ax.set_xlim(0, 100)
    ax.legend(ncol=4, fontsize=5.2, frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.17), handlelength=1.0, columnspacing=0.6, handletextpad=0.3)
    ax.set_title("(b) Verdict mix, last 25%", fontsize=7.5, pad=13)
    ax = axes[2]
    blk = [s_ for s_ in slices if s_["slice"].startswith("days_")]
    xs = [int(s_["slice"].split("_")[1].split("-")[0]) + 15 for s_ in blk]
    ax.plot(xs, [100 * s_["warn_plus"]["fpr"] for s_ in blk], "o-", color="#4C72B0", ms=3, lw=1, label="Legit. at WARN+")
    ax.plot(xs, [100 * s_["block"]["fpr"] for s_ in blk], "s--", color="#4C72B0", ms=3, lw=1, label="Legit. at BLOCK")
    ax.plot(xs, [100 * s_["warn_plus"]["recall"] for s_ in blk], "o-", color="#C44E52", ms=3, lw=1, label="Fraud at WARN+")
    ax.plot(xs, [100 * s_["block"]["recall"] for s_ in blk], "s--", color="#C44E52", ms=3, lw=1, label="Fraud at BLOCK")
    ax.set_xlabel("Day of stream (30-day blocks)"); ax.set_ylabel("Share flagged (%)"); ax.set_ylim(0, 100)
    ax.legend(fontsize=5.6, frameon=False, ncol=1, loc="best"); ax.grid(lw=0.3, alpha=0.5)
    ax.set_title("(c) Store maturity", fontsize=7.5)
    fig.tight_layout(w_pad=0.8); fig.savefig(FIG / f"fig5_e2e_risk_verdicts_{S}.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    fd = fam_df[fam_df.slice == "full_stream"].set_index("family")
    order = fd.sort_values("active_rate_fraud").index
    yy = np.arange(len(order))
    ax.barh(yy + 0.2, 100 * fd.loc[order, "active_rate_fraud"], height=0.4, color="#C44E52", label="Active, fraud")
    ax.barh(yy - 0.2, 100 * fd.loc[order, "active_rate_legit"], height=0.4, color="#4C72B0", label="Active, legitimate")
    ax.set_yticks(yy); ax.set_yticklabels([o.replace("_", " ") for o in order], fontsize=6.5)
    ax.set_xlabel("Rows where family scored > 0 (%)"); ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.grid(axis="x", lw=0.3, alpha=0.6)
    fig.tight_layout(); fig.savefig(FIG / f"fig6_family_activation_{S}.png", dpi=300); plt.close(fig)

    print(tdf.to_string(index=False)); print(vb.to_string()); print(te.round(4).to_string(index=False))
    print(json.dumps(corro, indent=1)); print(json.dumps({k: v for k, v in risk_cov.items()
                                                          if k != "risk_by_coverage_bin"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
