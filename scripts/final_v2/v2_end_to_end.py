"""Ten-family end-to-end evaluation with the FROZEN V2 model as the Classifier family.

Why a substitution is needed, and why it is valid
-------------------------------------------------
The running prototype's Classifier family serves the V1 artefact. The final
research configuration is V2-only, so this analysis replaces that one family
with the frozen V2 model (models/risk_model_v2.pkl, threshold 0.1123) and keeps
the other nine families EXACTLY as the production function
payee_check.check_payee computed them during the chronological replay
(results/graph_replay/replay_evidence_seed<S>.csv).

  * The V2 feature vector of every replayed payment was built during the
    replay by payee_history.build_v2_vector from the same pre-payment state
    (payer rows < t, store < t) and is stored in the v2__* columns.
  * The V2 family is scored with the production mapping (payee_check: graded
    score 0-18 below the threshold, 30-66 above, ML_SOLO_CEILING = 66) and the
    production counterfactual fact probes (ML_PROBES, PAYEE_HISTORY_FEATURES,
    ML_FACT_RELATIVE = 0.15, ML_FACT_FLOOR = 0.01), applied to the V2 frame.
  * Its availability is the production availability (amount known, payer
    profile, >= 5 prior payments), i.e. the recorded avail__ml_classifier.
  * Risk, corroboration, coverage, level and verdict are recomputed with the
    production functions evidence.combine_scores / agreement_bonus /
    confidence_score / evidence_level and core.verdict.verdict_from_score plus
    the HARD_BLOCK rule. The same re-aggregation reproduces every recorded
    replay output exactly (consistency audit, 0 mismatches over 180,000 rows),
    which is what makes the substitution sound.
  * The replay store is unaffected by the classifier (observations are written
    by record_payment regardless of the verdict), so no re-replay is needed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.app.core.verdict import BLOCK_AT, STEP_UP_AT, WARN_AT, verdict_from_score      # noqa: E402
from backend.app.services.evidence import (AGREEMENT_THRESHOLD, Evidence, Fact, Family,      # noqa: E402
                                           agreement_bonus, combine_scores, confidence_score,
                                           evidence_level)
from backend.app.services.payee_check import (HARD_BLOCK, ML_FACT_FLOOR, ML_FACT_RELATIVE,  # noqa: E402
                                              ML_PROBES, ML_SOLO_CEILING)
from backend.app.services.vpa import graded                                                 # noqa: E402
from backend.ml.features import PAYEE_HISTORY_FEATURES                                      # noqa: E402

import style                                                                                 # noqa: E402
from style import FRAUD, LEGIT, INK2, MUTED                                                  # noqa: E402

G = ROOT / "results" / "graph_replay"
R = ROOT / "results" / "final_v2"
VERDICTS = ["APPROVE", "WARN", "STEP_UP", "BLOCK"]
SCEN = ["social_engineering", "mule_collection", "account_takeover", "card_testing"]
FAM_LABEL = {"payee_history": "Payee history", "relationship": "Relationship", "payer_behaviour": "Payer behaviour",
             "ml_classifier": "Classifier (V2)", "address_and_qr": "Address and QR", "graph": "Graph",
             "amount_context": "Amount context", "message_pressure": "Message pressure", "link_safety": "Link safety",
             "stated_intent": "Stated intent"}


def v2_family(ev: pd.DataFrame, model, cols, thr):
    """Production Classifier-family mapping applied to the frozen V2 model."""
    avail = ev["avail__ml_classifier"].to_numpy().astype(bool)
    X = ev[[f"v2__{c}" for c in cols]].copy(); X.columns = cols
    X = X.astype(float)
    p = np.full(len(ev), np.nan)
    p[avail] = model.predict_proba(X[avail])[:, 1]
    score = np.zeros(len(ev), dtype=int)
    for i in np.where(avail)[0]:
        pi = p[i]
        score[i] = int(round(graded(0.0, thr, pi, 0, 18))) if pi < thr else int(round(graded(thr, 0.80, pi, 30, ML_SOLO_CEILING)))
    facts = [set() for _ in range(len(ev))]
    cut = np.maximum(ML_FACT_FLOOR, ML_FACT_RELATIVE * np.nan_to_num(p))
    Xa = X[avail]; ia = np.where(avail)[0]
    ph = Xa["payee_history_available"].to_numpy() > 0.5
    if ph.any():
        blind = Xa[ph].copy()
        for c in PAYEE_HISTORY_FEATURES:
            blind[c] = np.nan
        blind["payee_history_available"] = 0.0
        pb = model.predict_proba(blind)[:, 1]
        for j, i in enumerate(ia[ph]):
            if p[i] - pb[j] >= cut[i]:
                facts[i].add(Fact.PAYEE_SHAPE)
    for fact, neutral in ML_PROBES.items():
        probe = Xa.copy()
        for c, v in neutral.items():
            probe[c] = v
        pp = model.predict_proba(probe)[:, 1]
        for j, i in enumerate(ia):
            if p[i] - pp[j] >= cut[i]:
                facts[i].add(fact)
    return avail, p, score, facts


def evidence_list(row, fams, ml):
    out = []
    for fam in fams:
        if fam == "ml_classifier":
            a, s, f = ml
            out.append(Evidence(Family.ML_CLASSIFIER, available=bool(a), score=int(s) if a else 0,
                                facts=frozenset(f) if a else frozenset()))
            continue
        if not row[f"avail__{fam}"]:
            out.append(Evidence(Family(fam), available=False)); continue
        fs = row[f"facts__{fam}"]
        fs = [Fact(x) for x in str(fs).split("|")] if isinstance(fs, str) and fs else []
        out.append(Evidence(Family(fam), available=True, score=int(row[f"score__{fam}"]), facts=frozenset(fs)))
    return out


def aggregate(evs, hard):
    base = combine_scores([e.score for e in evs if e.available])
    bonus, _ = agreement_bonus(evs)
    risk = min(99, base + bonus)
    return risk, bonus, ("BLOCK" if hard else verdict_from_score(risk)), confidence_score(evs), evidence_level(evs).value


def detection(y, flagged):
    tp = int((flagged & (y == 1)).sum()); fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum()); tn = int((~flagged & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": r, "precision": p, "fpr": fp / (fp + tn),
            "f1": 2 * p * r / (p + r) if (tp and p + r) else 0.0}


def run_seed(seed, model, cols, thr):
    ev = pd.read_csv(G / f"replay_evidence_seed{seed}.csv", low_memory=False)
    ev = ev.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    fams = sorted(c.split("__", 1)[1] for c in ev.columns if c.startswith("avail__"))
    avail, p, score, facts = v2_family(ev, model, cols, thr)
    hard = ev.finding_codes.fillna("").astype(str).apply(lambda s: any(c in HARD_BLOCK for c in s.split("|") if c)).to_numpy()
    out = {k: [] for k in ("risk", "bonus", "verdict", "coverage", "level", "leading", "drisk_nov2")}
    mism_cov = mism_lvl = 0
    for i, row in enumerate(ev.to_dict("records")):
        evs = evidence_list(row, fams, (avail[i], score[i], facts[i]))
        risk, bonus, verdict, cov, lvl = aggregate(evs, hard[i])
        mism_cov += cov != row["coverage"]; mism_lvl += lvl != row["evidence_level"]
        scored = [(e.score, e.family.value) for e in evs if e.available and e.score > 0]
        top = max((s for s, _ in scored), default=0)
        out["leading"].append([f for s, f in scored if s == top] if scored else [])
        for k, v in (("risk", risk), ("bonus", bonus), ("verdict", verdict), ("coverage", cov), ("level", lvl)):
            out[k].append(v)
        # contribution of the V2 family: risk with it removed
        r2, _, _, _, _ = aggregate([e for e in evs if e.family != Family.ML_CLASSIFIER], hard[i])
        out["drisk_nov2"].append(risk - r2)
    res = pd.DataFrame({"txn_id": ev.txn_id, "timestamp": ev.timestamp, "is_fraud": ev.is_fraud, "scenario": ev.scenario,
                        "payee_status": ev.payee_status, "v2_probability": p, "v2_family_available": avail.astype(int),
                        "v2_family_score": np.where(avail, score, np.nan),
                        "v2_family_facts": ["|".join(sorted(f.value for f in s)) for s in facts],
                        "risk": out["risk"], "bonus": out["bonus"], "verdict": out["verdict"], "coverage": out["coverage"],
                        "evidence_level": out["level"], "risk_change_if_v2_removed": out["drisk_nov2"],
                        **{f"avail__{f}": ev[f"avail__{f}"] for f in fams if f != "ml_classifier"},
                        **{f"score__{f}": ev[f"score__{f}"] for f in fams if f != "ml_classifier"},
                        "finding_codes": ev.finding_codes, "graph_position_fired": ev.graph_position_fired})
    res["leading_families"] = ["|".join(l) for l in out["leading"]]
    assert mism_cov == 0 and mism_lvl == 0, "coverage/level must not depend on the classifier score"
    return res, fams


def summarise(res, seed):
    n = len(res); t0 = int(n * 0.75)
    res = res.copy(); res["period"] = np.where(np.arange(n) >= t0, "last25", "first75")
    out = {"seed": seed, "rows": n, "fraud": int(res.is_fraud.sum())}
    for name, d in (("full_stream", res), ("last25pct", res[res.period == "last25"])):
        y = d.is_fraud.to_numpy(); r = d.risk.to_numpy()
        m = d.v2_family_available.to_numpy() == 1
        s = {"n": int(len(d)), "fraud": int(y.sum()), "prevalence": float(y.mean()),
             "risk_pr_auc": float(average_precision_score(y, r)), "risk_roc_auc": float(roc_auc_score(y, r)),
             "v2_available_rate": float(m.mean()),
             "v2_alone_pr_auc_where_available": float(average_precision_score(y[m], d.v2_probability[m])),
             "v2_alone_roc_auc_where_available": float(roc_auc_score(y[m], d.v2_probability[m])),
             "risk_pr_auc_same_rows": float(average_precision_score(y[m], r[m])),
             "risk_roc_auc_same_rows": float(roc_auc_score(y[m], r[m])),
             "v2_alone_at_threshold": detection(y[m], (d.v2_probability[m] >= 0.1123).to_numpy()),
             "verdict_counts": {g: {v: int(((d.verdict == v) & (grp_mask(d, g))).sum()) for v in VERDICTS}
                                for g in ["all", "legitimate", "fraud"] + SCEN},
             "coverage": coverage_stats(d), "evidence_level_counts": d.evidence_level.value_counts().to_dict()}
        for op, first in (("warn_plus", "WARN"), ("step_up_plus", "STEP_UP"), ("block", "BLOCK")):
            fl = d.verdict.isin(VERDICTS[VERDICTS.index(first):]).to_numpy()
            s[op] = detection(y, fl)
            s[op]["scenario_recall"] = {sc: float(fl[(d.scenario == sc).to_numpy() & (y == 1)].mean()) for sc in SCEN}
        s["intervention_rate"] = float((d.verdict != "APPROVE").mean())
        out[name] = s
    # 30-day blocks (store maturity)
    day = (pd.to_datetime(res.timestamp) - pd.to_datetime(res.timestamp).min()).dt.total_seconds() / 86400
    out["blocks"] = []
    for b in sorted((day // 30).astype(int).unique()):
        d = res[(day // 30).astype(int) == b]; y = d.is_fraud.to_numpy()
        out["blocks"].append({"days": f"{30*b}-{30*b+30}", "n": int(len(d)), "fraud": int(y.sum()),
                              "legit_block_rate": float(((d.verdict == "BLOCK") & (y == 0)).sum() / max((y == 0).sum(), 1)),
                              "legit_warn_plus_rate": float(((d.verdict != "APPROVE") & (y == 0)).sum() / max((y == 0).sum(), 1)),
                              "fraud_block_recall": float(((d.verdict == "BLOCK") & (y == 1)).sum() / max(y.sum(), 1)),
                              "fraud_warn_plus_recall": float(((d.verdict != "APPROVE") & (y == 1)).sum() / max(y.sum(), 1)),
                              "risk_roc_auc": float(roc_auc_score(y, d.risk)) if 0 < y.sum() < len(y) else None})
    return out


def grp_mask(d, g):
    if g == "all":
        return pd.Series(True, index=d.index)
    if g == "legitimate":
        return d.is_fraud == 0
    if g == "fraud":
        return d.is_fraud == 1
    return (d.scenario == g) & (d.is_fraud == 1)


def coverage_stats(d):
    c = d.coverage
    st = {"mean": float(c.mean()), "median": float(c.median()), "min": int(c.min()), "max": int(c.max()),
          "by_level": {k: {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()), "min": int(v.min()), "max": int(v.max())}
                       for k, v in d.groupby("evidence_level").coverage},
          "by_label": {("fraud" if k else "legitimate"): {"mean": float(v.mean()), "median": float(v.median())}
                       for k, v in d.groupby("is_fraud").coverage},
          "by_payee_status": {k: {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median())}
                              for k, v in d.groupby("payee_status").coverage}}
    return st


def main() -> int:
    frozen = json.loads((ROOT / "models/metrics_v2.json").read_text())
    model = joblib.load(ROOT / "models/risk_model_v2.pkl"); cols = frozen["features"]
    thr = float(frozen["validation_results"]["threshold"])
    (R / "end_to_end").mkdir(parents=True, exist_ok=True)
    summaries, fam_rows, missing_checks = {}, [], {}
    for seed in (42, 43, 44):
        res, fams = run_seed(seed, model, cols, thr)
        res.to_csv(R / f"end_to_end/e2e_v2_rows_seed{seed}.csv.gz", index=False, compression="gzip")
        summaries[seed] = summarise(res, seed)
        # family behaviour in the last 25%
        n = len(res); d = res.iloc[int(n * 0.75):]; y = d.is_fraud.to_numpy()
        for fam in fams:
            if fam == "ml_classifier":
                av = d.v2_family_available.to_numpy().astype(bool); sc = d.v2_family_score.fillna(0).to_numpy()
            else:
                av = d[f"avail__{fam}"].to_numpy().astype(bool); sc = d[f"score__{fam}"].fillna(0).to_numpy()
            act = av & (sc > 0)
            lead = np.array([fam in l.split("|") for l in d.leading_families])
            fam_rows.append({"seed": seed, "family": fam, "available_rate": av.mean(), "active_rate_fraud": act[y == 1].mean(),
                             "active_rate_legit": act[y == 0].mean(),
                             "auc_within_available": (roc_auc_score(y[av], sc[av]) if av.any() and 0 < y[av].sum() < av.sum()
                                                      and np.unique(sc[av]).size > 1 else np.nan),
                             "leading_rate_fraud": lead[y == 1].mean()})
        # missing evidence is neither safe nor fraud: risk of payments where a family was unavailable
        mc = {}
        for fam in fams:
            a = (res.v2_family_available if fam == "ml_classifier" else res[f"avail__{fam}"]).to_numpy().astype(bool)
            if (~a).any():
                mc[fam] = {"unavailable_rows": int((~a).sum()),
                           "risk_median_when_unavailable": float(np.median(res.risk[~a])),
                           "risk_median_when_available": float(np.median(res.risk[a])),
                           "share_R0_when_unavailable": float((res.risk[~a] == 0).mean()),
                           "share_BLOCK_when_unavailable": float((res.verdict[~a] == "BLOCK").mean()),
                           "fraud_rate_when_unavailable": float(res.is_fraud[~a].mean())}
        missing_checks[seed] = mc
        print(f"[seed {seed}] done: last25 risk PR-AUC {summaries[seed]['last25pct']['risk_pr_auc']:.4f} "
              f"legit BLOCK {summaries[seed]['last25pct']['block']['fpr']:.4f}", flush=True)
    fam_df = pd.DataFrame(fam_rows); fam_df.to_csv(R / "end_to_end/family_behaviour_last25.csv", index=False)
    json.dump({"what": __doc__.split("\n")[0], "threshold_v2": thr, "per_seed": summaries,
               "missing_evidence_checks": missing_checks}, open(R / "end_to_end/e2e_v2_summary.json", "w"), indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
