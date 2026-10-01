"""Combine the per-seed replay experiments (Experiments 3 and 4) into the canonical files.

The replay stream is regenerated per seed; each seed is one replicate. Nothing
is pooled across seeds: every statistic is computed per seed, then summarised
(mean, sd, min, max) over the seed-level values.

Writes
  results/end_to_end/e2e_results.csv, e2e_summary.json, evidence_family_metrics.csv, e2e_audit_sample.jsonl
  results/graph_replay/replay_metrics.json, replay_validation.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import RESULTS, environment, source_state, write_json  # noqa: E402

E = RESULTS / "end_to_end"
G = RESULTS / "graph_replay"


def desc(v):
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if not len(v):
        return None
    return {"n_seeds": int(len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "values": [float(x) for x in v]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    seeds = args.seeds

    # ── Experiment 3 ─────────────────────────────────────────────────────
    res = pd.concat([pd.read_csv(E / f"e2e_results_seed{s}.csv").assign(seed=s) for s in seeds])
    res.to_csv(E / "e2e_results.csv", index=False)
    fam = pd.concat([pd.read_csv(E / f"evidence_family_metrics_seed{s}.csv").assign(seed=s) for s in seeds])
    fam.to_csv(E / "evidence_family_metrics.csv", index=False)
    with open(E / "e2e_audit_sample.jsonl", "w") as out:
        for s in seeds:
            for line in open(E / f"e2e_audit_sample_seed{s}.jsonl"):
                r = json.loads(line); r["seed"] = s
                out.write(json.dumps(r, default=str) + "\n")
    per = {s: json.loads((E / f"e2e_summary_seed{s}.json").read_text()) for s in seeds}

    def pick(slice_name, key, sub=None):
        vals = []
        for s in seeds:
            sl = next((x for x in per[s]["risk_and_verdicts"] if x["slice"] == slice_name), None)
            if sl is None:
                vals.append(None); continue
            v = sl[key] if sub is None else sl[key][sub]
            vals.append(v)
        return desc(vals)

    across = {}
    for sl in ("full_stream", "test_last25pct"):
        across[sl] = {
            "risk_pr_auc": pick(sl, "risk_pr_auc"), "risk_roc_auc": pick(sl, "risk_roc_auc"),
            "ml_only_pr_auc_where_available": pick(sl, "ml_only_pr_auc_where_available"),
            "ml_only_roc_auc_where_available": pick(sl, "ml_only_roc_auc_where_available"),
            "risk_pr_auc_same_rows": pick(sl, "risk_pr_auc_same_rows"),
            "intervention_rate_all": pick(sl, "intervention_rate_all"),
            **{f"{op}_{m}": pick(sl, op, m) for op in ("warn_plus", "step_up_plus", "block")
               for m in ("recall", "fpr", "precision", "flag_rate")},
        }
    fam_across = {}
    for sl in ("full_stream", "test_last25pct"):
        f = fam[fam.slice == sl]
        fam_across[sl] = {fm: {c: desc(g[c].tolist()) for c in
                               ("available_rate", "active_rate_fraud", "active_rate_legit", "p_fraud_given_active",
                                "auc_within_available", "leading_rate_fraud", "loo_verdict_down_fraud",
                                "loo_verdict_down_legit", "loo_mean_drisk_fraud", "loo_mean_drisk_legit")}
                          for fm, g in f.groupby("family")}
    write_json(E / "e2e_summary.json", {
        "what": "Experiment 3 across replay seeds; per-seed detail in e2e_summary_seed<S>.json.",
        "seeds": seeds,
        "consistency_audit": {s: per[s]["consistency_audit"] for s in seeds},
        "across_seeds": across, "families_across_seeds": fam_across,
        "per_seed_rows": {s: {"rows": per[s]["rows"], "fraud": per[s]["fraud"]} for s in seeds},
        "corroboration": {s: per[s]["corroboration"] for s in seeds},
        "risk_vs_coverage": {s: per[s]["risk_vs_coverage"] for s in seeds},
        "verdict_share_by_group_test_last25pct": {s: per[s]["verdict_share_by_group_test_last25pct"] for s in seeds},
        "rows_by_group_test_last25pct": {s: per[s]["rows_by_group_test_last25pct"] for s in seeds},
        "environment": environment(), "source": source_state(),
    })

    # ── Experiment 4 ─────────────────────────────────────────────────────
    mets = {s: json.loads((G / f"replay_metrics_seed{s}.json").read_text()) for s in seeds
            if (G / f"replay_metrics_seed{s}.json").exists()}
    variants = sorted({v for m in mets.values() for v in m["results"]})
    summ = {v: {k: desc([mets[s]["results"][v][k] for s in mets]) for k in
                ("pr_auc", "roc_auc", "recall", "fpr", "precision", "f1", "brier",
                 "recall_social_engineering", "recall_mule_collection", "recall_account_takeover",
                 "recall_card_testing")} for v in variants}
    runs = {s: json.loads((G / f"replay_run_seed{s}.json").read_text()) for s in seeds}
    write_json(G / "replay_metrics.json", {
        "what": "Experiment 4 predictive comparison across replay seeds (chronological 60/15/25 split per seed).",
        "seeds": list(mets), "variants": summ,
        "per_seed": {s: {"split": m["split"], "results": m["results"], "graph_signal": m["graph_signal"]}
                     for s, m in mets.items()},
        "replay_design": runs[seeds[0]]["design"],
        "replay_seconds": {s: runs[s]["seconds"] for s in seeds},
    })
    vals = {s: json.loads((G / f"replay_validation_seed{s}.json").read_text()) for s in seeds
            if (G / f"replay_validation_seed{s}.json").exists()}
    tests = {}
    for s, v in vals.items():
        for k, t in v["tests"].items():
            a = tests.setdefault(k, {"checked": 0, "violations": 0, "seeds": []})
            a["checked"] += t["checked"]; a["violations"] += t["violations"]; a["seeds"].append(s)
    for t in tests.values():
        t["pass"] = t["violations"] == 0
    write_json(G / "replay_validation.json", {
        "what": "Replay validation tests 1-10 across seeds; per-seed detail in replay_validation_seed<S>.json.",
        "order": "assess at t from state < t, then record_payment (confirm) - never the reverse",
        "seeds": list(vals), "tests": tests, "all_pass": all(t["pass"] for t in tests.values()),
        "power_probe_9b": {s: v["test9b_power_probe"] for s, v in vals.items()},
    })
    print(json.dumps({k: {kk: (vv["mean"] if isinstance(vv, dict) and vv else vv) for kk, vv in v.items()}
                      for k, v in across.items()}, indent=1))
    print(json.dumps(tests, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
