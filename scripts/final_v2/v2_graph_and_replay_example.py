"""Graph statistics of the replayed payer-payee graph, and a concrete
before / assess / after example from the chronological replay.

Graph statistics are read from the stores left by the replay
(_work/replay_seed<S>.db, opened read-only) and cross-checked against an
independent union-find built from the transaction files.

The example re-executes one payment exactly as the replay did: a fresh store is
filled with every payment strictly before the chosen one (record_payment), the
graph/history state is printed, the payment is assessed with
payee_check.check_payee at its own time, the result is compared with what the
replay recorded, and only then is the payment recorded and the state printed
again.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import re
re._MAXCACHE = 8192

import backend.app.core.clock as clock                     # noqa: E402
import backend.app.services.profile_store as store           # noqa: E402
import style                                                 # noqa: E402

R = ROOT / "results" / "final_v2"
G = ROOT / "results" / "graph_replay"
CLOCK = [None]


def graph_stats(seed: int) -> dict:
    db = sqlite3.connect(f"file:{ROOT / '_work' / f'replay_seed{seed}.db'}?mode=ro", uri=True)
    edges = pd.read_sql("SELECT vpa, payer_id, payments, first_at, last_at FROM payee_payers", db)
    cache = pd.read_sql("SELECT * FROM payee_graph_cache", db)
    comp = pd.read_sql("SELECT node, parent, size FROM graph_components", db)
    rep = pd.read_sql("SELECT vpa, payment_count, first_seen, last_seen FROM payee_reputation", db)
    obs_n = db.execute("SELECT COUNT(*) FROM payment_observations").fetchone()[0]
    # independent union-find from the transaction file
    tx = pd.read_csv(G / f"replay_transactions_seed{seed}.csv", usecols=["payer_id", "payee_key"])
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in zip("u:" + tx.payer_id, "v:" + tx.payee_key):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    sizes = Counter(find(x) for x in list(parent))
    roots = comp[comp.node == comp.parent]
    payer_deg = edges.groupby("payer_id").size(); payee_deg = edges.groupby("vpa").size()
    q = lambda s: {"mean": float(s.mean()), "median": float(s.median()), "p95": float(s.quantile(0.95)), "max": int(s.max())}
    out = {
        "seed": seed, "payments_observed": int(obs_n), "payments_on_edges": int(edges.payments.sum()),
        "payer_nodes": int(edges.payer_id.nunique()), "payee_nodes": int(edges.vpa.nunique()),
        "nodes_total": int(edges.payer_id.nunique() + edges.vpa.nunique()), "edges_payer_payee_pairs": int(len(edges)),
        "repeat_relationships_edges_with_2plus_payments": int((edges.payments >= 2).sum()),
        "repeat_relationship_share": float((edges.payments >= 2).mean()),
        "single_payment_relationships": int((edges.payments == 1).sum()),
        "payer_out_degree_distinct_payees": q(payer_deg), "payee_in_degree_distinct_payers": q(payee_deg),
        "payments_per_edge": q(edges.payments),
        "components_in_store_union_find": int(len(roots)),
        "largest_component_store": int(roots["size"].max()), "component_sizes_store_top5": sorted(roots["size"].tolist(), reverse=True)[:5],
        "components_independent_rebuild": int(len(sizes)), "largest_component_independent": int(max(sizes.values())),
        "union_find_nodes_store": int(len(comp)), "union_find_nodes_independent": int(len(parent)),
        "payee_cache_rows": int(len(cache)), "payee_reputation_rows": int(len(rep)),
        "cache_payer_count_matches_edges": bool((cache.set_index("vpa").payer_count.reindex(payee_deg.index).fillna(-1).astype(int)
                                                == payee_deg).all()),
        "payees_with_10plus_payers": int((payee_deg >= 10).sum()),
        "payees_repeat_ratio_below_0.1_with_10plus_payers": int(((cache.repeat_payer_count / cache.payer_count.clip(lower=1)) < 0.1)[cache.payer_count >= 10].sum()),
    }
    out["components_agree"] = (out["components_in_store_union_find"] == out["components_independent_rebuild"]
                               and out["largest_component_store"] == out["largest_component_independent"])
    ev = pd.read_csv(G / f"replay_evidence_seed{seed}.csv", usecols=["is_fraud", "scenario", "payee_status", "avail__graph",
                                                                     "graph_position_fired", "graph_shape_findings", "finding_codes"])
    codes = Counter(c for s in ev.finding_codes.fillna("") for c in str(s).split("|") if c)
    out["assessment_time"] = {
        "graph_evidence_available_rate": float(ev.avail__graph.mean()),
        "graph_available_rate_by_label": ev.groupby("is_fraud").avail__graph.mean().rename({0: "legitimate", 1: "fraud"}).to_dict(),
        "positional_finding_fired": {"fraud": int(ev.graph_position_fired[ev.is_fraud == 1].sum()),
                                     "legitimate": int(ev.graph_position_fired[ev.is_fraud == 0].sum())},
        "positional_by_scenario": ev[ev.graph_position_fired.astype(bool)].scenario.value_counts().to_dict(),
        "rows_with_graph_shape_findings": int((ev.graph_shape_findings > 0).sum()),
        "relationship_state_counts": ev.payee_status.value_counts().to_dict(),
        "relationship_state_by_label": {k: v.value_counts().to_dict() for k, v in ev.groupby("is_fraud").payee_status},
        "graph_finding_codes": {k: v for k, v in codes.items() if k.startswith("graph_")},
    }
    return out


def state(conn, payer, key):
    q = lambda sql, *a: conn.execute(sql, a).fetchone()
    edge = q("SELECT payments, first_at, last_at FROM payee_payers WHERE vpa=? AND payer_id=?", key, payer)
    cache = q("SELECT payer_count, repeat_payer_count, payments, arrival_first, arrival_last FROM payee_graph_cache WHERE vpa=?", key)
    repn = q("SELECT payment_count, first_seen, last_seen FROM payee_reputation WHERE vpa=?", key)
    from backend.app.services.graph_cache import graph_view
    gv = graph_view(conn, key)
    return {"pair_edge": None if edge is None else {"payments": edge[0], "first_at": edge[1], "last_at": edge[2]},
            "payee_aggregate": None if cache is None else {"distinct_payers": cache[0], "repeat_payers": cache[1], "payments": cache[2],
                                                        "arrival_first": cache[3], "arrival_last": cache[4]},
            "payee_reputation": None if repn is None else {"payment_count": repn[0], "first_seen": repn[1], "last_seen": repn[2]},
            "graph_view": {"available": gv.available, "component_size": gv.component_size, "arrival_span_days": gv.arrival_span_days,
                           "shared_payer_peak": gv.shared_payer_peak, "overlap_ratio": gv.overlap_ratio,
                           "position_findings": [getattr(f, "code", str(f)) for f in gv.position_findings],
                           "shape_findings": [getattr(f, "code", str(f)) for f in gv.shape_findings]},
            "observations_in_store": conn.execute("SELECT COUNT(*) FROM payment_observations").fetchone()[0]}


def example(seed: int, pick: str) -> dict:
    tx = pd.read_csv(G / f"replay_transactions_seed{seed}.csv", parse_dates=["timestamp"])
    ev = pd.read_csv(G / f"replay_evidence_seed{seed}.csv", low_memory=False)
    if pick == "repeat_merchant":
        cand = ev.index[(ev.payee_status == "PAID_OFTEN") & (ev.is_fraud == 0) & (ev.avail__graph == 1)]
    else:
        cand = ev.index[(ev.scenario == "mule_collection") & (ev.avail__graph == 1)]
    k = int(cand[len(cand) // 2]) if pick == "repeat_merchant" else int(cand[-1])
    k = min(k, 25_000) if pick == "repeat_merchant" and (ev.payee_status[:25000] == "PAID_OFTEN").any() else k
    if pick == "repeat_merchant":
        c2 = ev.index[(ev.payee_status == "PAID_OFTEN") & (ev.is_fraud == 0) & (ev.avail__graph == 1) & (ev.index < 25_000)]
        k = int(c2[-1]) if len(c2) else k
    db = Path(tempfile.mkdtemp(prefix="example-")) / "ex.db"
    store.DB_PATH = db; store._SCHEMA_DONE.clear()
    clock.now_app = lambda: CLOCK[0]
    from backend.app.services.payee_reputation import connect, payee_key, record_payment
    from backend.app.services.payee_check import check_payee
    from backend.app.services.statement_parser import generate_behavior_profile
    from backend.app.services.upi_qr import parse_upi_target
    conn = connect(); prior = {}
    for i, row in enumerate(tx.itertuples(index=False)):
        t = row.timestamp.to_pydatetime(); CLOCK[0] = t; when = t.isoformat(timespec="microseconds")
        key = payee_key(parse_upi_target(row.payload).payee_vpa, None)
        if i == k:
            before = state(conn, row.payer_id, key)
            hist = prior.get(row.payer_id, [])
            res = check_payee(row.payload, payer_id=row.payer_id, amount=float(row.amount), conn=conn,
                              intent=row.intent if isinstance(row.intent, str) else None,
                              message=row.message if isinstance(row.message, str) else None,
                              profile=generate_behavior_profile(hist) if hist else None,
                              history=pd.DataFrame(hist) if hist else None, timestamp=when)
            rec = ev.loc[k]
            same = (res["risk_score"] == rec.risk and res["verdict"] == rec.verdict and res["confidence_score"] == rec.coverage)
        record_payment(key, payer_id=row.payer_id, amount=float(row.amount), at=when,
                       display_name=row.payee_display if isinstance(row.payee_display, str) else None,
                       conn=conn, provenance="SYSTEM_CONFIRMED", observation_key=row.txn_id)
        prior.setdefault(row.payer_id, []).append({"timestamp": when, "amount": float(row.amount),
                                                  "merchant": row.payee_display if isinstance(row.payee_display, str) else key,
                                                  "upi_id": key, "status": "SUCCESS", "txn_type": "DEBIT", "time_known": 1})
        if i == k:
            after = state(conn, row.payer_id, key)
            break
    ev_fams = {e["family"]: {"available": e["available"], "score": e["score"], "facts": e.get("facts")} for e in res["evidence"]}
    return {"seed": seed, "kind": pick, "row_index": k, "txn_id": tx.txn_id[k], "time": str(tx.timestamp[k]),
            "label": int(tx.is_fraud[k]), "scenario": tx.scenario[k], "amount": float(tx.amount[k]),
            "payer": tx.payer_id[k], "payee": payee_key(parse_upi_target(tx.payload[k]).payee_vpa, None),
            "before": before,
            "assessment": {"risk": res["risk_score"], "coverage": res["confidence_score"], "evidence_level": res["evidence_level"],
                           "verdict": res["verdict"], "payee_status": res["payee_status"],
                           "agreement": res["agreement"], "families": ev_fams,
                           "findings": sorted({f["code"] for f in res["findings"]}),
                           "classifier_family_note": "the live prototype serves an earlier, non-reported model in this family; the V2 re-aggregation used in the report of this row is in e2e_v2_rows_seed42"},
            "identical_to_replay_record": bool(same), "after": after}


def main() -> int:
    (R / "graph").mkdir(parents=True, exist_ok=True); (R / "replay").mkdir(parents=True, exist_ok=True)
    stats = {s: graph_stats(s) for s in (42, 43, 44)}
    json.dump(stats, open(R / "graph/graph_statistics.json", "w"), indent=2, default=float)
    for s, v in stats.items():
        print(s, {k: v[k] for k in ("payer_nodes", "payee_nodes", "edges_payer_payee_pairs", "components_in_store_union_find",
                                   "largest_component_store", "components_agree", "cache_payer_count_matches_edges")}, flush=True)
    exs = [example(42, "repeat_merchant"), example(42, "mule")]
    json.dump(exs, open(R / "replay/replay_examples_seed42.json", "w"), indent=2, default=str)
    for e in exs:
        print(e["kind"], e["row_index"], e["identical_to_replay_record"], e["assessment"]["verdict"], flush=True)
    # figure: degree distributions + availability
    import matplotlib.pyplot as plt
    db = sqlite3.connect(f"file:{ROOT / '_work' / 'replay_seed42.db'}?mode=ro", uri=True)
    edges = pd.read_sql("SELECT vpa, payer_id, payments FROM payee_payers", db)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    ax = axes[0]
    d = edges.groupby("vpa").size()
    ax.hist(d, bins=np.logspace(0, np.log10(d.max() + 1), 30), color=style.BLUE)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Distinct payers per payee (in-degree)"); ax.set_ylabel("Payees")
    ax.set_title("Payee in-degree (seed 42, end of stream)")
    ax = axes[1]
    d2 = edges.groupby("payer_id").size()
    ax.hist(d2, bins=30, color=style.AQUA); ax.set_xlabel("Distinct payees per payer (out-degree)"); ax.set_ylabel("Payers")
    ax.set_title("Payer out-degree")
    ax = axes[2]
    a = stats[42]["assessment_time"]["relationship_state_counts"]
    order = ["UNKNOWN", "NEW_TO_USER_BUT_OBSERVED", "PAID_ONCE", "PAID_BEFORE", "PAID_OFTEN"]
    ax.barh([o.replace("_", " ").title() for o in order][::-1], [a.get(o, 0) for o in order][::-1], color=style.ORANGE)
    ax.set_xlabel("Assessed payments"); ax.set_title("Relationship state at assessment time")
    ax.grid(axis="y", visible=False)
    style.save(fig, R / "figures/fig14_graph_statistics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
