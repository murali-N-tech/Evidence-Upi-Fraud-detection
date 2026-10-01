"""Experiment 4 - replay validation tests 8 and 9 (offline, after the replay).

Tests 1-7 and 10 run ONLINE inside exp3_4_replay.py at the moment each state is
read. This script adds the two checks that need a second, independent run:

  8  No future leakage (independent rebuild). A FRESH store is built by a
     separate process from the transaction file alone, inserting payments in
     time order. At each sampled payment the running assessment
     (payee_check.check_payee) and the V2 graph view are recomputed from that
     fresh store, which by construction holds only earlier payments, and must
     equal what the replay recorded - risk, coverage, level, verdict, bonus,
     every family's availability/score/facts, finding codes and the seven V2
     graph features.

  9  Historical findings are not rewritten by later payments.
     9a ledger immutability: every observation row written before the midpoint
        of the stream is byte-identical (payer, payee, amount, time, provenance,
        key, in_aggregate) in the FINAL replay store, i.e. later inserts did
        not modify earlier records;
     9b the recorded pre-payment evidence is a snapshot: re-assessing the
        sampled payments against the FINAL store (which contains their future)
        is reported, to show that the boundary is material - if the final
        store gave the same answers, test 8 would have no power.

    python scripts/experiments/exp4_rebuild_validation.py --seed 42 --sample 60
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, environment, source_state, write_json  # noqa: E402

import backend.app.core.clock as clock                            # noqa: E402
import backend.app.services.profile_store as store                # noqa: E402

re._MAXCACHE = 8192   # performance only, as in exp3_4_replay.py
OUT = RESULTS / "graph_replay"
CLOCK = [None]
OBS_COLS = "payer_id, vpa, amount, at, provenance, observation_key, in_aggregate"


def patched_now():
    return CLOCK[0]


def pick_sample(ev: pd.DataFrame, k: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed + 777)
    idx = set()
    strata = [
        ev.index[ev.is_fraud == 1],
        ev.index[ev.graph_position_fired.astype(bool)],
        ev.index[(ev.is_fraud == 0) & (ev.payee_status == "PAID_OFTEN")],
        ev.index[(ev.is_fraud == 0) & (ev.payee_status == "UNKNOWN")],
        ev.index[ev.verdict == "BLOCK"],
        ev.index,
    ]
    per = max(1, k // len(strata))
    for s in strata:
        s = [i for i in s if i not in idx]
        if s:
            idx.update(rng.choice(s, size=min(per, len(s)), replace=False).tolist())
    return sorted(idx)


def compare(rec: dict, result: dict, gv, families, fn_v2) -> list[str]:
    diffs = []
    pairs = [("risk", result["risk_score"]), ("coverage", result["confidence_score"]),
             ("evidence_level", result["evidence_level"]), ("verdict", result["verdict"]),
             ("bonus", result["agreement"]["bonus"]),
             ("facts", "|".join(result["agreement"]["facts"])),
             ("payee_status", result["payee_status"]),
             ("finding_codes", "|".join(sorted({f["code"] for f in result["findings"]})))]
    for k, v in pairs:
        a = rec[k] if not (isinstance(rec[k], float) and np.isnan(rec[k])) else ""
        if str(a) != str(v):
            diffs.append(f"{k}: recorded={a!r} rebuilt={v!r}")
    ev = {e["family"]: e for e in result["evidence"]}
    for fam in families:
        e = ev[fam]
        if int(bool(e["available"])) != int(rec[f"avail__{fam}"]):
            diffs.append(f"avail__{fam}")
        s_rec = rec[f"score__{fam}"]; s_new = e["score"] if e["available"] else None
        if not ((s_new is None and (s_rec is None or np.isnan(s_rec))) or (s_new is not None and float(s_rec) == float(s_new))):
            diffs.append(f"score__{fam}: {s_rec} vs {s_new}")
        f_rec = rec[f"facts__{fam}"] if isinstance(rec[f"facts__{fam}"], str) else ""
        if f_rec != "|".join(e.get("facts") or []):
            diffs.append(f"facts__{fam}")
    for c, v in fn_v2().items():
        r = rec.get(f"v2__{c}")
        both_nan = (v is None or (isinstance(v, float) and np.isnan(v))) and (r is None or (isinstance(r, float) and np.isnan(r)))
        if not both_nan and not (r is not None and v is not None and abs(float(r) - float(v)) < 1e-9):
            diffs.append(f"v2__{c}: {r} vs {v}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--final-db", required=True, help="the store left by exp3_4_replay.py")
    args = ap.parse_args()
    tag = f"seed{args.seed}"

    tx = pd.read_csv(OUT / f"replay_transactions_{tag}.csv", parse_dates=["timestamp"])
    ev = pd.read_csv(OUT / f"replay_evidence_{tag}.csv", low_memory=False)
    assert len(tx) == len(ev) and (tx.txn_id.to_numpy() == ev.txn_id.to_numpy()).all()
    families = sorted(c.split("__", 1)[1] for c in ev.columns if c.startswith("avail__"))
    sample = pick_sample(ev, args.sample, args.seed)
    sample_set = set(sample)
    mid = len(tx) // 2

    fresh = Path(tempfile.mkdtemp(prefix="rebuild-")) / "fresh.db"
    store.DB_PATH = fresh
    store._SCHEMA_DONE.clear()
    clock.now_app = patched_now

    from backend.app.services.payee_reputation import connect, payee_key, record_payment, assess_payee
    from backend.app.services.payee_check import check_payee
    from backend.app.services.payee_history import build_v2_vector, payee_history
    from backend.app.services.graph_cache import graph_view
    from backend.app.services.statement_parser import generate_behavior_profile
    from backend.app.services.velocity import prepayment_velocity
    from backend.app.services.upi_qr import parse_upi_target

    conn = connect()
    prior: dict[str, list] = {}
    results8, results9b = [], []
    mid_snapshot = None
    started = time.time()

    def assess(i, row, c):
        t = row.timestamp.to_pydatetime(); CLOCK[0] = t
        when = t.isoformat(timespec="microseconds")
        hist = prior.get(row.payer_id, [])
        profile = generate_behavior_profile(hist) if hist else None
        history = pd.DataFrame(hist) if hist else None
        msg = row.message if isinstance(row.message, str) else None
        intent = row.intent if isinstance(row.intent, str) else None
        res = check_payee(row.payload, payer_id=row.payer_id, amount=float(row.amount), conn=c,
                          intent=intent, message=msg, profile=profile, history=history, timestamp=when)
        key = payee_key(parse_upi_target(row.payload).payee_vpa, None)
        gv = graph_view(c, key)

        def v2():
            rep = assess_payee(key, conn=c).as_dict()
            ph = payee_history(row.payer_id, key, before=when, conn=c)
            vel = prepayment_velocity(history, when)
            vec = build_v2_vector(ph, amount=float(row.amount), timestamp=when, profile=profile, reputation=rep,
                                  message=msg, intent=intent, payee_brand=None,
                                  payee_is_phone_address=bool(row.payee_is_phone), graph_view=gv, **vel)
            return {k: v for k, v in vec.values.items()
                    if k in ("graph_available", "arrival_span_days", "component_size", "shared_payer_peak",
                             "overlap_ratio", "position_flag", "one_shot_ratio") or k.startswith("graph")
                    or k in ev.columns.str.replace("v2__", "")}
        return compare(ev.loc[i].to_dict(), res, gv, families, v2)

    final = sqlite3.connect(f"file:{args.final_db}?mode=ro", uri=True)
    for i, row in enumerate(tx.itertuples(index=False)):
        if i in sample_set:
            d = assess(i, row, conn)
            results8.append({"i": i, "txn_id": row.txn_id, "is_fraud": int(row.is_fraud),
                             "scenario": row.scenario, "identical": not d, "diffs": d[:10]})
        if i == mid:
            mid_snapshot = conn.execute(f"SELECT {OBS_COLS} FROM payment_observations ORDER BY id").fetchall()
        when = row.timestamp.to_pydatetime().isoformat(timespec="microseconds")
        CLOCK[0] = row.timestamp.to_pydatetime()
        key = payee_key(parse_upi_target(row.payload).payee_vpa, None)
        record_payment(key, payer_id=row.payer_id, amount=float(row.amount), at=when,
                       display_name=row.payee_display if isinstance(row.payee_display, str) else None,
                       conn=conn, provenance="SYSTEM_CONFIRMED", observation_key=row.txn_id)
        prior.setdefault(row.payer_id, []).append({
            "timestamp": when, "amount": float(row.amount),
            "merchant": row.payee_display if isinstance(row.payee_display, str) else key,
            "upi_id": key, "status": "SUCCESS", "txn_type": "DEBIT", "time_known": 1})
        if (i + 1) % 10000 == 0:
            print(f"  rebuilt {i+1:,}/{len(tx):,}  {(time.time()-started)/60:.1f} min", flush=True)

    # 9a ledger immutability against the FINAL replay store
    final_rows = final.execute(f"SELECT {OBS_COLS} FROM payment_observations ORDER BY id").fetchall()
    n_mid = len(mid_snapshot)
    immut = {"rows_before_midpoint": n_mid,
             "identical_in_final_store": int(sum(1 for a, b in zip(mid_snapshot, final_rows[:n_mid]) if tuple(a) == tuple(b))),
             "final_store_rows": len(final_rows),
             "fresh_store_rows_at_end": conn.execute("SELECT COUNT(*) FROM payment_observations").fetchone()[0]}
    immut["pass"] = immut["identical_in_final_store"] == n_mid and immut["final_store_rows"] == len(tx)

    # 9b power probe: same sampled payments assessed against the FINAL store (contains the future)
    prior_all = {}
    for r in tx.itertuples(index=False):
        prior_all.setdefault(r.payer_id, []).append(r)
    import shutil
    probe_db = fresh.parent / "final_copy_for_probe.db"
    shutil.copyfile(args.final_db, probe_db)          # never write into the replay's own store
    store.DB_PATH = probe_db; store._SCHEMA_DONE.clear()
    fconn = connect()
    # the payer's own history stays strictly prior; only the SHARED store carries the future
    rebuilt_prior = {}
    for i, row in enumerate(tx.itertuples(index=False)):
        if i in sample_set:
            prior.clear(); prior.update(rebuilt_prior)
            d = assess(i, row, fconn)
            results9b.append({"i": i, "txn_id": row.txn_id, "differs_when_future_visible": bool(d), "n_diffs": len(d)})
        when = row.timestamp.to_pydatetime().isoformat(timespec="microseconds")
        key = payee_key(parse_upi_target(row.payload).payee_vpa, None)
        rebuilt_prior.setdefault(row.payer_id, []).append({
            "timestamp": when, "amount": float(row.amount),
            "merchant": row.payee_display if isinstance(row.payee_display, str) else key,
            "upi_id": key, "status": "SUCCESS", "txn_type": "DEBIT", "time_known": 1})

    n8 = len(results8); ok8 = sum(r["identical"] for r in results8)
    n9 = len(results9b); diff9 = sum(r["differs_when_future_visible"] for r in results9b)
    run = json.loads((OUT / f"replay_run_{tag}.json").read_text())
    online = {k: {"checked": v["checked"], "violations": v["violations"], "pass": v["violations"] == 0}
              for k, v in run["online_checks"].items()}
    tests = dict(online)
    tests["8_independent_rebuild_prior_only_identical"] = {"checked": n8, "violations": n8 - ok8, "pass": ok8 == n8}
    tests["9a_earlier_observations_not_rewritten"] = {"checked": immut["rows_before_midpoint"],
                                                    "violations": immut["rows_before_midpoint"] - immut["identical_in_final_store"],
                                                    "pass": immut["pass"]}
    write_json(OUT / f"replay_validation_{tag}.json", {
        "what": "Chronological replay validation. Tests 1-7 and 10 were checked online at every payment "
                "against an independent shadow model; 8 and 9 by an independent rebuild.",
        "seed": args.seed, "n_transactions": int(len(tx)),
        "order": "assess at t from state < t, then record_payment (confirm) - never the reverse",
        "tests": tests,
        "all_pass": all(v["pass"] for v in tests.values()),
        "test8_detail": results8,
        "test9a_detail": immut,
        "test9b_power_probe": {"sampled": n9, "differs_when_future_visible": diff9,
                               "share": diff9 / max(n9, 1),
                               "reading": "share of sampled payments whose assessment would change if the "
                                          "shared store contained the future; shows the boundary is material"},
        "sample_design": "stratified: fraud, graph-position rows, PAID_OFTEN legit, UNKNOWN legit, BLOCK, any; "
                         f"rng seed {args.seed + 777}",
        "seconds": round(time.time() - started, 1),
        "environment": environment(), "source": source_state(),
    })
    print(json.dumps(tests, indent=1)); print("9b:", diff9, "/", n9)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
