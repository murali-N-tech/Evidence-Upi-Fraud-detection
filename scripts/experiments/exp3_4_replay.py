"""Experiments 3 and 4: chronological replay through the RUNNING assessment path.

For every payment of the replay stream (backend/ml/replay_corpus.py), in
timestamp order:

    1. set the application clock to the payment time t;
    2. load the pre-payment state: the payer's own prior payments (profile via
       statement_parser.generate_behavior_profile, the function the import
       pipeline uses; velocity via velocity.prepayment_velocity), and the
       shared store (ledger, reputation aggregates, incremental graph) which
       at this point contains ONLY payments with time < t;
    3. call payee_check.check_payee - the production assessment function the
       /payee/check and /payee/confirm endpoints call - with the payload,
       amount, stated purpose and message;
    4. from the SAME pre-payment state, build the V1 feature vector
       (features.build_vector, as payee_check._ml_evidence does) and the V2
       feature vector (payee_history.build_v2_vector with graph_cache.graph_view,
       the path V2 would use if served);
    5. record the result;
    6. ONLY THEN record the payment as confirmed (payee_reputation.record_payment,
       which updates the ledger, reputation aggregates and graph in one
       transaction - the /payee/confirm observation path).

A shadow model of the store (plain Python) is maintained alongside and every
graph/history reading is cross-checked against it at the moment of reading.

Clock injection: the running code reads "now" through
backend.app.core.clock.now_app (payee ages are now - first_seen). Replaying a
historical stream therefore requires now_app to return the replayed t; this is
the only runtime substitution made, and it is a clock, not an evaluator.

    python scripts/experiments/exp3_4_replay.py --seed 42 --n 60000
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import re

import numpy as np
import pandas as pd

from common import RESULTS, environment, sha256_frame, source_state, write_json  # noqa: E402

# Performance only: the message/intent analysers compile more distinct regex
# patterns per call than the stdlib cache (512) holds, so under a 60k-payment
# loop they recompile on every call. A larger cache changes no result.
re._MAXCACHE = 8192
# Run with OMP_NUM_THREADS=1: the live classifier scores one row at a time, and
# OpenMP thread start-up dominates single-row prediction under CPU contention.
# Thread count changes timing only, never a prediction.

import backend.app.core.clock as clock                                     # noqa: E402
import backend.app.services.profile_store as store                         # noqa: E402

OUT_R = RESULTS / "graph_replay"
OUT_E = RESULTS / "end_to_end"
CLOCK = [None]


def patched_now():
    return CLOCK[0]


class Shadow:
    """Independent re-computation of what the store should hold before t."""

    def __init__(self):
        self.pair = defaultdict(int)             # (payer, payee) -> payments
        self.pair_first = {}
        self.payee_payers = defaultdict(dict)    # payee -> {payer: payments}
        self.payee_first_arrivals = defaultdict(list)
        self.payee_times = defaultdict(list)
        self.parent = {}
        self.size = {}
        self.last_time = None

    def find(self, x):
        self.parent.setdefault(x, x); self.size.setdefault(x, 1)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra; self.size[ra] += self.size[rb]

    def component(self, node):
        if node not in self.parent:
            return None
        return self.size[self.find(node)]

    def add(self, payer, payee, when):
        if payer not in self.payee_payers[payee]:
            self.payee_first_arrivals[payee].append(when)
        self.payee_payers[payee][payer] = self.payee_payers[payee].get(payer, 0) + 1
        self.payee_times[payee].append(when)
        self.pair[(payer, payee)] += 1
        self.pair_first.setdefault((payer, payee), when)
        self.union("u:" + payer, "v:" + payee)
        self.last_time = when


def _days(a, b):
    return (b - a).total_seconds() / 86400.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--limit", type=int, default=None, help="replay only the first N rows (timing)")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    from backend.ml.replay_corpus import REPLAY_DESIGN, generate_replay

    OUT_R.mkdir(parents=True, exist_ok=True); OUT_E.mkdir(parents=True, exist_ok=True)
    df = generate_replay(seed=args.seed, n_transactions=args.n)
    if args.limit:
        df = df.iloc[:args.limit].copy()
    tag = f"seed{args.seed}"
    df.to_csv(OUT_R / f"replay_transactions_{tag}.csv", index=False)

    # A private store: the replay never touches data/behavior_profiles.db.
    db = Path(args.db) if args.db else Path(tempfile.mkdtemp(prefix="replay-")) / "replay.db"
    if db.exists():
        db.unlink()
    store.DB_PATH = db
    store._SCHEMA_DONE.clear()
    clock.now_app = patched_now

    from backend.app.services.payee_reputation import (assess_payee, connect, payee_key,
                                                       record_payment)
    from backend.app.services.payee_check import check_payee
    from backend.app.services.payee_history import build_v2_vector, payee_history
    from backend.app.services.graph_cache import (MIN_FAN_IN, MIN_SHARED_PAYERS, ONE_SHOT_RATIO,
                                                  OVERLAP_RATIO, BURST_WINDOW_DAYS, graph_view)
    from backend.app.services.statement_parser import generate_behavior_profile
    from backend.app.services.velocity import prepayment_velocity
    from backend.app.services.upi_qr import parse_upi_target
    from backend.ml.features import build_vector
    from backend.ml.dataset import FEATURES as FEATURES_V1
    from backend.ml.dataset_v2 import FEATURES_V2

    conn = connect()
    shadow = Shadow()
    prior = defaultdict(list)          # payer -> list of statement-like rows (strictly earlier)
    checks = defaultdict(lambda: {"checked": 0, "violations": 0, "examples": []})
    evid_rows = []
    audit = open(OUT_E / f"e2e_audit_full_{tag}.jsonl", "w")

    def check(name, ok, detail=None):
        c = checks[name]; c["checked"] += 1
        if not ok:
            c["violations"] += 1
            if len(c["examples"]) < 5:
                c["examples"].append(detail)

    started = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        t: datetime = row.timestamp.to_pydatetime()
        CLOCK[0] = t
        when = t.isoformat(timespec="microseconds")
        payer = row.payer_id
        key = payee_key(parse_upi_target(row.payload).payee_vpa, None)

        # ── 10. pre-payment boundary: the store holds nothing at or after t ──
        latest = conn.execute("SELECT MAX(at) FROM payment_observations").fetchone()[0]
        check("10_store_contains_only_earlier_payments", latest is None or latest < when,
              {"txn": row.txn_id, "latest": latest, "t": when})

        hist_rows = prior[payer]
        profile = generate_behavior_profile(hist_rows) if hist_rows else None
        history = pd.DataFrame(hist_rows) if hist_rows else None

        result = check_payee(row.payload, payer_id=payer, amount=float(row.amount), conn=conn,
                             intent=row.intent if isinstance(row.intent, str) else None,
                             message=row.message if isinstance(row.message, str) else None,
                             profile=profile, history=history, timestamp=when)

        # ── the same pre-payment state, as model features ─────────────────
        rep = assess_payee(key, conn=conn)
        repd = rep.as_dict()
        ph = payee_history(payer, key, before=when, conn=conn)
        vel = prepayment_velocity(history, when)
        v1 = build_vector(amount=float(row.amount), timestamp=when, profile=profile,
                          reputation=repd, payer_seen_payee_before=ph.paid_before, **vel)
        gv = graph_view(conn, key)
        v2 = build_v2_vector(ph, amount=float(row.amount), timestamp=when, profile=profile,
                             reputation=repd,
                             message=row.message if isinstance(row.message, str) else None,
                             intent=row.intent if isinstance(row.intent, str) else None,
                             payee_brand=None, payee_is_phone_address=bool(row.payee_is_phone),
                             graph_view=gv, **vel)

        # ── shadow cross-checks at the moment of reading ──────────────────
        prior_pair = shadow.pair.get((payer, key), 0)
        check("1_first_pair_is_new", (prior_pair > 0) or (not ph.paid_before and
              result["payee_status"] in ("NEW_TO_USER_BUT_OBSERVED", "UNKNOWN")),
              {"txn": row.txn_id, "status": result["payee_status"]})
        check("2_prior_pair_count_matches_earlier_rows", ph.payment_count == prior_pair,
              {"txn": row.txn_id, "store": ph.payment_count, "shadow": prior_pair})
        s_payers = shadow.payee_payers.get(key, {})
        check("3_distinct_payers_only_prior", rep.distinct_payers == len(s_payers),
              {"txn": row.txn_id, "store": rep.distinct_payers, "shadow": len(s_payers)})
        if s_payers:
            arr = shadow.payee_first_arrivals[key]
            span = _days(min(arr), max(arr))
            check("4_arrival_window_only_prior",
                  gv.available and gv.arrival_span_days is not None and abs(gv.arrival_span_days - span) < 1e-3,
                  {"txn": row.txn_id, "store": gv.arrival_span_days, "shadow": span})
            check("5_component_is_pre_payment",
                  gv.component_size == shadow.component("v:" + key),
                  {"txn": row.txn_id, "store": gv.component_size, "shadow": shadow.component("v:" + key)})
            fired = bool(gv.position_findings)
            rep_ratio = sum(1 for n in s_payers.values() if n > 1) / max(len(s_payers), 1)
            gate = (gv.shared_payer_peak is not None and gv.shared_payer_peak >= MIN_SHARED_PAYERS
                    and gv.overlap_ratio is not None and gv.overlap_ratio >= OVERLAP_RATIO
                    and rep_ratio < ONE_SHOT_RATIO and len(s_payers) >= MIN_FAN_IN)
            check("6_position_finding_only_when_gates_hold", fired == gate,
                  {"txn": row.txn_id, "fired": fired, "gate": gate})
        else:
            check("7_unobserved_payee_graph_unavailable_not_zero",
                  (not gv.available) and np.isnan(v2.values.get("arrival_span_days", np.nan))
                  and v2.values.get("graph_available") == 0.0
                  and not result["evidence_available"].get("graph", True),
                  {"txn": row.txn_id})

        ev = {e["family"]: e for e in result["evidence"]}
        rec = {
            "txn_id": row.txn_id, "timestamp": when, "payer_id": payer, "payee_key": key,
            "is_fraud": int(row.is_fraud), "scenario": row.scenario,
            "risk": result["risk_score"], "coverage": result["confidence_score"],
            "evidence_level": result["evidence_level"], "verdict": result["verdict"],
            "bonus": result["agreement"]["bonus"],
            "n_facts": len(result["agreement"]["facts"]),
            "facts": "|".join(result["agreement"]["facts"]),
            "observed_by": json.dumps(result["agreement"]["observed_by"], sort_keys=True),
            "payee_status": result["payee_status"],
            "ml_probability": (result["ml"] or {}).get("probability"),
            "graph_position_fired": bool(gv.position_findings),
            "graph_shape_findings": len(gv.shape_findings),
        }
        for fam, e in ev.items():
            rec[f"avail__{fam}"] = int(bool(e["available"]))
            rec[f"score__{fam}"] = e["score"] if e["available"] else None
            rec[f"facts__{fam}"] = "|".join(e.get("facts") or [])
        rec["finding_codes"] = "|".join(sorted({f["code"] for f in result["findings"]}))
        for k, v in v1.values.items():
            rec[f"v1__{k}"] = v
        for k, v in v2.values.items():
            rec[f"v2__{k}"] = v
        # replay-state quantities the simulator's graph channel is conditioned on
        rec["state_payee_observed"] = int(bool(rep.known))
        rec["state_distinct_payers"] = rep.distinct_payers
        rec["state_age_days"] = rep.age_days
        rec["state_repeat_ratio"] = (rep.repeat_payers / rep.distinct_payers) if rep.distinct_payers else None
        evid_rows.append(rec)

        if row.is_fraud or (i % 97 == 0):
            audit.write(json.dumps({
                "txn_id": row.txn_id, "label": int(row.is_fraud), "scenario": row.scenario,
                "risk": result["risk_score"], "coverage": result["confidence_score"],
                "evidence_level": result["evidence_level"], "verdict": result["verdict"],
                "active_families": [f for f, e in ev.items() if e["available"] and e["score"]],
                "available": {f: bool(e["available"]) for f, e in ev.items()},
                "family_scores": {f: e["score"] for f, e in ev.items() if e["available"]},
                "corroboration": result["agreement"],
                "findings": result["findings"],
                "missing_evidence": result["missing_evidence"],
            }, default=str) + "\n")

        # ── 6. only now: the payer confirms and the payment is observed ────
        record_payment(key, payer_id=payer, amount=float(row.amount), at=when,
                       display_name=row.payee_display if isinstance(row.payee_display, str) else None,
                       conn=conn, provenance="SYSTEM_CONFIRMED", observation_key=row.txn_id)
        shadow.add(payer, key, t)
        prior[payer].append({"timestamp": when, "amount": float(row.amount),
                             "merchant": (row.payee_display if isinstance(row.payee_display, str)
                                          else key), "upi_id": key, "status": "SUCCESS",
                             "txn_type": "DEBIT", "time_known": 1})
        if (i + 1) % 2000 == 0:
            el = time.time() - started
            print(f"  {i+1:,}/{len(df):,} replayed  {el/60:.1f} min  "
                  f"({el/(i+1)*1000:.0f} ms/txn)", flush=True)
        if (i + 1) % 10000 == 0:
            pd.DataFrame(evid_rows).to_csv(OUT_R / f"replay_evidence_{tag}.partial.csv", index=False)

    audit.close()
    ev_df = pd.DataFrame(evid_rows)
    ev_df.to_csv(OUT_R / f"replay_evidence_{tag}.csv", index=False)
    write_json(OUT_R / f"replay_run_{tag}.json", {
        "seed": args.seed, "n_transactions": int(len(df)), "fraud": int(df.is_fraud.sum()),
        "db": str(db), "seconds": round(time.time() - started, 1),
        "design": REPLAY_DESIGN, "online_checks": checks,
        "corpus_sha256": sha256_frame(df.drop(columns=["timestamp"]).astype(str)),
        "evidence_sha256": sha256_frame(ev_df.astype(str)),
        "clock_injection": "backend.app.core.clock.now_app -> replay time t",
        "environment": environment(), "source": source_state(),
    })
    print(json.dumps({k: {"checked": v["checked"], "violations": v["violations"]}
                      for k, v in checks.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
