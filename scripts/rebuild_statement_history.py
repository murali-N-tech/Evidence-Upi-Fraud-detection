"""Rebuild statement-derived history so there is ONE authoritative history.

Rows imported before the canonical ingestion pipeline (no fingerprint) may
carry a wrong direction (every CREDIT read as DEBIT by the old CSV parser),
invented timestamps, raw statement text, and payee edges written by the old
bootstrap from those rows. This script separates them from the new data
without deleting anything it cannot recreate.

DRY RUN BY DEFAULT. Nothing is written unless --apply is given.

    # 1. See what would change (read-only):
    python scripts/rebuild_statement_history.py

    # 2. Apply (a timestamped backup of the whole database is written first):
    python scripts/rebuild_statement_history.py --apply --reimport-retained

    # Dev/demo database, full rebuild including the demo payee crowd:
    python scripts/rebuild_statement_history.py --apply --reimport-retained --reseed-demo

Stop the backend first: the rebuild takes the database's write lock.

What --apply does, in one transaction (steps 1-6) and then per statement:

 1. Backup      data/backups/<db>.<timestamp>.db via SQLite's backup API, plus
                a JSON manifest of row counts before and after.
 2. Quarantine  legacy statement rows (fingerprint IS NULL) are MOVED to
                statement_transactions_legacy - kept, never read by the app.
                --purge-raw-text clears raw_line there.
 3. Observations  STATEMENT_OBSERVED ledger rows are removed (they are
                re-derived in step 5). SYSTEM_CONFIRMED and OTHER_OBSERVED
                rows are kept.
 4. Stale payees  a payee with any edge from a statement user that has no
                ledger row behind it was built by the old bootstrap from the
                legacy rows. Its payee_reputation/payee_payers rows are
                cleared (report counts and the block flag are restored).
                Payees touched only by other edges (demo seeds, legacy
                confirms by users with no statements) are left exactly as
                they are. A payee mixing both kinds cannot be split exactly;
                the script refuses unless --drop-mixed-legacy is given.
 5. Replay      kept ledger rows on cleared payees are re-applied; canonical
                statement rows (fingerprinted) are re-recorded as
                STATEMENT_OBSERVED with their fingerprint as the key.
 6. Graph       payee_graph_cache / graph_components are rebuilt from the
                remaining edges (graph_cache.backfill).
 7. Re-import   --reimport-retained runs every retained file in
                data/uploaded_statements/ whose owner prefix is a known user
                through the canonical pipeline (dedupe makes this idempotent).
                --import FILE --user ID imports one more file.
 8. Profiles    rebuilt from each user's canonical rows. A user left with no
                canonical rows keeps the old profile unless
                --reset-orphan-profiles is given; they are listed either way.
 9. Demo        --reseed-demo regenerates the demo payee crowd (demo-* payees
                and demo_payer_/demo_crowd_ edges) if it was cleared.

Output is counts only - never statement content.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backend.app.services.profile_store as store  # noqa: E402

LEGACY_TABLE = "statement_transactions_legacy"
_NOT_COMPLETED = ("FAILED", "FAILURE", "DECLINED", "PENDING", "REVERSED", "CANCELLED", "CANCELED")
STATEMENT_USERS = "SELECT DISTINCT user_id FROM statement_transactions"


def _counts(conn: sqlite3.Connection) -> dict:
    def one(sql, *a):
        try:
            return conn.execute(sql, a).fetchone()[0]
        except sqlite3.OperationalError:
            return None

    return {
        "statement_rows": one("SELECT COUNT(*) FROM statement_transactions"),
        "legacy_statement_rows": one("SELECT COUNT(*) FROM statement_transactions WHERE fingerprint IS NULL"),
        "canonical_statement_rows": one("SELECT COUNT(*) FROM statement_transactions WHERE fingerprint IS NOT NULL"),
        "rows_with_raw_text": one("SELECT COUNT(*) FROM statement_transactions WHERE raw_line IS NOT NULL"),
        "quarantined_rows": one(f"SELECT COUNT(*) FROM {LEGACY_TABLE}"),
        "observations_statement": one("SELECT COUNT(*) FROM payment_observations WHERE provenance = 'STATEMENT_OBSERVED'"),
        "observations_confirmed": one("SELECT COUNT(*) FROM payment_observations WHERE provenance = 'SYSTEM_CONFIRMED'"),
        "observations_other": one("SELECT COUNT(*) FROM payment_observations WHERE provenance = 'OTHER_OBSERVED'"),
        "payee_reputation_rows": one("SELECT COUNT(*) FROM payee_reputation"),
        "payee_payer_edges": one("SELECT COUNT(*) FROM payee_payers"),
        "payee_reports": one("SELECT COUNT(*) FROM payee_reports"),
        "blocked_payees": one("SELECT COUNT(*) FROM payee_reputation WHERE blocked = 1"),
        "graph_cache_rows": one("SELECT COUNT(*) FROM payee_graph_cache"),
        "behavior_profiles": one("SELECT COUNT(*) FROM behavior_profiles"),
        "users": one("SELECT COUNT(*) FROM users"),
        "scored_transactions": one("SELECT COUNT(*) FROM scored_transactions"),
    }


def _has(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone() is not None


def _stale_edge_sql(conn: sqlite3.Connection, alias: str = "") -> str:
    """Edges from a statement user with no ledger row behind them."""
    p = f"{alias}." if alias else ""
    users = STATEMENT_USERS + (f" UNION SELECT DISTINCT user_id FROM {LEGACY_TABLE}"
                               if _has(conn, LEGACY_TABLE) else "")
    return (f"{p}payer_id IN ({users}) "
            f"AND NOT EXISTS (SELECT 1 FROM payment_observations o WHERE o.vpa = {p}vpa "
            f"AND o.payer_id = {p}payer_id AND o.in_aggregate = 1)")


def plan(conn: sqlite3.Connection) -> dict:
    """Read-only. Which payees' derived rows must be rebuilt.

    A payee is rebuilt when any of its edges is statement-derived: either an
    old unledgered edge from a statement user (the legacy bootstrap), or an
    edge backed by STATEMENT_OBSERVED ledger rows (re-derived in step 5).
    Ledgered SYSTEM_CONFIRMED / OTHER_OBSERVED edges on such a payee are
    replayed exactly. An UNLEDGERED edge from a non-statement user on such a
    payee has no per-payment amounts to replay - that payee is "mixed".
    """
    stale = _stale_edge_sql(conn, "pp")
    unledgered = ("NOT EXISTS (SELECT 1 FROM payment_observations o WHERE o.vpa = pp.vpa "
                  "AND o.payer_id = pp.payer_id AND o.in_aggregate = 1)")
    stale_payees = sorted({r[0] for r in conn.execute(
        f"SELECT DISTINCT vpa FROM payee_payers pp WHERE {stale}")} | {r[0] for r in conn.execute(
        "SELECT DISTINCT vpa FROM payment_observations WHERE provenance = 'STATEMENT_OBSERVED'")})
    preserved_unledgered = {r[0] for r in conn.execute(
        f"SELECT DISTINCT vpa FROM payee_payers pp WHERE {unledgered} AND NOT ({stale})")}
    return {
        "stale_payees": stale_payees,
        "stale_edges": conn.execute(f"SELECT COUNT(*) FROM payee_payers pp WHERE {stale}").fetchone()[0],
        "mixed_payees": sorted(preserved_unledgered & set(stale_payees)),
        "preserved_edges": conn.execute(f"SELECT COUNT(*) FROM payee_payers pp WHERE NOT ({stale})").fetchone()[0],
    }


def retained_files(conn: sqlite3.Connection, folder: Path) -> tuple[list, list]:
    extra = f" UNION SELECT DISTINCT user_id FROM {LEGACY_TABLE}" if _has(conn, LEGACY_TABLE) else ""
    known = {r[0] for r in conn.execute(f"{STATEMENT_USERS} UNION SELECT id FROM users{extra}")}
    usable, unknown = [], []
    for path in sorted(folder.glob("*")) if folder.exists() else []:
        m = re.match(r"^(.+?)_stmt_[0-9a-z]{8}_", path.name)
        owner = m.group(1) if m else None
        (usable if owner in known else unknown).append((owner, path))
    return usable, unknown


def backup(db_path: Path) -> Path:
    target_dir = db_path.parent / "backups"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{db_path.stem}.{datetime.now():%Y%m%d-%H%M%S}.db"
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def rebuild(conn: sqlite3.Connection, info: dict, purge_raw: bool) -> dict:
    from backend.app.services.graph_cache import backfill
    from backend.app.services.payee_reputation import apply_to_aggregates, record_payment

    done = {}
    now = datetime.now().isoformat()
    conn.execute(f"CREATE TABLE IF NOT EXISTS {LEGACY_TABLE} AS "
                 f"SELECT *, NULL AS quarantined_at FROM statement_transactions WHERE 0")
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        # 2. quarantine
        cols = [r[1] for r in conn.execute("PRAGMA table_info(statement_transactions)")]
        legacy_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({LEGACY_TABLE})")}
        for c in cols:
            if c not in legacy_cols:
                conn.execute(f"ALTER TABLE {LEGACY_TABLE} ADD COLUMN {c}")
        col_list = ", ".join(cols)
        done["quarantined"] = conn.execute(
            f"INSERT INTO {LEGACY_TABLE} ({col_list}, quarantined_at) "
            f"SELECT {col_list}, ? FROM statement_transactions WHERE fingerprint IS NULL", (now,)
        ).rowcount
        conn.execute("DELETE FROM statement_transactions WHERE fingerprint IS NULL")
        if purge_raw:
            conn.execute(f"UPDATE {LEGACY_TABLE} SET raw_line = NULL")

        # 3. statement observations (re-derived in 5)
        done["statement_observations_removed"] = conn.execute(
            "DELETE FROM payment_observations WHERE provenance = 'STATEMENT_OBSERVED'").rowcount

        # 4. stale payees: save flags, clear derived rows
        stale = list(info["stale_payees"])
        blocked = {r[0] for r in conn.execute("SELECT vpa FROM payee_reputation WHERE blocked = 1")}
        conn.executemany("DELETE FROM payee_payers WHERE vpa = ?", [(v,) for v in stale])
        conn.executemany("DELETE FROM payee_reputation WHERE vpa = ?", [(v,) for v in stale])
        done["payees_cleared"] = len(stale)

        # 5a. kept ledger rows on cleared payees
        stale_set = set(stale)
        replayed = 0
        for r in conn.execute("SELECT vpa, payer_id, amount, at FROM payment_observations "
                              "WHERE in_aggregate = 1 AND at IS NOT NULL ORDER BY at").fetchall():
            if r[0] in stale_set:
                apply_to_aggregates(conn, r[0], r[1], float(r[2] or 0), r[3])
                replayed += 1
        done["ledger_rows_replayed"] = replayed

        # 5b. canonical statement rows -> STATEMENT_OBSERVED
        rows = conn.execute(
            "SELECT user_id, counterparty_upi_id, counterparty_name, amount, timestamp, fingerprint "
            "FROM statement_transactions WHERE fingerprint IS NOT NULL AND direction = 'DEBIT' "
            "AND counterparty_upi_id IS NOT NULL "
            "AND COALESCE(validation_reasons, '') NOT LIKE '%timestamp_in_future%' "
            f"AND UPPER(COALESCE(status, 'SUCCESS')) NOT IN {_NOT_COMPLETED} "
            "ORDER BY COALESCE(timestamp, created_at)").fetchall()
        recorded = 0
        for r in rows:
            recorded += bool(record_payment(
                r[1], payer_id=r[0], amount=r[3], at=r[4], display_name=r[2], conn=conn,
                autocommit=False, provenance="STATEMENT_OBSERVED", observation_key=r[5],
                timed=r[4] is not None))
        done["statement_observations_recorded"] = recorded

        # restore report counts and block flags on cleared payees
        for vpa in stale:
            n = conn.execute("SELECT COUNT(DISTINCT reporter) FROM payee_reports WHERE vpa = ?",
                             (vpa,)).fetchone()[0]
            if n or vpa in blocked:
                conn.execute(
                    "INSERT INTO payee_reputation (vpa, first_seen, last_seen, reports, blocked) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(vpa) DO UPDATE SET "
                    "reports = excluded.reports, blocked = excluded.blocked",
                    (vpa, now, now, n, 1 if vpa in blocked else 0))

        # 6. graph
        conn.execute("DELETE FROM payee_graph_cache")
        conn.execute("DELETE FROM graph_components")
        done["graph"] = backfill(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return done


def rebuild_profiles(reset_orphans: bool) -> dict:
    from backend.app.services.statement_parser import generate_behavior_profile

    conn = store._get_connection()
    try:
        users = [r[0] for r in conn.execute("SELECT DISTINCT user_id FROM statement_transactions")]
        profiled = [r[0] for r in conn.execute("SELECT user_id FROM behavior_profiles")]
    finally:
        conn.close()
    for user in users:
        history = store.get_user_transactions(user)
        store.save_behavior_profile(user_id=user, source_type="rebuilt",
                                    profile=generate_behavior_profile(history.to_dict(orient="records")),
                                    transaction_count=int(len(history)))
    orphans = [u for u in profiled if u not in users]
    if reset_orphans and orphans:
        conn = store._get_connection()
        try:
            with conn:
                conn.executemany("DELETE FROM behavior_profiles WHERE user_id = ?", [(u,) for u in orphans])
        finally:
            conn.close()
    return {"profiles_rebuilt": len(users), "orphan_profiles": len(orphans),
            "orphan_profiles_reset": bool(reset_orphans and orphans)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(store.DB_PATH))
    ap.add_argument("--apply", action="store_true", help="write changes (default is a dry run)")
    ap.add_argument("--reimport-retained", action="store_true")
    ap.add_argument("--retained-dir", default="data/uploaded_statements")
    ap.add_argument("--import", dest="extra", action="append", default=[], metavar="FILE")
    ap.add_argument("--user", action="append", default=[], help="owner for each --import")
    ap.add_argument("--purge-raw-text", action="store_true")
    ap.add_argument("--reset-orphan-profiles", action="store_true")
    ap.add_argument("--reseed-demo", action="store_true")
    ap.add_argument("--drop-mixed-legacy", action="store_true")
    ap.add_argument("--no-backup", action="store_true", help="only for throwaway copies")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"no database at {db_path}")
        return 1
    store.DB_PATH = db_path
    store.DATA_DIR = db_path.parent
    store._SCHEMA_DONE.clear()

    from backend.app.services.payee_reputation import connect

    # Dry run: a read-only connection, so looking cannot change anything.
    conn = (connect() if args.apply
            else sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True))
    conn.row_factory = sqlite3.Row
    before = _counts(conn)
    info = plan(conn)
    usable, unknown = retained_files(conn, ROOT / args.retained_dir if not Path(args.retained_dir).is_absolute() else Path(args.retained_dir))
    if len(args.extra) != len(args.user):
        print("each --import needs a matching --user")
        return 2

    print("BEFORE", json.dumps(before, indent=1))
    print(f"stale payees (built from legacy statement rows): {len(info['stale_payees'])} "
          f"({info['stale_edges']} edges)")
    print(f"preserved edges (demo seeds / non-statement users): {info['preserved_edges']}")
    print(f"mixed payees (cannot be split exactly): {len(info['mixed_payees'])}")
    print(f"retained files re-importable: {len(usable)}; unknown owner: {len(unknown)}")

    if info["mixed_payees"] and not args.drop_mixed_legacy:
        print("REFUSING: mixed payees exist. Re-run with --drop-mixed-legacy to clear them "
              "entirely (their preserved edges are lost), or resolve them by hand.")
        conn.close()
        return 3
    if not args.apply:
        print("\nDRY RUN - nothing was written. Add --apply to rebuild.")
        conn.close()
        return 0

    if not args.no_backup:
        conn.close()
        print("backup:", backup(db_path))
        conn = connect()
    done = rebuild(conn, info, args.purge_raw_text)
    conn.close()

    from backend.app.ingestion.pipeline import import_statement

    imports = []
    jobs = [(o, p) for o, p in usable] if args.reimport_retained else []
    jobs += [(u, Path(f)) for f, u in zip(args.extra, args.user)]
    for owner, path in jobs:
        try:
            r = import_statement(owner, path.name, path.read_bytes())
            imports.append({"owner": owner, "status": r.status,
                            "stored": r.report["stored_transactions"],
                            "duplicates": r.report["duplicate_transactions"],
                            "partial": r.report["partial_transactions"],
                            "rejected": r.report["rejected_transactions"]})
        except Exception as exc:  # report and continue; content never printed
            imports.append({"owner": owner, "status": "FAILED", "error": type(exc).__name__})

    if args.reseed_demo:
        sys.path.insert(0, str(ROOT / "scripts"))
        import bootstrap_payee_reputation as boot

        conn = connect()
        try:
            if not conn.execute("SELECT 1 FROM payee_payers WHERE payer_id LIKE 'demo_payer_%' LIMIT 1").fetchone():
                boot.demo_payees(conn)
            if not conn.execute("SELECT 1 FROM payee_payers WHERE payer_id LIKE 'demo_crowd_%' LIMIT 1").fetchone():
                boot.demo_statement_payees(conn)
        finally:
            conn.close()

    profiles = rebuild_profiles(args.reset_orphan_profiles)
    conn = connect()
    after = _counts(conn)
    conn.close()
    manifest = {"db": str(db_path), "at": datetime.now().isoformat(), "before": before,
                "after": after, "steps": done, "imports": imports, "profiles": profiles}
    out = db_path.parent / "backups" / f"rebuild-manifest.{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))
    print("AFTER", json.dumps(after, indent=1))
    print("steps", json.dumps(done), "\nimports", json.dumps(imports), "\nprofiles", json.dumps(profiles))
    print("manifest:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
