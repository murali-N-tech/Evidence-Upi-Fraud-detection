"""scripts/rebuild_statement_history.py on a synthetic legacy database."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rebuild_statement_history.py"


def _legacy_db(path: Path) -> None:
    """A database as the old pipeline left it: fingerprint-less rows (a credit
    stored as a debit), bootstrap edges from them, plus data that must survive."""
    code = f"""
import sys; sys.path.insert(0, {str(ROOT)!r})
from pathlib import Path
import backend.app.services.profile_store as ps
ps.DB_PATH = Path({str(path)!r}); ps.DATA_DIR = ps.DB_PATH.parent; ps._SCHEMA_DONE.clear()
from backend.app.services import payee_reputation as pr
conn = pr.connect()
rows = [("legacy_user", "2026-06-01T10:00:00", 500.0, "synth.shop@ybl", "DEBIT", "raw text A"),
        ("legacy_user", "2026-06-02T10:00:00", 900.0, "synth.friend@ybl", "DEBIT", "raw text B")]
for u, ts, amt, upi, tt, raw in rows:
    conn.execute("INSERT INTO statement_transactions (statement_id, user_id, timestamp, amount, "
                 "merchant, upi_id, status, source_type, raw_line, txn_type, created_at) "
                 "VALUES ('s0', ?, ?, ?, 'X', ?, 'SUCCESS', 'csv', ?, ?, '2026-06-03')",
                 (u, ts, amt, upi, raw, tt))
conn.commit()
# the old bootstrap: unledgered edges built from those rows
for u, ts, amt, upi, tt, raw in rows:
    conn.execute("INSERT INTO payee_payers VALUES (?, ?, 1, ?, ?)", (upi, u, ts, ts))
    conn.execute("INSERT INTO payee_reputation (vpa, first_seen, last_seen, payment_count, total_amount) "
                 "VALUES (?, ?, ?, 1, ?)", (upi, ts, ts, amt))
conn.commit()
# must survive: a demo seed, a confirmed payment, a report and a block
conn.execute("INSERT INTO payee_payers VALUES ('demo-mule@ybl', 'demo_payer_1', 1, '2026-06-01', '2026-06-01')")
conn.execute("INSERT INTO payee_reputation (vpa, first_seen, last_seen, payment_count, total_amount) "
             "VALUES ('demo-mule@ybl', '2026-06-01', '2026-06-01', 1, 4999)")
conn.commit()
pr.record_payment("synth.confirmed@ybl", "confirm_user", 42, at="2026-06-05T09:00:00",
                  conn=conn, provenance="SYSTEM_CONFIRMED")
conn.execute("INSERT INTO payee_reports (vpa, reporter, reason, created_at) "
             "VALUES ('synth.shop@ybl', 'someone', 'x', '2026-06-04')")
conn.execute("UPDATE payee_reputation SET reports = 1, blocked = 1 WHERE vpa = 'synth.shop@ybl'")
conn.commit(); conn.close()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _run(db: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--db", str(db), *args],
                          capture_output=True, text=True, cwd=ROOT, timeout=120)


def _q(db: Path, sql: str, *a):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, a).fetchall()
    finally:
        conn.close()


def test_dry_run_changes_nothing(tmp_path):
    db = tmp_path / "legacy.db"
    _legacy_db(db)
    before = db.read_bytes()
    r = _run(db)
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout
    assert db.read_bytes() == before
    assert not (tmp_path / "backups").exists()


def test_apply_quarantines_rebuilds_and_preserves(tmp_path):
    db = tmp_path / "legacy.db"
    _legacy_db(db)
    r = _run(db, "--apply", "--purge-raw-text")
    assert r.returncode == 0, r.stderr + r.stdout
    assert list((tmp_path / "backups").glob("legacy.*.db")), "backup written first"
    # legacy rows moved, not deleted; raw text cleared on request
    assert _q(db, "SELECT COUNT(*) FROM statement_transactions")[0][0] == 0
    assert _q(db, "SELECT COUNT(*), SUM(raw_line IS NOT NULL) FROM statement_transactions_legacy")[0] == (2, 0)
    # stale statement-derived edges gone
    assert _q(db, "SELECT COUNT(*) FROM payee_payers WHERE payer_id = 'legacy_user'")[0][0] == 0
    assert _q(db, "SELECT COUNT(*) FROM payee_reputation WHERE vpa = 'synth.friend@ybl'")[0][0] == 0
    # preserved: demo seed exactly, confirmed payment, report count and block flag
    assert _q(db, "SELECT payment_count, total_amount FROM payee_reputation WHERE vpa = 'demo-mule@ybl'") == [(1, 4999.0)]
    assert _q(db, "SELECT payer_id, provenance FROM payment_observations") == [("confirm_user", "SYSTEM_CONFIRMED")]
    assert _q(db, "SELECT payment_count FROM payee_reputation WHERE vpa = 'synth.confirmed@ybl'") == [(1,)]
    assert _q(db, "SELECT reports, blocked, payment_count FROM payee_reputation WHERE vpa = 'synth.shop@ybl'") == [(1, 1, 0)]
    # graph rebuilt from what remains
    assert _q(db, "SELECT COUNT(*) FROM payee_graph_cache")[0][0] == 2


def test_apply_then_reimport_gives_one_history(tmp_path):
    db = tmp_path / "legacy.db"
    _legacy_db(db)
    csv = tmp_path / "legacy_user_stmt_abcd1234_s.csv"
    csv.write_text("timestamp,counterparty,upi_id,amount,direction\n"
                   "2026-06-01 10:00:00,SHOP,synth.shop@ybl,500,DEBIT\n"
                   "2026-06-02 10:00:00,FRIEND,synth.friend@ybl,900,CREDIT\n")
    r = _run(db, "--apply", "--import", str(csv), "--user", "legacy_user")
    assert r.returncode == 0, r.stderr + r.stdout
    edges = _q(db, "SELECT vpa, payments FROM payee_payers WHERE payer_id = 'legacy_user'")
    assert edges == [("synth.shop@ybl", 1)]          # the CREDIT is no longer a payment
    assert _q(db, "SELECT provenance, COUNT(*) FROM payment_observations "
                  "WHERE payer_id = 'legacy_user' GROUP BY 1") == [("STATEMENT_OBSERVED", 1)]
    # running it again changes nothing
    r2 = _run(db, "--apply", "--no-backup", "--import", str(csv), "--user", "legacy_user")
    assert r2.returncode == 0, r2.stderr
    assert _q(db, "SELECT vpa, payments FROM payee_payers WHERE payer_id = 'legacy_user'") == edges
    assert _q(db, "SELECT COUNT(*) FROM statement_transactions")[0][0] == 2
