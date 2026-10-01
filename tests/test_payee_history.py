"""payee_history(): the one source of "paid before", its temporal rule, its use
by /payee/check, V1 and V2, and the end-to-end regression from the bug report."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from backend.app.services import profile_store
from backend.app.services import payee_reputation as pr
from backend.app.services.payee_history import payee_history

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "statements"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(profile_store, "DB_PATH", tmp_path / "hist.db")
    profile_store._SCHEMA_DONE.clear()
    conn = pr.connect()
    yield conn
    conn.close()
    profile_store._SCHEMA_DONE.clear()


@pytest.fixture()
def api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(profile_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(profile_store, "DB_PATH", tmp_path / "api.db")
    profile_store._SCHEMA_DONE.clear()
    import backend.main as main

    client = TestClient(main.app)

    def login(upi: str) -> dict:
        r = client.post("/auth/register", json={"username": upi, "password": "hunter2hunter2"})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['token']}"}

    client.login = login
    yield client
    profile_store._SCHEMA_DONE.clear()


def _demo_statement() -> bytes:
    """docs/demo/DEMO_UPI_Statement.csv plus the rows appended in the bug
    report (CRLF, trailing spaces - as a spreadsheet re-save leaves them)."""
    return ((ROOT / "docs/demo/DEMO_UPI_Statement.csv").read_bytes()
            + (FIX / "demo_appended_rows.csv").read_bytes())


QR = ("upi://pay?pa=9063453458@axl&pn=CHINTHADA%20MURALI%20NAGARAJU"
      "&mc=0000&mode=02&purpose=00&am=19.87")


# ── Temporal rule ────────────────────────────────────────────────────────────

def test_only_payments_strictly_before_t_are_previous(db):
    pr.record_payment("synth.p@ybl", "u1", 100, at="2026-07-01T10:00:00", conn=db,
                      provenance="STATEMENT_OBSERVED", observation_key="k1")
    pr.record_payment("synth.p@ybl", "u1", 200, at="2026-07-05T10:00:00", conn=db,
                      provenance="STATEMENT_OBSERVED", observation_key="k2")
    at_t = payee_history("u1", "synth.p@ybl", before="2026-07-05T10:00:00", conn=db)
    assert at_t.payment_count == 1                          # the one AT T is excluded
    assert at_t.last_payment_at == datetime(2026, 7, 1, 10, 0)
    assert payee_history("u1", "synth.p@ybl", before="2026-07-01T10:00:00",
                         conn=db).payment_count == 0         # nothing before the first
    later = payee_history("u1", "synth.p@ybl", before="2026-07-05T10:00:01", conn=db)
    assert later.payment_count == 2 and later.days_since_first == pytest.approx(4.0, abs=1e-3)


def test_observation_keys_make_recording_idempotent(db):
    assert pr.record_payment("synth.q@ybl", "u1", 50, at="2026-07-01T10:00:00", conn=db,
                             provenance="STATEMENT_OBSERVED", observation_key="same")
    assert not pr.record_payment("synth.q@ybl", "u1", 50, at="2026-07-01T10:00:00", conn=db,
                                 provenance="STATEMENT_OBSERVED", observation_key="same")
    assert pr.assess_payee("synth.q@ybl", conn=db).payment_count == 1


def test_provenance_is_reported_as_observation_not_verification(db):
    pr.record_payment("synth.r@ybl", "u1", 50, at="2026-07-01T10:00:00", conn=db,
                      provenance="STATEMENT_OBSERVED", observation_key="a")
    pr.record_payment("synth.r@ybl", "u2", 50, at="2026-07-02T10:00:00", conn=db,
                      provenance="SYSTEM_CONFIRMED")
    rep = pr.assess_payee("synth.r@ybl", conn=db).as_dict()
    assert rep["observed_by"] == {"SYSTEM_CONFIRMED": 1, "STATEMENT_OBSERVED": 1,
                                  "OTHER_OBSERVED": 0}
    assert "verified" not in str(rep).lower()


def test_legacy_edges_without_a_ledger_still_count(db):
    """payee_payers rows written before the ledger existed."""
    db.execute("INSERT INTO payee_payers (vpa, payer_id, payments, first_at, last_at) "
               "VALUES ('synth.old@ybl', 'u1', 3, '2026-01-01T00:00:00', '2026-02-01T00:00:00')")
    db.commit()
    h = payee_history("u1", "synth.old@ybl", before="2026-09-01", conn=db)
    assert h.payment_count == 3 and h.provenance == {"OTHER_OBSERVED": 3}


# ── V1 and V2 read the same history ─────────────────────────────────────────

def test_v1_and_v2_inputs_come_from_the_same_history(db):
    from backend.ml.features import build_vector
    from backend.app.services.payee_history import build_v2_vector

    pr.record_payment("synth.s@ybl", "u1", 100, at="2026-07-01T10:00:00", conn=db,
                      provenance="STATEMENT_OBSERVED", observation_key="s1")
    h = payee_history("u1", "synth.s@ybl", before="2026-07-11T10:00:00", conn=db)
    rep = pr.assess_payee("synth.s@ybl", conn=db).as_dict()
    v1 = build_vector(amount=100.0, timestamp="2026-07-11T10:00:00",
                      profile={"median_amount": 100.0}, reputation=rep, **h.v1_inputs())
    assert v1.values["payee_is_new"] == 0.0
    v2 = build_v2_vector(h, amount=100.0, timestamp="2026-07-11T10:00:00",
                                      profile={"median_amount": 100.0}, reputation=rep)
    assert v2.values["prior_payments_to_payee"] == 1.0
    assert v2.values["days_since_first_payment_to_payee"] == pytest.approx(10.0)


# ── Phone-number name mismatch ──────────────────────────────────────────────

def test_phone_style_address_gets_no_high_name_mismatch():
    from backend.app.services.upi_qr import parse_upi_target

    codes = {f.code: f.severity for f in parse_upi_target(QR).all_findings}
    assert "name_mismatch" not in codes
    assert codes.get("name_not_comparable") == "info"


def test_merchant_address_name_mismatch_is_still_detected():
    from backend.app.services.upi_qr import parse_upi_target

    qr = parse_upi_target("upi://pay?pa=rakesh9911@ybl&pn=Reliance%20Digital&am=48999")
    assert {f.code: f.severity for f in qr.all_findings}.get("name_mismatch") == "high"


# ── End to end: the reported bug ─────────────────────────────────────────────

def test_regression_9063453458_statement_payment_is_found_at_check_time(api, monkeypatch):
    import backend.app.services.payee_check as pc

    captured = {}
    real = pc.build_vector

    def spy(**kwargs):
        vector = real(**kwargs)
        captured["values"] = dict(vector.values)
        return vector

    monkeypatch.setattr(pc, "build_vector", spy)

    headers = api.login("synth.payer@okaxis")
    up = api.post("/statement/upload", headers=headers,
                  files={"file": ("DEMO_UPI_Statementedited.csv", _demo_statement(), "text/csv")})
    assert up.status_code == 200, up.text
    body = up.json()
    assert "extracted_transactions" not in body
    assert body["import"]["payee_relationships_updated"] >= 1

    r = api.post("/payee/check", headers=headers, json={"payload": QR, "amount": 19.87})
    assert r.status_code == 200, r.text
    res = r.json()
    codes = {f["code"] for f in res["findings"]}
    rel = next(e for e in res["evidence"] if e["family"] == "relationship")

    assert res["payee"]["key"] == "9063453458@axl"
    assert res["payee_status"] == "PAID_ONCE"
    assert res["history_state"]["state"] == "HISTORY_FOUND"
    assert res["payee_history"]["payment_count"] == 1
    assert res["payee_history"]["first_payment_at"] == "2026-06-18T09:56"
    assert res["payee_history"]["sources"] == {"STATEMENT_OBSERVED": 1}
    assert rel["code"] == "relationship_established"
    assert "payee_unseen" not in codes                       # no "No payment history"
    assert "relationship_new" not in {e["code"] for e in res["evidence"]}
    assert "first payment" not in str(res).lower()
    assert "name_mismatch" not in codes                      # phone-style address
    # V1 received real history, from the same source.
    assert captured["values"]["payee_is_new"] == 0.0
    assert captured["values"]["payee_history_available"] == 1.0
    assert captured["values"]["payee_distinct_payers"] == 1.0
    # ...and the score is still computed normally.
    assert isinstance(res["risk_score"], (int, float)) and res["verdict"]


def test_regression_unknown_payee_still_reports_no_history(api):
    headers = api.login("synth.payer@okaxis")
    api.post("/statement/upload", headers=headers,
             files={"file": ("s.csv", _demo_statement(), "text/csv")})
    res = api.post("/payee/check", headers=headers,
                   json={"payload": "unknown-example@upi", "amount": 100}).json()
    assert res["payee_status"] == "UNKNOWN"
    assert res["history_state"]["state"] == "NO_HISTORY"
    assert "payee_unseen" in {f["code"] for f in res["findings"]}
    assert next(e for e in res["evidence"] if e["family"] == "relationship")["code"] == "relationship_new"


def test_partial_row_for_this_payee_is_not_reported_as_no_history(api):
    headers = api.login("synth.payer@okaxis")
    api.post("/statement/upload", headers=headers,
             files={"file": ("s.csv", _demo_statement(), "text/csv")})
    res = api.post("/payee/check", headers=headers,
                   json={"payload": "9000000301@ybl", "amount": 100}).json()
    assert res["payee_status"] == "HISTORY_UNAVAILABLE_DUE_TO_PARTIAL_IMPORT"
    assert res["history_state"]["state"] == "PARTIAL_IMPORT"
    unseen = [f for f in res["findings"] if f["code"] == "payee_unseen"]
    assert unseen and "could not be read" in unseen[0]["message"]


def test_name_only_statement_row_is_identity_unavailable(api):
    headers = api.login("synth.payer@okaxis")
    csv = (b"Date,Name,Type,Amount\n01/07/2026,SYNTH NAMEONLY,Paid,100\n")
    api.post("/statement/upload", headers=headers, files={"file": ("n.csv", csv, "text/csv")})
    res = api.post("/payee/check", headers=headers,
                   json={"payload": "upi://pay?pa=synth.nameonly@ybl&pn=SYNTH%20NAMEONLY",
                         "amount": 100}).json()
    assert res["payee_status"] == "NAME_ONLY_HISTORY"
    assert res["history_state"]["state"] == "IDENTITY_UNAVAILABLE"


def test_failed_import_is_import_error_not_no_history(api):
    headers = api.login("synth.payer@okaxis")
    bad = api.post("/statement/upload", headers=headers,
                   files={"file": ("x.pdf", b"%PDF-1.4 truncated", "application/pdf")})
    assert bad.status_code == 400
    res = api.post("/payee/check", headers=headers,
                   json={"payload": "synth.somebody@ybl", "amount": 100}).json()
    assert res["history_state"]["state"] == "IMPORT_ERROR"
    assert "failed" in res["history_state"]["message"].lower()


def test_another_users_statement_is_aggregate_only(api):
    alice = api.login("synth.alice@okaxis")
    api.post("/statement/upload", headers=alice,
             files={"file": ("s.csv", _demo_statement(), "text/csv")})
    bob = api.login("synth.bob@okaxis")
    res = api.post("/payee/check", headers=bob, json={"payload": QR, "amount": 19.87}).json()
    assert res["payee_status"] == "NEW_TO_USER_BUT_OBSERVED"
    assert res["payee_history"]["payment_count"] == 0
    assert res["reputation"]["distinct_payers"] == 1
    text = str(res)
    for leak in ("2026-06-18", "1000.0", "CHINTJADA", "synth.alice"):
        assert leak not in text, leak
    assert res["reputation"]["detail_withheld"] is True


def test_confirm_records_a_system_confirmed_payment(api):
    headers = api.login("synth.payer@okaxis")
    r = api.post("/payee/confirm", headers=headers,
                 json={"payload": "synth.confirmed@ybl", "amount": 42})
    assert r.json()["recorded"] is True
    res = api.post("/payee/check", headers=headers,
                   json={"payload": "synth.confirmed@ybl", "amount": 42}).json()
    assert res["payee_status"] == "PAID_ONCE"
    assert res["payee_history"]["sources"] == {"SYSTEM_CONFIRMED": 1}


def test_statement_import_updates_the_graph_once(api):
    from backend.app.services.graph_cache import graph_view

    headers = api.login("synth.payer@okaxis")
    for _ in range(2):
        api.post("/statement/upload", headers=headers,
                 files={"file": ("s.csv", _demo_statement(), "text/csv")})
    conn = pr.connect()
    try:
        view = graph_view(conn, "9063453458@axl")
        assert view.available
        edges = conn.execute("SELECT COUNT(*) FROM payee_payers WHERE vpa = ?",
                             ("9063453458@axl",)).fetchone()[0]
        payments = conn.execute("SELECT payments FROM payee_payers WHERE vpa = ?",
                                ("9063453458@axl",)).fetchone()[0]
    finally:
        conn.close()
    assert (edges, payments) == (1, 1)


def test_upload_response_is_a_summary(api):
    headers = api.login("synth.payer@okaxis")
    body = api.post("/statement/upload", headers=headers,
                    files={"file": ("s.csv", _demo_statement(), "text/csv")}).json()
    for key in ("total_rows_detected", "valid_transactions", "partial_transactions",
                "rejected_transactions", "duplicate_transactions", "upi_linked_transactions",
                "payee_relationships_updated", "profile_updated", "graph_observations_updated",
                "cross_source_duplicates", "ambiguous_cross_source_kept"):
        assert key in body["import"], key
    assert body["status"] == "PARTIAL"                        # the DEBET row
    rows = api.get("/statement-transactions", headers=headers).json()["transactions"]
    assert rows and all("raw_line" not in r for r in rows)


def test_bootstrap_script_does_not_recount_imported_statements(db):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import bootstrap_payee_reputation as boot
    from backend.app.ingestion.pipeline import import_statement

    import_statement("u1", "s.csv", _demo_statement(), imported_at=datetime(2026, 9, 21, 12))
    before = db.execute("SELECT COUNT(*) FROM payment_observations").fetchone()[0]
    reps = db.execute("SELECT SUM(payment_count) FROM payee_reputation").fetchone()[0]
    boot.bootstrap(db, reset=False)
    boot.bootstrap(db, reset=False)
    assert db.execute("SELECT COUNT(*) FROM payment_observations").fetchone()[0] == before
    assert db.execute("SELECT SUM(payment_count) FROM payee_reputation").fetchone()[0] == reps


def test_every_report_field_the_upload_screen_reads_is_in_the_response(api):
    """The import summary UI reads `report.<field>`; each one must exist."""
    import re

    tsx = (ROOT / "src/components/StatementProfiling.tsx").read_text(encoding="utf-8")
    fields = set(re.findall(r"report\.([a-z_]+)", tsx)) - {"status"}
    headers = api.login("synth.payer@okaxis")
    body = api.post("/statement/upload", headers=headers,
                    files={"file": ("s.csv", _demo_statement(), "text/csv")}).json()
    missing = fields - set(body["import"])
    assert not missing, missing
    assert {"total_rows_detected", "valid_transactions", "partial_transactions",
            "rejected_transactions", "duplicate_transactions", "upi_linked_transactions",
            "profile_updated", "graph_observations_updated"} <= fields
