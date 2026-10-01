"""Statement ingestion: canonical model, identity, direction, validation,
dedupe, storage and privacy. Fixtures are synthetic (see
tests/fixtures/statements/README.md)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from backend.app.ingestion.canonical import Direction, Status
from backend.app.ingestion.direction import parse_direction
from backend.app.ingestion.identity import is_phone_style, normalize_upi_id
from backend.app.ingestion.pipeline import import_statement
from backend.app.services import profile_store
from backend.app.services.payee_history import payee_history
from backend.app.services.vpa import normalise_vpa

FIX = Path(__file__).parent / "fixtures" / "statements"
ROOT = Path(__file__).resolve().parents[1]
IMPORTED_AT = datetime(2026, 9, 21, 12, 0)
T = "2026-09-21T12:00:00"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(profile_store, "DB_PATH", tmp_path / "ingest.db")
    profile_store._SCHEMA_DONE.clear()
    yield tmp_path / "ingest.db"
    profile_store._SCHEMA_DONE.clear()


def _import(user: str, name: str, content: bytes | None = None):
    data = content if content is not None else (FIX / name).read_bytes()
    return import_statement(user, name, data, imported_at=IMPORTED_AT)


def _rows(db_path: Path, user: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM statement_transactions WHERE user_id = ?",
                            (user,)).fetchall()
    finally:
        conn.close()


def _count(db_path: Path, sql: str, *args) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, args).fetchone()[0]
    finally:
        conn.close()


# ── 1. One normaliser ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "9063453458@axl", "9063453458@AXL", '"9063453458@axl"', " UPI ID: 9063453458@axl",
    "9063453458 @ axl", "​9063453458@axl﻿", " 9063453458@axl ",
    "“9063453458@axl”", "VPA: 9063453458@axl",
])
def test_upi_id_variants_normalise_to_one_identity(raw):
    assert normalize_upi_id(raw) == "9063453458@axl"
    # Lookup uses the same function, so a stored key is always found.
    assert normalise_vpa(raw) == "9063453458@axl"


@pytest.mark.parametrize("raw", ["9063 453458@axl", "a@b@c", "plainname", "x@1bank", "", None,
                                 "name@gmail.com.", "@ybl"])
def test_malformed_identifiers_are_not_repaired_into_valid_ones(raw):
    assert normalize_upi_id(raw) is None


def test_phone_style_detection():
    assert is_phone_style("9063453458")
    assert not is_phone_style("rakesh9911")
    assert not is_phone_style("chaipoint")


# ── 2. Direction ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("DEBIT", Direction.DEBIT), ("debit", Direction.DEBIT), ("DR", Direction.DEBIT),
    ("Dr.", Direction.DEBIT), ("WITHDRAWAL", Direction.DEBIT), ("Sent", Direction.DEBIT),
    ("PAID", Direction.DEBIT), ("paid to", Direction.DEBIT),
    ("CREDIT", Direction.CREDIT), ("CR", Direction.CREDIT), ("DEPOSIT", Direction.CREDIT),
    ("Received", Direction.CREDIT), ("RECEIVED FROM", Direction.CREDIT),
])
def test_direction_representations(raw, expected):
    assert parse_direction(raw) is expected


@pytest.mark.parametrize("raw", ["DEBET", "CREDT", "", "  ", "transfer", "?", None])
def test_unknown_or_misspelt_direction_is_none_never_debit(raw):
    assert parse_direction(raw) is None


# ── 3. CSV adapters (synthetic provider-like fixtures) ───────────────────────

def test_gpay_like_csv(db):
    r = _import("u1", "gpay_like.csv")
    rep = r.report
    assert rep["total_rows_detected"] == 7
    assert rep["credit_transactions"] == 1          # "Received" is not spending
    assert rep["malformed_upi_id"] == 1             # "synth grocer@..." has a space
    assert rep["unmapped_columns"] == []
    # Quoted, upper-case, padded UPI ID lands on the same payee.
    assert payee_history("u1", "synth.grocery@okhdfcbank", before=T).payment_count == 2
    # A FAILED payment is stored but is not a previous payment.
    assert payee_history("u1", "synth.shop@oksbi", before=T).paid_before is False


def test_phonepe_like_csv_statuses_and_reasons(db):
    r = _import("u1", "phonepe_like.csv")
    rep = r.report
    assert rep["rejected_transactions"] == 1        # empty amount
    assert rep["unknown_direction"] == 1            # DEBET
    assert rep["missing_timestamp"] == 1
    assert rep["future_timestamp"] == 1
    assert rep["debits_without_upi_id"] == 1        # "Paid to SYNTH AUTO"
    assert r.status == "PARTIAL"
    stored = _rows(db, "u1")
    # Rejected rows are counted, not stored; nothing else is dropped.
    assert len(stored) == rep["total_rows_detected"] - 1
    # The VPA is lifted from the details text only when exactly one is there.
    assert {row["counterparty_upi_id"] for row in stored} >= {"synth.pharmacy@ybl", "synth.bakery@ybl"}


def test_paytm_like_signed_amounts(db):
    r = _import("u1", "paytm_like.csv")
    assert (r.report["debit_transactions"], r.report["credit_transactions"]) == (2, 1)
    assert payee_history("u1", "9000000202@paytm", before=T).paid_before is False


def test_bank_like_csv_preamble_repeated_header_and_columns(db):
    r = _import("u1", "bank_like.csv")
    rep = r.report
    assert rep["total_rows_detected"] == 4          # repeated header is not a row
    assert (rep["debit_transactions"], rep["credit_transactions"]) == (3, 1)
    h = payee_history("u1", "synth.mart@okaxis", before=T)
    assert h.payment_count == 2 and h.total_amount == 2650.0


def test_csv_with_bom_and_crlf(db):
    content = "﻿timestamp,counterparty,upi_id,amount,direction\r\n" \
              "2026-07-01 10:00:00,SYNTH A,synth.a@ybl,10,DEBIT\r\n"
    r = _import("u1", "bom.csv", content.encode("utf-8"))
    assert r.report["valid_transactions"] == 1
    assert payee_history("u1", "synth.a@ybl", before=T).paid_before


def test_counterparty_column_is_read_and_kept_separate_from_upi_id(db):
    _import("u1", "gpay_like.csv")
    row = next(r for r in _rows(db, "u1") if r["counterparty_upi_id"] == "synthtea.stall@okaxis")
    assert row["counterparty_name"] == "SYNTH TEA STALL"
    assert row["merchant"] == "SYNTH TEA STALL"


def test_a_csv_without_recognisable_headers_is_an_import_error(db):
    from backend.app.ingestion.pipeline import ImportError_

    with pytest.raises(ImportError_):
        _import("u1", "junk.csv", b"foo,bar\n1,2\n")


# ── 4. Direction regressions (the demo statement's DEBET/CREDIT rows) ────────

def test_credit_rows_are_credits_and_debet_is_not_a_debit(db):
    _import("u1", "demo_appended_rows.csv",
            (ROOT / "docs/demo/DEMO_UPI_Statement.csv").read_bytes()
            + (FIX / "demo_appended_rows.csv").read_bytes())
    rows = {(r["counterparty_upi_id"], r["amount"]): r for r in _rows(db, "u1")
            if r["counterparty_upi_id"] in ("9000000301@ybl", "9000000302@ybl")}
    assert rows[("9000000301@ybl", 2000.0)]["direction"] is None          # DEBET
    assert rows[("9000000301@ybl", 2000.0)]["validation_status"] == "PARTIAL"
    assert rows[("9000000301@ybl", 800.0)]["direction"] == "CREDIT"
    assert rows[("9000000302@ybl", 2000.0)]["direction"] == "CREDIT"
    # Money received is not a payment to that person; DEBET is not either.
    assert not payee_history("u1", "9000000301@ybl", before=T).paid_before
    assert payee_history("u1", "9000000302@ybl", before=T).payment_count == 1
    profile = profile_store.get_behavior_profile("u1")
    assert profile["unknown_direction_count"] == 1
    assert profile["credit_count"] >= 2
    df = profile_store.get_user_transactions("u1")
    # Spending statistics come from DEBIT rows only: no credit, no unknown.
    assert profile["max_amount"] == df.loc[df["txn_type"] == "DEBIT", "amount"].max()


# ── 5. Validation: dates are never invented ──────────────────────────────────

def test_unreadable_date_is_null_not_now(db):
    _import("u1", "phonepe_like.csv")
    row = next(r for r in _rows(db, "u1") if r["counterparty_upi_id"] == "synth.mystery@ybl")
    assert row["timestamp"] is None
    assert "timestamp_unreadable" in row["validation_reasons"]
    h = payee_history("u1", "synth.mystery@ybl", before="2099-01-01")
    # Counted as paid (the statement says so), but it contributes no date.
    assert h.payment_count == 1 and h.first_payment_at is None and h.days_since_first is None


def test_future_dated_rows_are_partial_and_never_previous_payments(db):
    _import("u1", "phonepe_like.csv")
    assert not payee_history("u1", "synth.future@ybl", before="2099-01-01").paid_before
    # Hidden from every temporal consumer.
    df = profile_store.get_user_transactions("u1")
    assert df.loc[df["counterparty_upi_id"] == "synth.future@ybl", "timestamp"].isna().all()
    assert _count(db, "SELECT COUNT(*) FROM payment_observations WHERE vpa = ?",
                  "synth.future@ybl") == 0


def test_legacy_parser_no_longer_fabricates_timestamps():
    from backend.app.services.statement_parser import _normalize_timestamp

    assert _normalize_timestamp("not a date") is None
    assert _normalize_timestamp(None) is None
    assert _normalize_timestamp("") is None


# ── 6. Deduplication ─────────────────────────────────────────────────────────

def test_uploading_the_same_statement_twice_counts_nothing_twice(db):
    first = _import("u1", "bank_like.csv")
    obs = _count(db, "SELECT COUNT(*) FROM payment_observations")
    rep_count = _count(db, "SELECT payment_count FROM payee_reputation WHERE vpa = ?",
                       "synth.mart@okaxis")
    second = _import("u1", "bank_like.csv")
    assert second.report["stored_transactions"] == 0
    assert second.report["duplicate_transactions"] == first.report["stored_transactions"]
    assert second.report["payee_relationships_updated"] == 0
    assert _count(db, "SELECT COUNT(*) FROM payment_observations") == obs
    assert _count(db, "SELECT payment_count FROM payee_reputation WHERE vpa = ?",
                  "synth.mart@okaxis") == rep_count
    assert payee_history("u1", "synth.mart@okaxis", before=T).payment_count == 2


def test_identical_rows_inside_one_file_are_both_kept(db):
    row = "2026-07-01 09:56:00,SYNTH TEA,synth.tea@ybl,10,DEBIT\n"
    content = ("timestamp,counterparty,upi_id,amount,direction\n" + row + row).encode()
    r = _import("u1", "twice.csv", content)
    assert r.report["stored_transactions"] == 2
    assert r.report["repeated_rows_in_file"] == 1
    assert payee_history("u1", "synth.tea@ybl", before=T).payment_count == 2
    # ...and re-importing the file still leaves exactly two.
    _import("u1", "twice.csv", content)
    assert payee_history("u1", "synth.tea@ybl", before=T).payment_count == 2


def test_same_minute_same_amount_different_payees_are_not_merged(db):
    content = ("timestamp,counterparty,upi_id,amount,direction\n"
               "2026-07-01 09:56:00,SHOP,synth.one@ybl,100,DEBIT\n"
               "2026-07-01 09:56:00,SHOP,synth.two@ybl,100,DEBIT\n").encode()
    r = _import("u1", "two.csv", content)
    assert r.report["stored_transactions"] == 2


def test_a_transaction_id_listed_twice_in_one_file_is_one_payment(db):
    content = ("Date,Name,UPI ID,Type,Amount,Transaction ID\n"
               "01/07/2026,SYNTH X,synth.x@ybl,Paid,10,ABC123\n"
               "01/07/2026,SYNTH X,synth.x@ybl,Paid,10,ABC123\n").encode()
    r = _import("u1", "dup.csv", content)
    assert r.report["stored_transactions"] == 1
    assert r.report["duplicate_transactions"] == 1


# ── 7. Ownership and privacy ─────────────────────────────────────────────────

def test_a_statement_belongs_only_to_its_importer(db):
    _import("alice", "bank_like.csv")
    assert payee_history("bob", "synth.mart@okaxis", before=T).payment_count == 0
    assert profile_store.get_user_transactions("bob").empty


def test_raw_statement_text_is_not_stored(db):
    _import("u1", "gpay_like.csv")
    assert all(r["raw_line"] is None for r in _rows(db, "u1"))
    assert "raw_line" not in profile_store.get_user_transactions("u1").columns


def test_the_import_report_holds_counts_not_content(db):
    r = _import("u1", "gpay_like.csv")
    blob = json.dumps(r.report)
    for content in ("SYNTH TEA STALL", "synthtea.stall@okaxis", "610000000001", "1,245.50"):
        assert content not in blob


def test_temporary_pdf_copy_is_removed_even_when_the_pdf_is_broken(monkeypatch, tmp_path):
    import tempfile

    from backend.app.services.statement_parser import _parse_pdf_with_pdfplumber

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    result = _parse_pdf_with_pdfplumber(b"%PDF-1.4 truncated")
    assert result["transactions"] == []
    assert list(tmp_path.glob("*.pdf")) == []


# ── 8. PDF (synthetic, generated at test time) ───────────────────────────────

def _bank_rows(n: int):
    rows = [(f"{(i % 28) + 1:02d}/07/2026",
             "UPI/DR/6183%07d/SYNTH MART WITH A LONG NARRATION THAT WRAPS INSIDE ITS CELL/"
             "synth.mart@okaxis/Groceries and household items" % i,
             "6183%07d" % i, f"{100 + i}.00", "", "9,000.00") for i in range(n)]
    rows.append(("03/07/2026", "UPI/CR/6183999999/SYNTH EMPLOYER/payroll", "6183999999",
                 "", "30,000.00", "99,000.00"))
    return rows


def test_bank_like_pdf_multi_page_repeated_headers_and_wrapped_rows(db):
    from statement_pdf_fixtures import bank_like_table_pdf

    rows = _bank_rows(60)                       # three pages, header repeated on each
    r = _import("u1", "bank_like.pdf", bank_like_table_pdf(rows))
    rep = r.report
    assert rep["source_format"] == "pdf"
    assert rep["source_provider"] == "generic_pdf"      # "@okaxis" is not Axis Bank
    assert rep["total_rows_detected"] == len(rows)      # no header/footer became a row
    assert (rep["debit_transactions"], rep["credit_transactions"]) == (60, 1)
    h = payee_history("u1", "synth.mart@okaxis", before=T)
    assert h.payment_count == 60
    # Date-only rows: dated, but no clock time is claimed.
    stored = _rows(db, "u1")
    assert all(row["time_known"] == 0 for row in stored)


def test_gpay_like_pdf_names_without_upi_ids_are_identity_unavailable(db):
    from statement_pdf_fixtures import gpay_like_pdf, have_rupee_font

    if not have_rupee_font():
        pytest.skip("DejaVu font (rupee glyph) not installed")
    rows = [(datetime(2026, 7, d, 9, 15), "Paid to", "SYNTH TEA STALL", 20.0, f"6100000{d:05d}")
            for d in range(1, 6)]
    rows.append((datetime(2026, 7, 7, 18, 2), "Received from", "SYNTH FRIEND A", 500.0, "610000099999"))
    r = _import("u1", "gpay_like.pdf", gpay_like_pdf(rows))
    rep = r.report
    assert rep["source_provider"] == "google_pay_like"
    assert (rep["debit_transactions"], rep["credit_transactions"]) == (5, 1)
    # This layout prints names, not UPI IDs: nothing is invented.
    assert rep["upi_linked_transactions"] == 0
    assert rep["debits_without_upi_id"] == 5
    assert rep["payee_relationships_updated"] == 0


def test_phonepe_like_pdf_and_csv_reach_the_same_canonical_facts(db, tmp_path):
    from statement_pdf_fixtures import have_rupee_font, phonepe_like_pdf

    if not have_rupee_font():
        pytest.skip("DejaVu font (rupee glyph) not installed")
    rows = [dict(when=datetime(2026, 6, 18, 9, 56), name="SYNTH PAYEE", vpa="9000000401@axl",
                 amount=1000.0, direction="DEBIT"),
            dict(when=datetime(2026, 6, 20, 9, 56), name="SYNTH PAYEE B", vpa="9000000402@ybl",
                 amount=800.0, direction="CREDIT")]
    pdf = _import("pdf_user", "phonepe_like.pdf", phonepe_like_pdf(rows, tmp_path))
    csv = _import("csv_user", "same.csv", (
        "timestamp,counterparty,upi_id,amount,direction\n"
        "2026-06-18 09:56:00,SYNTH PAYEE,9000000401@axl,1000,DEBIT\n"
        "2026-06-20 09:56:00,SYNTH PAYEE B,9000000402@ybl,800,CREDIT\n").encode())
    assert pdf.report["source_provider"] == "phonepe_like"

    def facts(user):
        return sorted((r["timestamp"], r["amount"], r["direction"], r["counterparty_upi_id"],
                       r["counterparty_name"]) for r in _rows(db, user))

    assert facts("pdf_user") == facts("csv_user")
    for user in ("pdf_user", "csv_user"):
        h = payee_history(user, "9000000401@axl", before=T)
        assert (h.payment_count, h.first_payment_at) == (1, datetime(2026, 6, 18, 9, 56))
        assert not payee_history(user, "9000000402@ybl", before=T).paid_before


def test_existing_synthetic_demo_pdf_imports_completely(db):
    r = _import("u1", "DEMO_UPI_Statement.pdf",
                (ROOT / "docs/demo/DEMO_UPI_Statement.pdf").read_bytes())
    rep = r.report
    assert r.status == "COMPLETE"
    assert rep["total_rows_detected"] == rep["valid_transactions"] == 288
    assert rep["upi_linked_transactions"] == 288
