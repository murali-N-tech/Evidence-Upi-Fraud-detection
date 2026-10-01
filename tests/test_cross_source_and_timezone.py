"""Cross-source duplicate identity and the IST time convention."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.app.core.clock import now_app, to_app_naive
from backend.app.ingestion.pipeline import import_statement
from backend.app.services import profile_store
from backend.app.services.payee_history import payee_history
from backend.app.services.velocity import prepayment_velocity

IMPORTED_AT = datetime(2026, 9, 21, 12, 0)
T = "2026-09-21T12:00:00"
HEAD = "timestamp,counterparty,upi_id,amount,direction\n"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(profile_store, "DB_PATH", tmp_path / "x.db")
    profile_store._SCHEMA_DONE.clear()
    yield tmp_path / "x.db"
    profile_store._SCHEMA_DONE.clear()


def _csv(user, body, name="s.csv", head=HEAD):
    return import_statement(user, name, (head + body).encode(), imported_at=IMPORTED_AT)


# ── A-D: cross-source identity ───────────────────────────────────────────────

def test_A_same_payment_in_csv_without_ids_and_pdf_with_ids_is_one(db, tmp_path):
    from statement_pdf_fixtures import have_rupee_font, phonepe_like_pdf

    if not have_rupee_font():
        pytest.skip("DejaVu font (rupee glyph) not installed")
    _csv("u1", "2026-06-18 09:56:00,SYNTH PAYEE,9000000401@axl,1000,DEBIT\n")
    pdf = phonepe_like_pdf([dict(when=datetime(2026, 6, 18, 9, 56), name="SYNTH PAYEE",
                                 vpa="9000000401@axl", amount=1000.0, direction="DEBIT")], tmp_path)
    r = import_statement("u1", "p.pdf", pdf, imported_at=IMPORTED_AT)
    assert r.report["cross_source_duplicates"] == 1
    assert r.report["stored_transactions"] == 0
    assert payee_history("u1", "9000000401@axl", before=T).payment_count == 1


def test_A2_csv_with_utr_and_csv_with_other_id_scheme_is_one(db):
    _csv("u1", "2026-06-18 09:56:00,SYNTH PAYEE,9000000401@axl,1000,DEBIT,500000000001\n",
         head="timestamp,counterparty,upi_id,amount,direction,utr\n")
    r = _csv("u1", "2026-06-18 09:57:00,SYNTH PAYEE,9000000401@axl,1000,DEBIT,T2606180956001\n",
             name="b.csv", head="timestamp,counterparty,upi_id,amount,direction,transaction id\n")
    assert r.report["cross_source_duplicates"] == 1


def test_same_shaped_ids_that_differ_are_two_payments(db):
    head = "timestamp,counterparty,upi_id,amount,direction,utr\n"
    _csv("u1", "2026-06-18 09:56:00,SYNTH P,synth.p@ybl,100,DEBIT,500000000001\n", head=head)
    r = _csv("u1", "2026-06-18 09:56:00,SYNTH P,synth.p@ybl,100,DEBIT,500000000002\n",
             name="b.csv", head=head)
    assert r.report["cross_source_duplicates"] == 0
    assert payee_history("u1", "synth.p@ybl", before=T).payment_count == 2


def test_B_same_amount_and_time_different_upi_ids_are_two(db):
    _csv("u1", "2026-06-18 09:56:00,SHOP,synth.one@ybl,100,DEBIT\n")
    r = _csv("u1", "2026-06-18 09:56:00,SHOP,synth.two@ybl,100,DEBIT\n", name="b.csv")
    assert r.report["cross_source_duplicates"] == 0 and r.report["stored_transactions"] == 1


def test_C_same_payee_and_amount_different_times_are_two(db):
    _csv("u1", "2026-06-18 09:56:00,SHOP,synth.one@ybl,100,DEBIT\n")
    r = _csv("u1", "2026-06-18 18:10:00,SHOP,synth.one@ybl,100,DEBIT\n", name="b.csv")
    assert r.report["cross_source_duplicates"] == 0
    assert payee_history("u1", "synth.one@ybl", before=T).payment_count == 2


def test_D_identical_rows_within_one_source_are_kept_and_match_one_to_one(db):
    row = "2026-06-18 09:56:00,TEA,synth.tea@ybl,10,DEBIT\n"
    first = _csv("u1", row * 2)
    assert first.report["stored_transactions"] == 2
    # Another export showing THREE such teas: two are known, one is new.
    r = _csv("u1", row * 3, name="b.csv")
    assert r.report["cross_source_duplicates"] == 2
    assert r.report["stored_transactions"] == 1
    assert payee_history("u1", "synth.tea@ybl", before=T).payment_count == 3


def test_date_only_match_needs_to_be_unique_or_both_are_kept(db):
    _csv("u1", "2026-06-18 09:56:00,SHOP,synth.s@ybl,50,DEBIT\n"
               "2026-06-18 17:00:00,SHOP,synth.s@ybl,50,DEBIT\n")
    date_head = "Date,Name,UPI ID,Type,Amount\n"
    r = _csv("u1", "18/06/2026,SHOP,synth.s@ybl,Paid,50\n", name="b.csv", head=date_head)
    assert r.report["cross_source_duplicates"] == 0
    assert r.report["ambiguous_cross_source_kept"] == 1
    # A unique date-level counterpart is merged.
    _csv("u2", "2026-06-18 09:56:00,SHOP,synth.s@ybl,50,DEBIT\n")
    r2 = _csv("u2", "18/06/2026,SHOP,synth.s@ybl,Paid,50\n", name="b.csv", head=date_head)
    assert r2.report["cross_source_duplicates"] == 1


def test_a_name_is_never_matched_against_a_different_upi_id(db):
    _csv("u1", "2026-06-18 09:56:00,SHOP,synth.a@ybl,50,DEBIT\n")
    r = _csv("u1", "2026-06-18 09:56:00,SHOP,synth.b@ybl,50,DEBIT\n", name="b.csv")
    assert r.report["cross_source_duplicates"] == 0


# ── Timezone ─────────────────────────────────────────────────────────────────

def test_naive_statement_times_are_ist_and_never_shifted():
    assert to_app_naive("2026-06-18T23:30:00") == datetime(2026, 6, 18, 23, 30)


def test_aware_times_are_converted_to_ist():
    # 18:45 UTC is 00:15 IST the next day.
    assert to_app_naive("2026-06-18T18:45:00Z") == datetime(2026, 6, 19, 0, 15)
    assert to_app_naive("2026-06-19T00:15:00+05:30") == datetime(2026, 6, 19, 0, 15)


@pytest.mark.parametrize("zone", ["UTC", "America/New_York", "Asia/Kolkata"])
def test_now_is_ist_whatever_the_server_zone(monkeypatch, zone):
    if not hasattr(time, "tzset"):
        pytest.skip("tzset unavailable on this platform")
    monkeypatch.setenv("TZ", zone)
    time.tzset()
    try:
        expected = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        assert abs((now_app() - expected.replace(tzinfo=None)).total_seconds()) < 5
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def test_2330_then_0015_is_a_new_ist_day_for_velocity():
    history = pd.DataFrame({"timestamp": ["2026-06-18T23:30:00"]})
    v = prepayment_velocity(history, "2026-06-19T00:15:00")
    assert v["txns_today"] == 1.0                      # 23:30 was yesterday
    assert v["seconds_since_last_txn"] == 45 * 60
    assert v["txns_last_hour"] == 1.0
    # The same moment given as UTC by a client or a UTC server, normalised
    # the way every serving caller does (clock.app_iso) before velocity.py.
    from backend.app.core.clock import app_iso

    same = prepayment_velocity(history, app_iso("2026-06-18T18:45:00Z"))
    assert same == v


def test_payee_history_dates_use_ist(db):
    _csv("u1", "2026-06-18 23:30:00,NIGHT,synth.n@ybl,10,DEBIT\n")
    h = payee_history("u1", "synth.n@ybl", before="2026-06-18T18:45:00Z")   # 00:15 IST
    assert h.payment_count == 1
    assert h.last_payment_at == datetime(2026, 6, 18, 23, 30)
    assert h.days_since_last == pytest.approx(45 / (24 * 60))
    assert payee_history("u1", "synth.n@ybl", before="2026-06-18T17:59:00Z").payment_count == 0


def test_hour_feature_uses_ist():
    """backend/ml/features.py is a hashed V1 artefact and is not modified;
    serving hands it IST-normalised timestamps instead."""
    from backend.app.core.clock import app_iso
    from backend.ml.features import build_vector

    late = build_vector(amount=100.0, timestamp=app_iso("2026-06-18T18:45:00Z"),
                        profile={"median_amount": 100.0})
    assert late.values["is_night"] == 1.0            # 00:15 IST, not 18:45


def test_payer_context_normalises_utc_client_stamps(tmp_path, monkeypatch):
    import backend.main as main

    history = pd.DataFrame({"timestamp": ["2026-06-18T18:00:00Z"]})   # 23:30 IST
    monkeypatch.setattr(main, "get_all_transactions", lambda sender, limit: history)
    ctx = main._payer_context("u", pd.Timestamp("2026-06-19T00:15:00"))
    assert ctx["txns_today"] == 1.0 and ctx["seconds_since_last_txn"] == 45 * 60


def test_future_check_uses_ist_not_server_clock(db, monkeypatch):
    """A row at 00:10 IST today must not be 'future' on a UTC server where it
    is still yesterday evening locally."""
    import backend.app.core.clock as clock

    fixed = datetime(2026, 6, 19, 0, 20)             # IST now
    monkeypatch.setattr(clock, "now_app", lambda: fixed)
    r = import_statement("u1", "s.csv",
                         (HEAD + "2026-06-19 00:10:00,X,synth.x@ybl,5,DEBIT\n").encode())
    assert r.report["future_timestamp"] == 0
