"""Pre-payment velocity: the past, and nothing else.

A payment is being decided at time T. The payer's own ledger may inform that
decision; the ledger's future may not. The distinction is not pedantic - with
client-supplied timestamps the old rule (`|now - stamp| <= 3600`, and "same
calendar day") counted rows that had not happened yet, so a caller could
inflate their own velocity, and a feature that looks like behaviour quietly
became a feature that looks like the answer.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.app.services.velocity import NO_PRIOR_GAP_SECONDS, prepayment_velocity

T = pd.Timestamp("2026-09-19T14:00:00")


def ledger(*stamps: str) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": list(stamps),
                         "amount": [100.0] * len(stamps)})


def test_a_payment_ten_minutes_before_counts():
    v = prepayment_velocity(ledger("2026-09-19T13:50:00"), T)
    assert v["seconds_since_last_txn"] == 600.0
    assert v["txns_last_hour"] == 1.0
    assert v["txns_today"] == 2.0          # one earlier payment, plus this one


def test_a_payment_thirty_minutes_after_is_ignored():
    """The future cannot inform a decision being made now."""
    v = prepayment_velocity(ledger("2026-09-19T14:30:00"), T)
    assert v == {"seconds_since_last_txn": NO_PRIOR_GAP_SECONDS,
                 "txns_last_hour": 0.0, "txns_today": 1.0}


def test_a_payment_stamped_exactly_at_t_is_ignored():
    """That row is this payment, or a duplicate of it."""
    v = prepayment_velocity(ledger("2026-09-19T14:00:00"), T)
    assert v["txns_last_hour"] == 0.0
    assert v["seconds_since_last_txn"] == NO_PRIOR_GAP_SECONDS
    assert v["txns_today"] == 1.0


def test_no_previous_transaction_gives_the_documented_cold_start():
    for history in (None, pd.DataFrame(), ledger()):
        v = prepayment_velocity(history, T)
        assert v["seconds_since_last_txn"] == NO_PRIOR_GAP_SECONDS
        assert v["txns_last_hour"] == 0.0
        assert v["txns_today"] == 1.0


def test_multiple_transactions_before_t_are_all_counted():
    v = prepayment_velocity(ledger("2026-09-19T13:05:00", "2026-09-19T13:20:00",
                                   "2026-09-19T13:50:00", "2026-09-19T09:00:00"), T)
    assert v["seconds_since_last_txn"] == 600.0        # the closest one
    assert v["txns_last_hour"] == 3.0                  # 13:05, 13:20 and 13:50 are all inside the hour
    assert v["txns_today"] == 5.0                      # four earlier today, plus this one


def test_the_hour_window_is_half_open_at_both_ends():
    exactly_an_hour = prepayment_velocity(ledger("2026-09-19T13:00:00"), T)
    assert exactly_an_hour["txns_last_hour"] == 1.0, "T - 1h is inside the window"
    just_outside = prepayment_velocity(ledger("2026-09-19T12:59:59"), T)
    assert just_outside["txns_last_hour"] == 0.0


def test_yesterdays_payments_are_not_todays():
    v = prepayment_velocity(ledger("2026-09-18T23:59:00"), T)
    assert v["txns_today"] == 1.0
    assert v["txns_last_hour"] == 0.0
    assert v["seconds_since_last_txn"] > 3600


def test_a_future_row_cannot_shorten_the_gap():
    """The regression in one line: a later row must not become 'the last one'."""
    honest = prepayment_velocity(ledger("2026-09-19T13:00:00"), T)
    with_future = prepayment_velocity(
        ledger("2026-09-19T13:00:00", "2026-09-19T13:59:00", "2026-09-19T15:00:00"), T)
    assert with_future["seconds_since_last_txn"] == 60.0, "13:59 is legitimately the last"
    assert honest["seconds_since_last_txn"] == 3600.0
    only_future = prepayment_velocity(ledger("2026-09-19T15:00:00", "2026-09-19T16:00:00"), T)
    assert only_future["seconds_since_last_txn"] == NO_PRIOR_GAP_SECONDS


def test_clock_skew_cannot_produce_the_strongest_possible_signal():
    v = prepayment_velocity(ledger("2026-09-19T13:59:59.5"), T)
    assert v["seconds_since_last_txn"] >= 1.0


# ── the endpoint that was passing nothing ───────────────────────────────────

def test_payee_check_now_feeds_the_model_real_velocity():
    """Before this change /payee/check passed no velocity at all, so the model
    received the parameter defaults on every request."""
    import backend.app.services.payee_check as pc

    captured: dict = {}
    original = pc.build_vector

    def spy(**kwargs):
        vector = original(**kwargs)
        captured.update(vector.values)
        return vector

    pc.build_vector = spy
    try:
        history = ledger("2026-09-19T13:50:00", "2026-09-19T13:20:00",
                         "2026-09-19T09:00:00",
                         "2026-09-19T14:30:00",      # future - must not count
                         "2026-09-19T14:00:00")      # this payment - must not count
        pc.check_payee(payload="stranger.payee@ybl", payer_id="u1", amount=4500.0,
                       profile={"transaction_count": 288, "debit_count": 240,
                                "median_amount": 222.0, "avg_amount": 1523.0,
                                "max_amount": 18500.0, "most_active_hour": 19,
                                "average_daily_transactions": 3.0},
                       history=history, timestamp=T.isoformat())
    finally:
        pc.build_vector = original

    assert captured["seconds_since_last_txn"] == 600.0
    assert captured["txns_last_hour"] == 2.0
    assert captured["txns_today_over_avg"] == pytest.approx(4.0 / 3.0)


def test_an_explicit_velocity_from_the_caller_still_wins():
    import backend.app.services.payee_check as pc

    captured: dict = {}
    original = pc.build_vector

    def spy(**kwargs):
        vector = original(**kwargs)
        captured.update(vector.values)
        return vector

    pc.build_vector = spy
    try:
        pc.check_payee(payload="stranger.payee@ybl", payer_id="u1", amount=90.0,
                       profile={"transaction_count": 288, "debit_count": 240,
                                "median_amount": 222.0, "avg_amount": 610.0,
                                "most_active_hour": 19, "average_daily_transactions": 3.0},
                       history=ledger("2026-09-19T13:50:00"), timestamp=T.isoformat(),
                       velocity={"seconds_since_last_txn": 42.0, "txns_last_hour": 7.0,
                                 "txns_today": 9.0})
    finally:
        pc.build_vector = original

    assert captured["seconds_since_last_txn"] == 42.0
    assert captured["txns_last_hour"] == 7.0


def test_the_predict_path_also_excludes_the_future(monkeypatch):
    """backend.main._payer_context had the same defect and the same fix."""
    import backend.main as main

    rows = pd.DataFrame({
        "timestamp": ["2026-09-19T13:50:00", "2026-09-19T14:30:00", "2026-09-19T14:00:00",
                      "2026-09-19T09:00:00"],
        "amount": [1.0, 2.0, 3.0, 4.0],
    })
    monkeypatch.setattr(main, "get_all_transactions", lambda sender, limit=200: rows)
    context = main._payer_context("someone@okaxis", T.isoformat())
    assert context["seconds_since_last_txn"] == 600.0
    assert context["txns_last_hour"] == 1.0
    assert context["txns_today"] == 3.0          # 13:50 and 09:00, plus this payment
