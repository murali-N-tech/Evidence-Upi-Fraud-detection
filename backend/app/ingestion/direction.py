"""Strict transaction-direction reading.

Exact phrase matching after case/spacing/punctuation cleanup. An unrecognised
value - including a misspelling such as "DEBET" - is None, never a guess: a
wrong DEBIT puts money that came IN into the spending baseline and invents an
outgoing relationship with the sender.
"""

from __future__ import annotations

import re
from typing import Optional

from backend.app.ingestion.canonical import Direction

_DEBIT = {
    "debit", "dr", "d/r", "debited", "withdrawal", "withdraw", "withdrawn",
    "sent", "paid", "paid to", "payment", "payment to", "transfer to",
    "transferred to", "money sent", "outgoing", "out", "spent",
}
_CREDIT = {
    "credit", "cr", "c/r", "credited", "deposit", "deposited", "received",
    "received from", "money received", "incoming", "in", "refund", "refunded",
    "cashback",
}


def _clean(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\s_]+", " ", text)
    return text.strip(" .:-")


def parse_direction(value: object) -> Optional[Direction]:
    if value is None:
        return None
    text = _clean(value)
    if not text:
        return None
    if text in _DEBIT:
        return Direction.DEBIT
    if text in _CREDIT:
        return Direction.CREDIT
    return None


def direction_from_amount_columns(debit: Optional[float],
                                  credit: Optional[float]) -> Optional[Direction]:
    """Separate Debit/Credit (Withdrawal/Deposit) columns. Exactly one must
    carry a positive amount; both or neither is ambiguous."""
    has_d = debit is not None and debit > 0
    has_c = credit is not None and credit > 0
    if has_d and not has_c:
        return Direction.DEBIT
    if has_c and not has_d:
        return Direction.CREDIT
    return None


def direction_from_signed_amount(amount: Optional[float]) -> Optional[Direction]:
    """A single signed amount column: negative is money out."""
    if amount is None or amount == 0:
        return None
    return Direction.DEBIT if amount < 0 else Direction.CREDIT
