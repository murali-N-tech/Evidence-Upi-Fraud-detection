"""Row validation. Every row gets a status and reasons; none is dropped silently."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from backend.app.ingestion.canonical import CanonicalTransaction, Status

# A statement row dated after the import is a parsing error or a tampered
# file; either way it cannot be a payment that happened before the upload.
# One day of tolerance absorbs timezone differences between the statement's
# wall clock and the server's.
FUTURE_TOLERANCE = timedelta(days=1)


def validate(tx: CanonicalTransaction, imported_at: datetime) -> CanonicalTransaction:
    reasons: list[str] = list(tx.reasons)

    if tx.amount is None or tx.amount <= 0:
        reasons.append("amount_unreadable")
        tx.status = Status.REJECTED
        tx.reasons = reasons
        return tx

    partial = False
    if tx.timestamp is None:
        reasons.append("timestamp_unreadable")
        partial = True
    elif tx.timestamp > imported_at + FUTURE_TOLERANCE:
        reasons.append("timestamp_in_future")
        partial = True
    if tx.direction is None:
        reasons.append("direction_unknown")
        partial = True
    if tx.raw_upi_text and tx.counterparty_upi_id is None:
        reasons.append("upi_id_malformed")
        partial = True
    if tx.counterparty_upi_id is None and not tx.raw_upi_text \
            and "identifier_not_complete_vpa" not in reasons:
        # Not a defect of the row - many statements print names only - but
        # the payee cannot be identified by address, and the report says so.
        reasons.append("upi_id_unavailable")

    tx.status = Status.PARTIAL if partial else Status.VALID
    tx.reasons = reasons
    return tx


def usable_timestamp(tx: CanonicalTransaction) -> Optional[datetime]:
    """The timestamp temporal features may read, or None."""
    if tx.timestamp is None or "timestamp_in_future" in tx.reasons:
        return None
    return tx.timestamp
