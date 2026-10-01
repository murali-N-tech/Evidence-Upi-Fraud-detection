"""payee_history(): the one authoritative answer to "has this payer paid this
payee before T, and what does that relationship look like?"

Every consumer reads it - /payee/check (relationship family, payee status),
/payee/confirm, the V1 feature builder (payee_is_new), the V2 feature builder
(prior_payments_to_payee, days_since_first_payment_to_payee) and the statement
import summary - so no two of them can disagree. It used to be answered in
three places from three different tables, and the statement importer wrote
none of them.

Temporal rule, mandatory: only payments with timestamp STRICTLY before T are
previous payments. A payment at T is the one being made; after T is the
future. A payment whose source gave no date counts as "paid before" only if
it was observed (imported) before T, and never contributes a date.

Privacy: this reads the requesting payer's OWN ledger rows only. Other
payers are visible solely through assess_payee()'s aggregates.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from backend.app.services.vpa import normalise_vpa

# At or above this many prior payments the relationship is "frequent".
PAID_OFTEN_MIN = 5


class PayeeStatus(str, Enum):
    PAID_BEFORE = "PAID_BEFORE"
    PAID_ONCE = "PAID_ONCE"
    PAID_OFTEN = "PAID_OFTEN"
    NEW_TO_USER_BUT_OBSERVED = "NEW_TO_USER_BUT_OBSERVED"
    UNKNOWN = "UNKNOWN"
    IDENTITY_UNAVAILABLE = "IDENTITY_UNAVAILABLE"
    # History exists for a PARTIAL identifier consistent with this address
    # (e.g. SBI's 10-character VPA field), or for a matching NAME only. Both
    # are evidence of a past counterparty, neither proves this exact VPA.
    PARTIAL_IDENTITY = "PARTIAL_IDENTITY"
    NAME_ONLY_HISTORY = "NAME_ONLY_HISTORY"
    HISTORY_UNAVAILABLE_DUE_TO_PARTIAL_IMPORT = "HISTORY_UNAVAILABLE_DUE_TO_PARTIAL_IMPORT"


def _moment(value: Any) -> Optional[datetime]:
    """Naive IST (backend/app/core/clock.py)."""
    from backend.app.core.clock import to_app_naive

    return to_app_naive(value)


@dataclass
class PayeeHistory:
    upi_id: str
    before: Optional[datetime]
    payment_count: int = 0             # all previous payments, dated or not
    dated_payment_count: int = 0
    first_payment_at: Optional[datetime] = None
    last_payment_at: Optional[datetime] = None
    total_amount: float = 0.0
    provenance: dict[str, int] = field(default_factory=dict)

    @property
    def paid_before(self) -> bool:
        return self.payment_count > 0

    @property
    def relationship_status(self) -> Optional[PayeeStatus]:
        if self.payment_count <= 0:
            return None
        if self.payment_count == 1:
            return PayeeStatus.PAID_ONCE
        if self.payment_count >= PAID_OFTEN_MIN:
            return PayeeStatus.PAID_OFTEN
        return PayeeStatus.PAID_BEFORE

    def _days(self, moment: Optional[datetime]) -> Optional[float]:
        if moment is None or self.before is None:
            return None
        return max(0.0, (self.before - moment).total_seconds() / 86_400.0)

    @property
    def days_since_first(self) -> Optional[float]:
        return self._days(self.first_payment_at)

    @property
    def days_since_last(self) -> Optional[float]:
        return self._days(self.last_payment_at)

    # ── Model inputs. Both feature builders read these and nothing else. ──
    def v1_inputs(self) -> dict[str, Any]:
        return {"payer_seen_payee_before": self.paid_before}

    def v2_inputs(self) -> dict[str, Any]:
        return {
            "payer_seen_payee_before": self.paid_before,
            "prior_payments_to_payee": self.payment_count,
            "days_since_first_payment_to_payee": self.days_since_first,
        }

    def as_dict(self) -> dict[str, Any]:
        """For the requesting payer only: their own exact history."""
        def iso(m: Optional[datetime]) -> Optional[str]:
            return m.isoformat(timespec="minutes") if m else None

        days_first, days_last = self.days_since_first, self.days_since_last
        return {
            "upi_id": self.upi_id,
            "history_available": True,
            "paid_before": self.paid_before,
            "payment_count": self.payment_count,
            "dated_payment_count": self.dated_payment_count,
            "first_payment_at": iso(self.first_payment_at),
            "last_payment_at": iso(self.last_payment_at),
            "total_amount": round(self.total_amount, 2),
            "days_since_first_payment": round(days_first, 1) if days_first is not None else None,
            "days_since_last_payment": round(days_last, 1) if days_last is not None else None,
            "relationship_status": self.relationship_status.value if self.relationship_status else None,
            "sources": dict(self.provenance),
        }


def payee_history(payer_id: str, upi_id: str, before: Any = None,
                  conn: Optional[sqlite3.Connection] = None) -> PayeeHistory:
    """This payer's payments to this payee strictly before `before`.

    `before=None` means no time bound (all recorded payments); every
    risk-scoring caller passes the decision time.
    """
    from backend.app.services.payee_reputation import connect

    key = normalise_vpa(upi_id)
    bound = _moment(before)
    history = PayeeHistory(upi_id=key, before=bound)
    if not payer_id or not key:
        return history

    def earlier(moment: Optional[datetime]) -> bool:
        return moment is not None and (bound is None or moment < bound)

    own = conn is None
    conn = conn or connect()
    try:
        rows = conn.execute(
            "SELECT amount, at, provenance, in_aggregate, created_at "
            "FROM payment_observations WHERE payer_id = ? AND vpa = ?",
            (payer_id, key),
        ).fetchall()
        legacy = conn.execute(
            "SELECT payments, first_at, last_at FROM payee_payers "
            "WHERE payer_id = ? AND vpa = ?", (payer_id, key),
        ).fetchone()
    finally:
        if own:
            conn.close()

    counts: dict[str, int] = {}
    aggregated = 0
    for r in rows:
        if r["in_aggregate"]:
            aggregated += 1
        at = _moment(r["at"])
        if at is not None:
            if not earlier(at):
                continue
            history.dated_payment_count += 1
            history.first_payment_at = min(filter(None, [history.first_payment_at, at]))
            history.last_payment_at = max(filter(None, [history.last_payment_at, at]))
        elif not earlier(_moment(r["created_at"])):
            # Undated, and not even known to have been imported before T.
            continue
        history.payment_count += 1
        history.total_amount += float(r["amount"] or 0)
        counts[r["provenance"]] = counts.get(r["provenance"], 0) + 1

    # Edges recorded before the ledger existed (payee_payers rows with no
    # ledger rows behind them). Counted, attributed to OTHER_OBSERVED, and
    # dated only when their recorded first/last time is before T.
    if legacy is not None:
        unledgered = int(legacy["payments"] or 0) - aggregated
        first = _moment(legacy["first_at"])
        if unledgered > 0 and earlier(first):
            history.payment_count += unledgered
            counts["OTHER_OBSERVED"] = counts.get("OTHER_OBSERVED", 0) + unledgered
            history.first_payment_at = min(filter(None, [history.first_payment_at, first]))
            last = _moment(legacy["last_at"])
            if earlier(last):
                history.last_payment_at = max(filter(None, [history.last_payment_at, last]))

    history.provenance = counts
    return history


# ── Why there is no history, when there is none ─────────────────────────────

class HistoryState(str, Enum):
    HISTORY_FOUND = "HISTORY_FOUND"
    NO_HISTORY = "NO_HISTORY"
    PARTIAL_IMPORT = "PARTIAL_IMPORT"
    IDENTITY_UNAVAILABLE = "IDENTITY_UNAVAILABLE"
    IMPORT_ERROR = "IMPORT_ERROR"
    NOT_SIGNED_IN = "NOT_SIGNED_IN"
    PARTIAL_IDENTITY = "PARTIAL_IDENTITY"


STATE_MESSAGES = {
    HistoryState.HISTORY_FOUND: "You have paid this payee before.",
    HistoryState.NO_HISTORY: "No payment history was found for this payee.",
    HistoryState.PARTIAL_IMPORT: (
        "Statement was partially processed. Some transactions could not be analyzed, "
        "so a previous payment to this payee may not be visible."),
    HistoryState.IDENTITY_UNAVAILABLE: (
        "The statement contains a transaction, but a usable UPI identifier could not "
        "be extracted, so it cannot be matched to this address."),
    HistoryState.PARTIAL_IDENTITY: (
        "Historical payment found with a matching bank narration fragment, but the "
        "statement does not contain the complete VPA, so exact address matching "
        "cannot be verified."),
    HistoryState.IMPORT_ERROR: (
        "Statement processing failed, so your payment history could not be checked. "
        "Upload the statement again."),
    HistoryState.NOT_SIGNED_IN: (
        "No signed-in payer, so your own history with this payee cannot be read."),
}


def _name_key(name: Optional[str]) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def classify(
    payer_id: Optional[str],
    key: Optional[str],
    display_name: Optional[str],
    history: Optional[PayeeHistory],
    payee_known_globally: bool,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Payee status (7 values) and history state (why), from stored facts.

    Exact matching only. A name-only statement row is matched to the scanned
    name by exact normalised equality, and even then it only explains why
    there is no history - it is never counted as a previous payment.
    """
    from backend.app.services.payee_reputation import connect
    from backend.app.services.profile_store import latest_import

    is_address = bool(key) and not str(key).startswith("name:") and "@" in str(key)

    if not payer_id:
        state = HistoryState.NOT_SIGNED_IN
        status = PayeeStatus.IDENTITY_UNAVAILABLE if not is_address else (
            PayeeStatus.NEW_TO_USER_BUT_OBSERVED if payee_known_globally else PayeeStatus.UNKNOWN)
        return _result(status, state)
    if not is_address:
        return _result(PayeeStatus.IDENTITY_UNAVAILABLE, HistoryState.IDENTITY_UNAVAILABLE,
                       "No usable UPI ID could be read from this input, so payment "
                       "history cannot be looked up.")
    if history is not None and history.paid_before:
        return _result(history.relationship_status, HistoryState.HISTORY_FOUND)

    own = conn is None
    conn = conn or connect()
    try:
        before = history.before.isoformat() if history and history.before else None
        time_clause = " AND (timestamp IS NULL OR ? IS NULL OR timestamp < ?)"
        this_payee_partial = conn.execute(
            "SELECT COUNT(*) FROM statement_transactions WHERE user_id = ? "
            "AND counterparty_upi_id = ? AND direction IS NULL" + time_clause,
            (payer_id, key, before, before),
        ).fetchone()[0]
        unattributable = conn.execute(
            "SELECT COUNT(*) FROM statement_transactions WHERE user_id = ? AND ("
            "COALESCE(validation_reasons, '') LIKE '%upi_id_malformed%' OR "
            "(counterparty_upi_id IS NULL AND direction IS NULL AND validation_status IS NOT NULL))",
            (payer_id,),
        ).fetchone()[0]
        # Partial identifiers (never matched exactly) that are CONSISTENT with
        # this address: the scanned VPA begins with the printed fragment.
        partial_rows = conn.execute(
            "SELECT counterparty_identifier, source_provider, COUNT(*) AS n, "
            "MIN(substr(timestamp, 1, 10)) AS first_day, MAX(substr(timestamp, 1, 10)) AS last_day "
            "FROM statement_transactions WHERE user_id = ? AND direction = 'DEBIT' "
            "AND counterparty_upi_id IS NULL AND counterparty_identifier IS NOT NULL "
            "AND identifier_type IN ('PARTIAL_VPA_FRAGMENT', 'NON_VPA_IDENTIFIER')"
            + time_clause + " GROUP BY counterparty_identifier, source_provider",
            (payer_id, before, before),
        ).fetchall()
        name_rows = []
        if display_name:
            name_rows = conn.execute(
                "SELECT counterparty_name FROM statement_transactions WHERE user_id = ? "
                "AND counterparty_upi_id IS NULL AND direction = 'DEBIT' "
                "AND counterparty_name IS NOT NULL",
                (payer_id,),
            ).fetchall()
        latest = latest_import(payer_id, conn=conn)
    except sqlite3.OperationalError:
        this_payee_partial = unattributable = 0
        name_rows, latest, partial_rows = [], None, []
    finally:
        if own:
            conn.close()

    wanted = _name_key(display_name)
    # Exact name, or a bank's truncated name field that is a prefix of the
    # displayed name ("CHINTHAD" for "CHINTHADA M..."), at least 6 letters.
    name_match = bool(wanted) and any(
        (k := _name_key(r[0])) and (k == wanted or (len(k) >= 6 and wanted.startswith(k)))
        for r in name_rows)
    partial = _consistent_partial(str(key), partial_rows)

    if this_payee_partial:
        return _result(PayeeStatus.HISTORY_UNAVAILABLE_DUE_TO_PARTIAL_IMPORT,
                       HistoryState.PARTIAL_IMPORT,
                       "Your statement has a transaction with this payee whose direction "
                       "(paid or received) could not be read, so it is not counted as a "
                       "previous payment.")
    if partial is not None:
        fragment, provider, count, first_day, last_day = partial
        bank = {"sbi": "SBI", "bank_of_baroda": "Bank of Baroda"}.get(provider or "", "bank")
        when = (f" on {last_day}" if first_day == last_day else
                f" between {first_day} and {last_day}") if last_day else ""
        return _result(
            PayeeStatus.PARTIAL_IDENTITY, HistoryState.PARTIAL_IDENTITY,
            f"{count} historical payment(s){when} found with matching bank narration "
            f"fragment '{fragment}', but the uploaded {bank} statement does not contain "
            f"the complete VPA, so exact address matching cannot be verified.")
    if name_match:
        return _result(PayeeStatus.NAME_ONLY_HISTORY, HistoryState.IDENTITY_UNAVAILABLE)
    status = PayeeStatus.NEW_TO_USER_BUT_OBSERVED if payee_known_globally else PayeeStatus.UNKNOWN
    if latest and latest["status"] == "FAILED":
        return _result(status, HistoryState.IMPORT_ERROR)
    if unattributable:
        return _result(status, HistoryState.PARTIAL_IMPORT)
    return _result(status, HistoryState.NO_HISTORY)


def _result(status: PayeeStatus, state: HistoryState,
            message: Optional[str] = None) -> dict[str, Any]:
    return {"payee_status": status.value, "history_state": state.value,
            "message": message or STATE_MESSAGES[state]}


def build_v2_vector(history: PayeeHistory, **kwargs: Any):
    """V2 feature vector whose relationship inputs come from payee_history().

    Lives here, not in backend/ml/features_v2.py, because that file is a
    hashed V2 research artefact. V2 is not served; this is the path it will
    use when it is, so V1 and V2 cannot read contradictory history.
    """
    from backend.ml.features_v2 import build_vector_v2

    return build_vector_v2(**history.v2_inputs(), **kwargs)


MIN_FRAGMENT_LENGTH = 6


def _consistent_partial(key: str, rows) -> Optional[tuple]:
    """A stored partial identifier the scanned address is consistent with.

    Consistent means the address starts with the fragment as printed (a
    trailing "@" kept: "swiggyupi@" fits "swiggyupi@axisbank" but not
    "swiggyupindia@..."). This only EXPLAINS history - it never sets
    paid_before and never reaches the model as a relationship.
    """
    best = None
    for r in rows or []:
        fragment = (r[0] or "").strip().lower()
        core = fragment.rstrip("@")
        if len(core) < MIN_FRAGMENT_LENGTH:
            continue
        if "@" in fragment and not fragment.endswith("@"):
            fits = key.startswith(fragment)
        elif fragment.endswith("@"):
            fits = key.startswith(fragment)
        else:
            fits = key.split("@", 1)[0].startswith(core)
        if fits and (best is None or r[2] > best[2]):
            best = (r[0], r[1], int(r[2]), r[3], r[4])
    return best
