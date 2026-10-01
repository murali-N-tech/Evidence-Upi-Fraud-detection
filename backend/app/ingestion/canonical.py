"""The canonical transaction every provider adapter produces.

Every field is nullable. An adapter that cannot read a value leaves it None;
nothing downstream invents a timestamp, a direction, a transaction ID or a UPI
ID to fill the gap. The validator decides what a row with gaps can still be
used for.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Direction(str, Enum):
    DEBIT = "DEBIT"     # money out of the importing user's account
    CREDIT = "CREDIT"   # money in


class Status(str, Enum):
    VALID = "VALID"          # usable everywhere it is relevant
    PARTIAL = "PARTIAL"      # stored and counted, excluded where a gap matters
    REJECTED = "REJECTED"    # not a usable transaction (no amount); counted, not stored


IDENTIFIER_TYPES = ("COMPLETE_VPA", "PARTIAL_VPA_FRAGMENT", "NON_VPA_IDENTIFIER",
                    "NAME_ONLY", "UNKNOWN")


class Provenance(str, Enum):
    """Where an observed payment came from. None of these means "verified by
    a PSP/NPCI" - this system has no such feed and must never claim one."""

    SYSTEM_CONFIRMED = "SYSTEM_CONFIRMED"      # the payer confirmed it in this app
    STATEMENT_OBSERVED = "STATEMENT_OBSERVED"  # read from a statement the payer uploaded
    OTHER_OBSERVED = "OTHER_OBSERVED"          # legacy/bootstrapped aggregates of unknown origin


@dataclass
class CanonicalTransaction:
    timestamp: Optional[datetime] = None
    time_known: bool = False
    amount: Optional[float] = None
    direction: Optional[Direction] = None
    counterparty_name: Optional[str] = None
    counterparty_upi_id: Optional[str] = None
    transaction_id: Optional[str] = None
    description: Optional[str] = None
    status_text: Optional[str] = None          # SUCCESS/FAILED as printed
    source_provider: Optional[str] = None      # "generic_csv", "google_pay_like", ...
    source_format: Optional[str] = None        # "csv" | "pdf"
    source_row: Optional[int] = None           # position in the source file
    confidence: float = 1.0                    # adapter's confidence in the row
    # Filled by validation / dedupe.
    status: Status = Status.VALID
    reasons: list[str] = field(default_factory=list)
    raw_upi_text: Optional[str] = None         # what the source said, before normalisation
    fingerprint: Optional[str] = None
    # Bank-statement detail (backend/app/ingestion/providers). All optional;
    # an adapter that cannot read a field leaves it None.
    timestamp_confidence: Optional[str] = None     # HIGH (date+time) / MEDIUM (date) / LOW
    transaction_type: Optional[str] = None         # e.g. UPI, NEFT, IMPS, INTEREST, OPENING
    payment_method: Optional[str] = None           # rail: UPI / NEFT / IMPS / CARD / ...
    balance: Optional[float] = None
    counterparty_bank_code: Optional[str] = None   # IFSC bank prefix as printed (BARB, UTIB)
    upi_fragment: Optional[str] = None             # a VPA the statement cut short - never an identity
    # What the narration printed in the counterparty-identifier position,
    # kept whole even when it is not a complete VPA, and what kind it is.
    counterparty_identifier: Optional[str] = None
    identifier_type: Optional[str] = None          # see IDENTIFIER_TYPES
    name_source: Optional[str] = None              # e.g. BANK_NARRATION
    name_confidence: Optional[float] = None
    upi_confidence: Optional[float] = None
    raw_narration: Optional[str] = None            # the PDF lines as printed, "|"-separated
    counterparty_display_label: Optional[str] = None
    display_label_source: Optional[str] = None     # e.g. VPA_LOCAL_PART - never a name
    narration: Optional[str] = None                # reconstructed; kept in memory, not stored
    extraction_confidence: Optional[float] = None
    extraction_source: Optional[str] = None        # e.g. "bank_of_baroda:narration_grammar"

    @property
    def resolved_identifier_type(self) -> str:
        """COMPLETE_VPA / PARTIAL_VPA_FRAGMENT / NON_VPA_IDENTIFIER /
        NAME_ONLY / UNKNOWN. Only COMPLETE_VPA may identify a payee exactly."""
        if self.counterparty_upi_id:
            return "COMPLETE_VPA"
        if self.identifier_type in ("PARTIAL_VPA_FRAGMENT", "NON_VPA_IDENTIFIER") \
                and self.counterparty_identifier:
            return self.identifier_type
        if self.counterparty_name:
            return "NAME_ONLY"
        return "UNKNOWN"

    @property
    def transaction_date(self):
        return self.timestamp.date() if self.timestamp else None

    @property
    def transaction_time(self):
        return self.timestamp.time() if (self.timestamp and self.time_known) else None

    @property
    def identity_state(self) -> str:
        """EXACT_VPA / VPA_ONLY / NAME_ONLY / UNKNOWN.

        Only a complete, validated UPI ID establishes identity. A name - or a
        truncated VPA fragment - never does.
        """
        if self.counterparty_upi_id and self.counterparty_name:
            return "EXACT_VPA"
        if self.counterparty_upi_id:
            return "VPA_ONLY"
        if self.counterparty_name:
            return "NAME_ONLY"
        return "UNKNOWN"

    @property
    def is_outgoing_payment(self) -> bool:
        return self.direction is Direction.DEBIT and self.status is not Status.REJECTED

    def as_store_row(self) -> dict[str, Any]:
        """The dict profile_store / the behaviour profile read."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "time_known": bool(self.time_known and self.timestamp is not None),
            "amount": self.amount,
            # A display label (e.g. a VPA local part) may label the row; it is
            # never written into counterparty_name.
            "merchant": (self.counterparty_name or self.counterparty_display_label
                         or self.description or None),
            "upi_id": self.counterparty_upi_id,
            "status": (self.status_text or "SUCCESS").upper(),
            "reference_number": self.transaction_id,
            # Direction is stored as it was read. UNKNOWN is its own value:
            # defaulting it to DEBIT is how credits used to become spending.
            "txn_type": self.direction.value if self.direction else "UNKNOWN",
            "direction": self.direction.value if self.direction else None,
            "counterparty_name": self.counterparty_name,
            "counterparty_upi_id": self.counterparty_upi_id,
            "transaction_id": self.transaction_id,
            "source_provider": self.source_provider,
            "source_format": self.source_format,
            "confidence": self.confidence,
            "validation_status": self.status.value,
            "validation_reasons": ",".join(self.reasons) or None,
            "fingerprint": self.fingerprint,
            "identity_state": self.identity_state,
            "timestamp_confidence": self.timestamp_confidence,
            "payment_method": self.payment_method,
            "transaction_type": self.transaction_type,
            "counterparty_bank_code": self.counterparty_bank_code,
            "balance": self.balance,
            "extraction_source": self.extraction_source,
            "counterparty_identifier": self.counterparty_identifier,
            "identifier_type": self.resolved_identifier_type,
            "name_source": self.name_source,
            "name_confidence": self.name_confidence,
            "upi_confidence": self.upi_confidence,
            # The reconstructed narration: the user's own audit trail. Stored
            # per user like every other statement field; never logged and
            # never returned by the upload endpoint.
            "narration": self.narration,
        }

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        d["direction"] = self.direction.value if self.direction else None
        d["status"] = self.status.value
        return d
