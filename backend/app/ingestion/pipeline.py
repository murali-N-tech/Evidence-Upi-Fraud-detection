"""Statement import: adapter -> canonical -> validate -> dedupe -> store ->
payee history / graph -> profile -> import report.

One pipeline for every provider and format. The report carries counts only;
no transaction content leaves this module in it.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.app.ingestion.canonical import (
    CanonicalTransaction,
    Direction,
    Provenance,
    Status,
)
from backend.app.ingestion.dedupe import assign_fingerprints
from backend.app.ingestion.validate import usable_timestamp, validate

_NOT_COMPLETED = {"FAILED", "FAILURE", "DECLINED", "PENDING", "REVERSED", "CANCELLED", "CANCELED"}


class ImportError_(Exception):
    """The file could not be processed at all. Message is user-safe."""


@dataclass
class ImportResult:
    statement_id: Optional[str]
    status: str                      # COMPLETE / PARTIAL / FAILED
    report: dict[str, Any] = field(default_factory=dict)
    profile: Optional[dict[str, Any]] = None


def _parse(filename: str, content: bytes) -> tuple[str, dict[str, Any]]:
    ext = Path(filename or "").suffix.lower()
    if ext in (".csv", ".tsv", ".txt"):
        from backend.app.ingestion.adapters.csv_generic import CsvLayoutError, parse_csv

        try:
            return "csv", parse_csv(content)
        except CsvLayoutError as exc:
            raise ImportError_(str(exc)) from exc
    if ext == ".pdf":
        from backend.app.ingestion.adapters.pdf_generic import parse_pdf

        return "pdf", parse_pdf(content, filename)
    raise ValueError("Unsupported file type. Upload a PDF or CSV statement.")


def completed_payment(tx: CanonicalTransaction) -> bool:
    return (tx.status_text or "SUCCESS").upper() not in _NOT_COMPLETED


def eligible_observation(tx: CanonicalTransaction) -> bool:
    """May this row become an outgoing payer->payee observation?"""
    return (
        tx.status is not Status.REJECTED
        and tx.direction is Direction.DEBIT
        and tx.transaction_type != "REVERSAL"
        # Only a COMPLETE VPA creates an exact payee relationship. Partial
        # and non-VPA identifiers stay in the user's own transaction history.
        and tx.counterparty_upi_id is not None
        and "timestamp_in_future" not in tx.reasons
        and completed_payment(tx)
    )


def import_statement(
    user_id: str,
    filename: str,
    content: bytes,
    imported_at: Optional[datetime] = None,
    max_rows: int = 20_000,
) -> ImportResult:
    from backend.app.services import profile_store as store
    from backend.app.services.payee_reputation import connect, record_payment
    from backend.app.services.statement_parser import generate_behavior_profile

    from backend.app.core.clock import now_app

    # IST, not the server's zone: the future-date check compares this with
    # statement wall times (clock.py).
    imported_at = imported_at or now_app()
    statement_id = f"stmt_{uuid.uuid4().hex[:8]}"

    fmt, parsed = _parse(filename, content)
    rows: list[CanonicalTransaction] = parsed["transactions"]
    warnings: list[str] = list(parsed.get("warnings") or [])
    detected = len(rows)
    if detected > max_rows:
        warnings.append(f"Only the first {max_rows} rows were processed.")
        rows = rows[:max_rows]

    for tx in rows:
        validate(tx, imported_at)
    repeated_in_file = assign_fingerprints(user_id, rows)

    storable = [tx for tx in rows if tx.status is not Status.REJECTED]

    # Same payments already stored from a DIFFERENT statement (e.g. the CSV
    # and the PDF export of one month). See dedupe.cross_source_duplicates
    # for the decision rule; ambiguous cases are kept, not merged.
    from backend.app.ingestion.dedupe import cross_source_duplicates

    probe = connect()
    try:
        crossed, ambiguous = cross_source_duplicates(probe, user_id, storable, statement_id)
    finally:
        probe.close()
    cross_dupes = [tx for i, tx in enumerate(storable) if i in crossed]
    storable = [tx for i, tx in enumerate(storable) if i not in crossed]

    inserted = store.insert_statement_rows(
        user_id, statement_id, fmt, [tx.as_store_row() for tx in storable])
    new_rows = [storable[i] for i in inserted]

    # Outgoing observations: payee history, reputation aggregates and the
    # graph, all through record_payment's one transaction per import. Only
    # rows stored for the first time are offered, and each carries its
    # fingerprint as the observation key, so a re-import counts nothing.
    recorded_timed = recorded_untimed = 0
    payees: set[str] = set()
    conn = connect()
    try:
        conn.execute("BEGIN")
        for tx in new_rows:
            if not eligible_observation(tx):
                continue
            moment = usable_timestamp(tx)
            counted = record_payment(
                tx.counterparty_upi_id, payer_id=user_id, amount=tx.amount,
                at=moment.isoformat() if moment else None,
                display_name=tx.counterparty_name, conn=conn, autocommit=False,
                provenance=Provenance.STATEMENT_OBSERVED.value,
                observation_key=tx.fingerprint, timed=moment is not None,
            )
            if counted:
                payees.add(tx.counterparty_upi_id)
                if moment is not None:
                    recorded_timed += 1
                else:
                    recorded_untimed += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # The profile is rebuilt from the user's complete stored history, not
    # just this file.
    profile = None
    if storable:
        history = store.get_user_transactions(user_id)
        profile = generate_behavior_profile(history.to_dict(orient="records"))
        store.save_behavior_profile(user_id=user_id, source_type=fmt, profile=profile,
                                    transaction_count=int(len(history)))

    reasons = Counter(r for tx in rows for r in tx.reasons)
    statuses = Counter(tx.status.value for tx in rows)
    readable = storable + cross_dupes          # every row that was read and kept
    stamps = [tx.timestamp for tx in readable if usable_timestamp(tx)]
    debits_without_id = sum(
        1 for tx in readable
        if tx.direction is Direction.DEBIT and tx.counterparty_upi_id is None)
    report = {
        "source_format": fmt,
        "source_provider": parsed.get("provider") or (rows[0].source_provider if rows else None),
        "total_rows_detected": detected,
        "valid_transactions": statuses.get("VALID", 0),
        "partial_transactions": statuses.get("PARTIAL", 0),
        "rejected_transactions": statuses.get("REJECTED", 0),
        "duplicate_transactions": len(storable) - len(new_rows) + len(cross_dupes),
        "cross_source_duplicates": len(cross_dupes),
        "ambiguous_cross_source_kept": ambiguous,
        "repeated_rows_in_file": repeated_in_file,
        "stored_transactions": len(new_rows),
        "debit_transactions": sum(1 for tx in readable if tx.direction is Direction.DEBIT),
        "credit_transactions": sum(1 for tx in readable if tx.direction is Direction.CREDIT),
        "unknown_direction": reasons.get("direction_unknown", 0),
        "missing_timestamp": reasons.get("timestamp_unreadable", 0),
        "future_timestamp": reasons.get("timestamp_in_future", 0),
        "malformed_upi_id": reasons.get("upi_id_malformed", 0),
        "upi_linked_transactions": sum(1 for tx in readable if tx.counterparty_upi_id),
        # Bank-statement identity (providers/). Counts only.
        "upi_transactions": sum(1 for tx in readable if tx.payment_method == "UPI"),
        "counterparty_vpas_found": sum(1 for tx in readable if tx.counterparty_upi_id),
        "distinct_counterparty_vpas": len({tx.counterparty_upi_id for tx in readable
                                           if tx.counterparty_upi_id}),
        "counterparty_names_found": sum(1 for tx in readable if tx.counterparty_name),
        "missing_identity": sum(1 for tx in readable if tx.identity_state == "UNKNOWN"),
        "truncated_upi_ids": sum(1 for tx in readable if tx.upi_fragment),
        "identity_states": dict(Counter(tx.identity_state for tx in readable)),
        # Identifier breakdown (only COMPLETE_VPA is used for exact matching)
        "reversal_transactions": sum(1 for tx in readable if tx.transaction_type == "REVERSAL"),
        "complete_vpas_found": sum(1 for tx in readable
                                   if tx.resolved_identifier_type == "COMPLETE_VPA"),
        "partial_identifiers_found": sum(1 for tx in readable
                                         if tx.resolved_identifier_type == "PARTIAL_VPA_FRAGMENT"),
        "non_vpa_identifiers_found": sum(1 for tx in readable
                                         if tx.resolved_identifier_type == "NON_VPA_IDENTIFIER"),
        "reference_ids_found": sum(1 for tx in readable if tx.transaction_id),
        "no_identity_rows": sum(1 for tx in readable if tx.resolved_identifier_type == "UNKNOWN"),
        "identifier_types": dict(Counter(tx.resolved_identifier_type for tx in readable)),
        "wrapped_narrations_reconstructed": sum(1 for tx in readable
                                                if tx.raw_narration and "|" in tx.raw_narration),
        "provider_confidence": parsed.get("provider_confidence"),
        "provider_evidence": parsed.get("provider_evidence") or [],
        "debits_without_upi_id": debits_without_id,
        "payee_relationships_updated": len(payees),
        "graph_observations_updated": recorded_timed,
        "undated_payments_recorded": recorded_untimed,
        "profile_updated": profile is not None,
        "date_range": {
            "from": min(stamps).date().isoformat() if stamps else None,
            "to": max(stamps).date().isoformat() if stamps else None,
        },
        "unmapped_columns": parsed.get("unmapped_columns") or [],
        "warnings": warnings,
    }
    if not rows:
        status = "FAILED"
        report["warnings"].append("No transactions could be read from this file.")
    elif report["partial_transactions"] or report["rejected_transactions"]:
        status = "PARTIAL"
    else:
        status = "COMPLETE"
    report["status"] = status
    store.save_import_report(user_id, statement_id, status, report)
    return ImportResult(statement_id=statement_id, status=status, report=report,
                        profile=profile)


def record_failed_import(user_id: str, reason: str) -> str:
    """Remember that an import failed, so a later payee check can say
    "import failed" instead of "no history". Stores the category, not the
    file or its contents."""
    from backend.app.services import profile_store as store

    statement_id = f"stmt_{uuid.uuid4().hex[:8]}"
    store.save_import_report(user_id, statement_id, "FAILED",
                             {"status": "FAILED", "reason": reason})
    return statement_id
