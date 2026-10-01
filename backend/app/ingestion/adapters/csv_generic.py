"""Tabular statements (CSV / TSV / semicolon-separated).

Provider-agnostic: columns are found by header synonyms (see columns.py), so
a GPay-like, PhonePe-like or bank-like export is handled by the same code as
long as its headers are recognisable. Only formats with fixtures in
tests/fixtures/statements are claimed as tested.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any, Optional

from backend.app.ingestion.canonical import CanonicalTransaction, Direction
from backend.app.ingestion.columns import map_headers
from backend.app.ingestion.direction import (
    direction_from_amount_columns,
    parse_direction,
)
from backend.app.ingestion.identity import normalize_upi_id
from backend.app.ingestion.adapters.common import single_vpa_in_text
from backend.app.ingestion.values import clean_text, parse_amount, parse_timestamp

_MAX_PREAMBLE = 30   # bank exports put account details above the header


class CsvLayoutError(ValueError):
    """The file is tabular but no transaction header could be found."""


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding == "utf-16" and "\x00" in text:
            continue
        return text
    raise CsvLayoutError("The file is not readable text.")


def _dialect(sample: str) -> Any:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def _is_header(mapping: dict[str, str]) -> bool:
    has_amount = any(k in mapping for k in ("amount", "debit", "credit"))
    has_when = any(k in mapping for k in ("timestamp", "date"))
    return has_amount and has_when


def parse_csv(content: bytes, provider: str = "generic_csv") -> dict[str, Any]:
    text = _decode(content)
    lines = text.splitlines()
    dialect = _dialect("\n".join(lines[:50]))
    rows = list(csv.reader(io.StringIO(text), dialect))

    header_at: Optional[int] = None
    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for i, row in enumerate(rows[:_MAX_PREAMBLE]):
        m, u = map_headers(row)
        if _is_header(m):
            header_at, mapping, unmapped = i, m, u
            break
    if header_at is None:
        raise CsvLayoutError(
            "No header row with a date and an amount column was found. "
            "Recognised headers include Date, Amount, Debit/Credit, Direction, UPI ID.")

    header = rows[header_at]
    index = {field: header.index(col) for field, col in mapping.items()}

    def cell(row: list[str], field: str) -> Optional[str]:
        i = index.get(field)
        return row[i] if i is not None and i < len(row) else None

    amounts_signed = False
    if "amount" in index and "debit" not in index and "credit" not in index:
        amounts_signed = any(
            (parse_amount(cell(r, "amount")) or 0) < 0 for r in rows[header_at + 1:]
        )

    out: list[CanonicalTransaction] = []
    blank = 0
    for offset, row in enumerate(rows[header_at + 1:], start=1):
        if not any(c.strip() for c in row):
            blank += 1
            continue
        # A repeated header (concatenated exports) is not a transaction.
        if [c.strip() for c in row] == [c.strip() for c in header]:
            continue

        tx = CanonicalTransaction(source_provider=provider, source_format="csv",
                                  source_row=header_at + offset)

        if "timestamp" in index:
            tx.timestamp, tx.time_known = parse_timestamp(cell(row, "timestamp"))
        if tx.timestamp is None and "date" in index:
            date_text = cell(row, "date") or ""
            time_text = cell(row, "time") or ""
            tx.timestamp, tx.time_known = parse_timestamp(f"{date_text} {time_text}".strip())
            if tx.timestamp is None and time_text:
                tx.timestamp, _ = parse_timestamp(date_text)
                tx.time_known = False

        debit = parse_amount(cell(row, "debit"))
        credit = parse_amount(cell(row, "credit"))
        amount_text = cell(row, "amount")
        amount = parse_amount(amount_text)

        direction: Optional[Direction] = None
        if "direction" in index:
            direction = parse_direction(cell(row, "direction"))
        if direction is None and ("debit" in index or "credit" in index):
            direction = direction_from_amount_columns(
                abs(debit) if debit else None, abs(credit) if credit else None)
        if direction is None and amount_text:
            marker = re.search(r"(?i)\b(dr|cr)\.?\s*$", amount_text.strip())
            if marker:
                direction = parse_direction(marker.group(1))
        if direction is None and amounts_signed and "direction" not in index:
            direction = Direction.DEBIT if (amount or 0) < 0 else (
                Direction.CREDIT if (amount or 0) > 0 else None)

        if amount is None:
            if direction is Direction.DEBIT:
                amount = debit
            elif direction is Direction.CREDIT:
                amount = credit
            else:
                amount = debit if (debit and not credit) else credit if (credit and not debit) else None
        tx.amount = abs(amount) if amount is not None else None
        tx.direction = direction

        tx.counterparty_name = clean_text(cell(row, "counterparty_name"))
        tx.description = clean_text(cell(row, "description"), limit=200)
        tx.transaction_id = clean_text(cell(row, "transaction_id"), limit=64)
        tx.status_text = clean_text(cell(row, "status"), limit=20)

        raw_upi = clean_text(cell(row, "counterparty_upi_id"), limit=300)
        if raw_upi is None and tx.description:
            lifted = single_vpa_in_text(tx.description)
            if lifted:
                raw_upi = lifted
                tx.confidence = min(tx.confidence, 0.8)
        tx.raw_upi_text = raw_upi
        tx.counterparty_upi_id = normalize_upi_id(raw_upi) if raw_upi else None
        out.append(tx)

    return {
        "transactions": out,
        "unmapped_columns": unmapped,
        "mapped_columns": sorted(mapping),
        "warnings": [],
    }
