"""Header -> canonical field mapping for tabular statements.

Exact matches on a cleaned header (case, spacing, punctuation), not substring
search: the old ``"upi" in header`` rule mapped any column containing those
letters, and ``"txn" in header`` could read a transaction TYPE column as the
transaction ID. Unmapped headers are reported, not guessed at.
"""

from __future__ import annotations

import re
from typing import Iterable

SYNONYMS: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "datetime", "date time", "transaction date time",
                  "txn date time", "date & time", "date and time"),
    "date": ("date", "txn date", "transaction date", "value date", "posting date",
             "tran date", "value dt", "txn dt"),
    "time": ("time", "txn time", "transaction time"),
    "amount": ("amount", "amount inr", "amount rs", "txn amount", "transaction amount",
               "amt", "amount (inr)", "amount (rs)", "amount(inr)", "amount(rs.)",
               "amount (rs.)", "value"),
    "debit": ("debit", "debit amount", "withdrawal", "withdrawals", "withdrawal amount",
              "withdrawal amt", "debit amt", "dr amount", "paid out", "money out"),
    "credit": ("credit", "credit amount", "deposit", "deposits", "deposit amount",
               "deposit amt", "credit amt", "cr amount", "paid in", "money in"),
    "direction": ("direction", "type", "txn type", "transaction type", "dr cr", "dr/cr",
                  "cr/dr", "debit/credit", "credit/debit", "nature", "flow"),
    "counterparty_name": ("counterparty", "counterparty name", "counter party",
                          "merchant", "merchant name", "payee", "payee name", "payer",
                          "payer name", "recipient", "recipient name", "receiver",
                          "receiver name", "beneficiary", "beneficiary name", "name",
                          "paid to", "received from", "to/from", "to / from"),
    "counterparty_upi_id": ("upi id", "upi_id", "upi", "vpa", "payee vpa", "payer vpa",
                            "upi address", "counterparty upi", "counterparty upi id",
                            "counterparty vpa", "beneficiary vpa", "to vpa", "from vpa",
                            "virtual payment address"),
    "transaction_id": ("transaction id", "txn id", "upi ref", "upi ref no",
                       "upi reference", "reference", "reference no", "reference number",
                       "ref no", "ref number", "utr", "utr no", "utr number", "rrn",
                       "chq/ref no", "chq / ref no", "cheque/ref no", "upi transaction id"),
    "description": ("description", "narration", "remarks", "particulars", "details",
                    "transaction details", "note", "notes"),
    "status": ("status", "txn status", "transaction status"),
}


def clean_header(header: object) -> str:
    text = str(header or "").replace("﻿", "").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" :.")


_LOOKUP = {syn: field for field, syns in SYNONYMS.items() for syn in syns}


def map_headers(headers: Iterable[object]) -> tuple[dict[str, str], list[str]]:
    """Return ({canonical_field: original_header}, [unmapped headers]).

    The first header that maps to a field wins; a later duplicate is listed
    as unmapped so the report shows it rather than it silently overriding.
    """
    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for header in headers:
        if header is None:
            continue
        field = _LOOKUP.get(clean_header(header))
        if field and field not in mapping:
            mapping[field] = header
        elif str(header).strip():
            unmapped.append(str(header))
    return mapping, unmapped
