"""Deterministic transaction fingerprints.

Two different questions, answered differently:

* The SAME payment imported again (the same statement uploaded twice, or two
  overlapping exports) must be stored once. Its fingerprint is identical on
  every import, so the store's unique index turns the re-import into a no-op.
* Two IDENTICAL-LOOKING rows inside one file (two Rs 10 teas at 09:56 from the
  same stall, no reference printed) are two payments. They get different
  fingerprints through their occurrence index: the 1st and 2nd copy of that
  row in the file. Re-importing the file reproduces the same indices, so the
  pair is still stored exactly twice.

When a transaction ID exists it identifies the payment on its own (with
direction and amount as a guard against an ID column that is not unique).
Without one, the fingerprint uses every identifying field that is present:
time, amount, direction, UPI ID, and the counterparty name only when there is
no UPI ID. Two payments of the same amount in the same minute to DIFFERENT
addresses therefore never collapse.

The provider/format is deliberately NOT part of the fingerprint: the same
payment exported as CSV and as PDF must not count twice. The row position is
not part of it either, because a re-export with one extra row at the top
would otherwise duplicate the whole statement.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Iterable

from backend.app.ingestion.canonical import CanonicalTransaction, Status


def _name_key(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def identity_tuple(tx: CanonicalTransaction) -> tuple:
    direction = tx.direction.value if tx.direction else ""
    amount = f"{tx.amount:.2f}" if tx.amount is not None else ""
    if tx.transaction_id:
        # A reference ID identifies the payment together with its provider,
        # date, direction and amount - never date + amount alone.
        return ("id", re.sub(r"\s+", "", tx.transaction_id).upper(), direction, amount,
                tx.source_provider or "", tx.timestamp.date().isoformat() if tx.timestamp else "")
    return (
        "row",
        tx.timestamp.isoformat(timespec="minutes") if tx.timestamp else "",
        amount,
        direction,
        tx.counterparty_upi_id or "",
        "" if tx.counterparty_upi_id else _name_key(tx.counterparty_name or tx.description),
    )


def assign_fingerprints(user_id: str, rows: Iterable[CanonicalTransaction]) -> int:
    """Fingerprint every non-rejected row. Returns how many rows in this file
    repeat an earlier identical row (kept, and reported as such)."""
    seen: Counter = Counter()
    repeats = 0
    for tx in rows:
        if tx.status is Status.REJECTED:
            continue
        ident = identity_tuple(tx)
        occurrence = seen[ident]
        seen[ident] += 1
        if occurrence and ident[0] == "row":
            repeats += 1
        # A repeated transaction ID inside one file is the same payment
        # listed twice (a PDF page overlap), so only id-less rows carry the
        # occurrence index.
        suffix = [str(occurrence)] if ident[0] == "row" else []
        payload = "|".join([user_id, *map(str, ident), *suffix])
        tx.fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return repeats


def fingerprint_dicts(user_id: str, rows: list[dict]) -> None:
    """Give store-shaped dicts that lack one the same kind of fingerprint.

    For callers that bypass the canonical pipeline (maintenance scripts, the
    legacy parse_statement_file path, tests). Same identity rule, so a row
    stored either way dedupes against the other.
    """
    from backend.app.ingestion.identity import normalize_upi_id

    seen: Counter = Counter()
    for d in rows:
        if d.get("fingerprint"):
            continue
        direction = str(d.get("direction") or d.get("txn_type") or "").upper()
        direction = direction if direction in ("DEBIT", "CREDIT") else ""
        try:
            amount = f"{float(d.get('amount')):.2f}"
        except (TypeError, ValueError):
            amount = ""
        txn_id = d.get("transaction_id") or d.get("reference_number")
        upi = d.get("counterparty_upi_id") or normalize_upi_id(d.get("upi_id"))
        if txn_id:
            ident = ("id", re.sub(r"\s+", "", str(txn_id)).upper(), direction, amount,
                     str(d.get("source_provider") or ""), str(d.get("timestamp") or "")[:10])
        else:
            ts = str(d.get("timestamp") or "")[:16]
            name = d.get("counterparty_name") or d.get("merchant")
            ident = ("row", ts, amount, direction, upi or "", "" if upi else _name_key(name))
        occurrence = seen[ident]
        seen[ident] += 1
        suffix = [str(occurrence)] if ident[0] == "row" else []
        payload = "|".join([user_id, *map(str, ident), *suffix])
        d["fingerprint"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# ── Cross-source identity ───────────────────────────────────────────────────
#
# The fingerprint above only recognises the same payment from the same KIND
# of source: a CSV without transaction IDs and a PDF that prints them yield
# different fingerprints for one payment. This second stage compares an
# incoming row with the user's already-stored rows from OTHER statements.
#
# Decision rule. A match needs ALL of:
#   1. same user, same direction (both known), same amount to the paisa;
#   2. the same payee: equal normalised UPI IDs when both rows have one;
#      otherwise equal normalised counterparty names (non-empty) - a name is
#      never matched against a different UPI ID;
#   3. reference IDs do not contradict: two IDs of the same shape that differ
#      are two payments (IDs of different shapes - a PhonePe "T..." ID and a
#      12-digit UTR - are different numbering schemes and are not compared);
#   4. time agreement, at one of two tiers:
#        STRONG  - both rows carry a clock time and differ by <= 2 minutes,
#                  or the reference IDs are equal;
#        DATE    - one row has a date only; same calendar date (IST), AND the
#                  match is unique: exactly one unclaimed stored candidate
#                  and exactly one such incoming row on that date. Anything
#                  else is AMBIGUOUS and both rows are kept.
#   5. one-to-one: a stored row absorbs at most one incoming row per import,
#      so two genuine identical payments in the new file still need two
#      stored counterparts to be treated as already known.
# A row with no timestamp is never merged across sources.

STRONG_WINDOW_SECONDS = 120


def _id_shape(value: str) -> str:
    return re.sub(r"[0-9]", "9", re.sub(r"[A-Za-z]", "A", value))


def _ids_contradict(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    a, b = re.sub(r"\s+", "", a).upper(), re.sub(r"\s+", "", b).upper()
    return a != b and _id_shape(a) == _id_shape(b)


def _ids_equal(a: str | None, b: str | None) -> bool:
    return bool(a and b) and re.sub(r"\s+", "", a).upper() == re.sub(r"\s+", "", b).upper()


def _same_payee(upi_a, name_a, upi_b, name_b) -> bool:
    if upi_a and upi_b:
        return upi_a == upi_b
    na, nb = _name_key(name_a), _name_key(name_b)
    return bool(na) and na == nb


def cross_source_duplicates(conn, user_id: str, rows: list[CanonicalTransaction],
                            statement_id: str) -> tuple[set[int], int]:
    """Indices of `rows` that are payments already stored from another
    statement, and how many rows were ambiguous (kept)."""
    from datetime import datetime as _dt

    duplicates: set[int] = set()
    ambiguous = 0
    claimed: set[int] = set()
    date_level: dict[tuple, list[int]] = {}

    def candidates(tx: CanonicalTransaction):
        return conn.execute(
            "SELECT id, timestamp, time_known, counterparty_upi_id, counterparty_name, "
            "merchant, transaction_id FROM statement_transactions "
            "WHERE user_id = ? AND direction = ? AND ROUND(amount, 2) = ROUND(?, 2) "
            "AND statement_id != ? AND fingerprint IS NOT NULL AND timestamp IS NOT NULL "
            "AND substr(timestamp, 1, 10) = ?",
            (user_id, tx.direction.value, tx.amount, statement_id,
             tx.timestamp.date().isoformat()),
        ).fetchall()

    for index, tx in enumerate(rows):
        if tx.status is Status.REJECTED or tx.direction is None or tx.timestamp is None \
                or tx.amount is None:
            continue
        found = []
        for c in candidates(tx):
            if c["id"] in claimed:
                continue
            if not _same_payee(tx.counterparty_upi_id, tx.counterparty_name or tx.description,
                               c["counterparty_upi_id"], c["counterparty_name"] or c["merchant"]):
                continue
            if _ids_contradict(tx.transaction_id, c["transaction_id"]):
                continue
            stored = _dt.fromisoformat(c["timestamp"])
            if _ids_equal(tx.transaction_id, c["transaction_id"]):
                tier = "STRONG"
            elif tx.time_known and c["time_known"]:
                if abs((stored - tx.timestamp).total_seconds()) > STRONG_WINDOW_SECONDS:
                    continue
                tier = "STRONG"
            else:
                tier = "DATE"
            found.append((tier, c["id"]))
        strong = [cid for tier, cid in found if tier == "STRONG"]
        if strong:
            claimed.add(strong[0])
            duplicates.add(index)
            continue
        dated = [cid for tier, cid in found if tier == "DATE"]
        if dated:
            key = (tx.timestamp.date(), tx.amount, tx.direction,
                   tx.counterparty_upi_id or _name_key(tx.counterparty_name or tx.description))
            date_level.setdefault(key, []).append((index, tuple(sorted(dated))))

    for key, entries in date_level.items():
        pool = {cid for _, cids in entries for cid in cids} - claimed
        if len(entries) == 1 and len(pool) == 1:
            claimed.add(pool.pop())
            duplicates.add(entries[0][0])
        else:
            ambiguous += len(entries)
    return duplicates, ambiguous
