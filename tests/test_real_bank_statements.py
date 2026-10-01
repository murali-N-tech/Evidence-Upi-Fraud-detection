"""Validation against the REAL SBI and Bank of Baroda statements.

The PDFs are private and are not committed: they live (git-ignored) in
data/uploaded_statements/. These tests skip where they are absent and assert
aggregate structure only - no statement content appears here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.app.ingestion.providers import PdfDoc, detect

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data" / "uploaded_statements"
SBI_PDF = FOLDER / "dev_sbi_statement.pdf"
BOB_PDF = next(iter(sorted(FOLDER.glob("*SECURED*.pdf"))), None) if FOLDER.exists() else None


def _parse(path):
    if path is None or not Path(path).exists():
        pytest.skip("real statement not present on this machine")
    doc = PdfDoc.load(Path(path).read_bytes())
    adapter, detections = detect(doc)
    return adapter, detections, adapter.parse(doc)


def test_real_sbi_statement():
    adapter, detections, rows = _parse(SBI_PDF)
    assert adapter.name == "sbi" and detections[0].confidence >= 0.6
    assert len(rows) == 415
    upi = [r for r in rows if r.extraction_source == "sbi:upi"]
    assert len(upi) == 409
    assert sum(r.direction.value == "DEBIT" for r in upi) == 310
    assert sum(r.transaction_type == "REVERSAL" for r in rows) == 2
    # reconstruction invariant: SBI's identifier field is exactly 10 chars and
    # its bank code 4 letters - a mis-joined wrap breaks one or the other
    assert all(len(r.counterparty_identifier) == 10 and " " not in r.counterparty_identifier
               for r in upi)
    assert all(re.fullmatch(r"[A-Z]{4}", r.counterparty_bank_code) for r in upi)
    assert all(r.counterparty_name and r.transaction_id for r in upi)
    # no complete VPA is invented from a fragment + bank code
    assert all(r.counterparty_upi_id is None for r in upi)
    assert all(r.resolved_identifier_type == "PARTIAL_VPA_FRAGMENT" for r in upi)
    assert all(r.timestamp is not None for r in rows)


def test_real_bank_of_baroda_statement():
    adapter, detections, rows = _parse(BOB_PDF)
    assert adapter.name == "bank_of_baroda" and detections[0].confidence >= 0.6
    assert len(rows) == 60
    upi = [r for r in rows if r.payment_method == "UPI"]
    assert len(upi) == 59
    assert sum(r.direction.value == "DEBIT" for r in rows) == 40
    kinds = [r.resolved_identifier_type for r in upi]
    assert kinds.count("COMPLETE_VPA") == 30
    assert kinds.count("PARTIAL_VPA_FRAGMENT") == 14
    assert kinds.count("NON_VPA_IDENTIFIER") == 15
    # reconstruction invariant: the joined narration fits BoB's 49-char cap
    assert all(len(r.narration) <= 49 and " " not in r.narration for r in upi)
    assert all(r.counterparty_identifier for r in upi)
    assert all(r.counterparty_name is None for r in rows)      # BoB prints none
    assert all(r.time_known for r in upi)
