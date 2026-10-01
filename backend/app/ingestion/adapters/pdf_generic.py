"""PDF statements.

Wraps the existing extraction strategies in statement_parser (pdfplumber
table / text-line / app-block / GPay-block, with the pypdf fallback) and maps
their rows onto the canonical model. Those strategies already handle
multi-page files, repeated page headers (``_looks_like_header``) and wrapped
descriptions (block assembly); this adapter adds nothing provider-specific.

The provider label is detection only - it names the layout for the import
report and never changes how a row is interpreted after canonicalisation.
Text-layer PDFs only: there is no OCR path, so a scanned statement yields a
clear "no text" import error rather than a guess.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any, Optional

from backend.app.ingestion.adapters.common import single_vpa_in_text
from backend.app.ingestion.canonical import CanonicalTransaction
from backend.app.ingestion.direction import parse_direction
from backend.app.ingestion.identity import normalize_upi_id
from backend.app.ingestion.values import clean_text


def _first_pages_text(content: bytes, pages: int = 2) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join((p.extract_text() or "") for p in reader.pages[:pages])
    except Exception:
        return ""


def detect_provider(content: bytes) -> str:
    from backend.app.services.statement_parser import _detect_pdf_provider

    provider = _detect_pdf_provider(_first_pages_text(content))
    return f"{provider}_like" if provider else "generic_pdf"


_NAME_CUT = re.compile(
    r"\s*(?:UPI\s*ID|UPI\s*Ref|Transaction\s*ID|UTR\b|Txn\s*ID|Paid\s*by|Credited\s*to|"
    r"Debited\s*from|Page\s+\d+|Generated\b|₹).*$",
    re.IGNORECASE,
)
_NAME_LEAD = re.compile(r"^(?:paid\s*to|received\s*from|sent\s*to|payment\s*to)\s*", re.IGNORECASE)


def _clean_pdf_name(text: Optional[str]) -> Optional[str]:
    """The counterparty name out of a block's merged description.

    Block strategies merge every line of a transaction - and, on the last
    block of a page, the page footer - into one description. The name is the
    part before the first labelled field. Supporting metadata only: identity
    is the UPI ID.
    """
    if not text:
        return None
    name = _NAME_LEAD.sub("", _NAME_CUT.sub("", text)).strip(" -:|")
    return name[:80] or None


def _to_canonical(row: dict[str, Any], provider: str, index: int) -> CanonicalTransaction:
    ts: Optional[datetime] = None
    if row.get("timestamp"):
        try:
            ts = datetime.fromisoformat(str(row["timestamp"]))
        except ValueError:
            ts = None
    description = clean_text(row.get("merchant"), limit=300)
    if description == "UNKNOWN_MERCHANT":
        description = None
    name = _clean_pdf_name(description)
    raw_upi = clean_text(row.get("upi_id"), limit=300) or single_vpa_in_text(description)
    tx = CanonicalTransaction(
        timestamp=ts,
        time_known=bool(ts is not None and row.get("time_known")),
        amount=row.get("amount"),
        direction=parse_direction(row.get("txn_type")),
        counterparty_name=name,
        description=description,
        transaction_id=clean_text(row.get("reference_number"), limit=64),
        status_text=clean_text(row.get("status"), limit=20),
        source_provider=provider,
        source_format="pdf",
        source_row=index,
        confidence=0.9,
    )
    tx.raw_upi_text = raw_upi
    tx.counterparty_upi_id = normalize_upi_id(raw_upi) if raw_upi else None
    return tx


def parse_pdf(content: bytes, filename: str = "statement.pdf") -> dict[str, Any]:
    from backend.app.services.statement_parser import parse_statement_file

    # A bank adapter first, when the document's structure identifies one
    # with confidence (backend/app/ingestion/providers). Otherwise the
    # generic multi-strategy parser below.
    from backend.app.ingestion.providers import PdfDoc as _Doc, detect as _detect_bank

    try:
        doc = _Doc.load(content)
    except Exception:
        doc = None
    if doc is not None:
        adapter, detections = _detect_bank(doc)
        if adapter is not None:
            rows = adapter.parse(doc)
            best = detections[0]
            return {
                "transactions": rows,
                "warnings": [],
                "strategy": f"provider:{adapter.name}",
                "provider": adapter.name,
                "provider_confidence": best.confidence,
                "provider_evidence": best.evidence,
                "unmapped_columns": [],
                "mapped_columns": [],
            }

    provider = detect_provider(content)
    parsed = parse_statement_file("statement.pdf", content)
    rows = [_to_canonical(r, provider, i) for i, r in enumerate(parsed["transactions"], 1)]
    return {
        "transactions": rows,
        "warnings": list(parsed.get("warnings") or []),
        "strategy": parsed.get("source_type"),
        "provider": provider,
        "unmapped_columns": [],
        "mapped_columns": [],
    }
