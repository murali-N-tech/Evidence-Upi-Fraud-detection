"""State Bank of India account statement (PDF).

Validated on a real SBI account statement (17 pages, 415 rows). Layout:

    Txn Date | Value Date | Description | Debit | Credit | Balance

dd/mm/yyyy dates; TOP-aligned rows (date on the description's first line,
continuations below); the description is wrapped by rendered width - at a
space, before a "/", or mid-token at the column edge; a "**...**" footer
follows the last row of each page.

UPI narration grammar (from the supplied examples):
    UPI/<DR|CR>/<12-digit reference>/<name>/<bank code>/<VPA field>/<suffix>
e.g.  UPI/DR/635380282018/Swiggy Ltd/UTIB/swiggyupi@/Payment
The VPA field is ALWAYS exactly 10 characters on the real statement (409 of
409 UPI rows) and never carries a complete handle, so it is a FRAGMENT, not
an identity: SBI UPI rows are NAME_ONLY (+ bank code + fragment). A VPA field
that were complete and valid would be used as the identity. Also seen:
UPI/REV/<ref> (reversal credits) and a few NEFT/IMPS-style narrations.
Direction comes from the Debit/Credit columns; the narration's DR/CR is only
a cross-check (a disagreement leaves direction unknown).
"""

from __future__ import annotations

import re
from typing import Any

from backend.app.ingestion.canonical import Direction
from backend.app.ingestion.providers.base import (
    Detection,
    PdfDoc,
    ProviderAdapter,
    complete_vpa,
    structural_score,
)

_UPI = re.compile(r"^UPI/(DR|CR)/(\d{12})/([^/]*)/([A-Za-z]{4})/([^/]*)(?:/(.*))?$")
_REV = re.compile(r"^UPI/REV/(\d{12})\b")
_GRAMMAR = re.compile(r"UPI/(?:DR|CR)/\d{12}/")
VPA_FIELD_WIDTH = 10          # every supplied example fills exactly 10 chars


class StateBankOfIndia(ProviderAdapter):
    name = "sbi"
    columns = {
        "date": ("TXN", "DATE"),
        "narration": ("DESCRIPTION", "NARRATION", "PARTICULARS"),
        "ref": ("REF", "REF.", "CHQ.NO.", "CHEQUE"),      # optional; absent on real SBI
        "debit": ("DEBIT", "WITHDRAWAL"),
        "credit": ("CREDIT", "DEPOSIT"),
        "balance": ("BALANCE",),
    }
    date_pattern = re.compile(r"\d{1,2}[-/ ](?:\d{2}|[A-Za-z]{3})[-/ ]\d{2,4}")
    date_formats = ("%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y", "%d %b %Y", "%d-%b-%Y",
                    "%d-%b-%y")
    default_cap = 60
    # The description text starts well left of its (centred) header; the
    # Value Date column in between is excluded because its words are dates.
    narration_from = "after_date"

    def detect(self, doc: PdfDoc) -> Detection:
        text = doc.text
        low = text.lower()
        return structural_score(doc, [
            (0.35, "narration grammar UPI/DR|CR/<ref>/... (>=3 rows)",
             lambda d: len(_GRAMMAR.findall(text)) >= 3),
            (0.25, "header Txn Date / Value Date / Description",
             lambda d: "txn date" in low and "value date" in low and "description" in low),
            (0.25, "State Bank of India name / sbi.co.in",
             lambda d: "state bank of india" in low or "onlinesbi" in low or "sbi.co.in" in low),
            (0.15, "IFSC prefix SBIN0",
             lambda d: re.search(r"\bSBIN0[A-Z0-9]{6}\b", text) is not None),
        ], self.name)

    def parse_narration(self, narration: str, at_cap: bool) -> dict[str, Any]:
        text = narration or ""
        rev = _REV.match(text)
        if rev:
            # UPI/REV/<ref>: an earlier UPI payment returned. Money comes back
            # (the Credit column says so); it is never a payment to anyone.
            return {"grammar": "upi_reversal", "reference_id": rev.group(1),
                    "payment_method": "UPI", "transaction_type": "REVERSAL",
                    "identifier_type": None, "confidence": 0.85}
        m = _UPI.match(text)
        if not m:
            return {"grammar": "other", "confidence": 0.6}
        dr_cr, ref, name, bank, field, suffix = m.groups()
        field = (field or "").strip()
        upi, fragment = complete_vpa(
            field, followed_by_suffix=False,
            # The field is fixed-width: filling it means it may be cut. A
            # trailing "@" ("swiggyupi@") is never a complete VPA.
            at_cap=len(field) >= VPA_FIELD_WIDTH)
        name = re.sub(r"\s+", " ", name).strip() or None
        if upi:
            identifier_type, upi_conf = "COMPLETE_VPA", 0.95
        elif field:
            # SBI's identifier is a VPA cut to 10 characters. Kept whole;
            # never completed with a handle guessed from the bank code.
            identifier_type, upi_conf = "PARTIAL_VPA_FRAGMENT", 0.3
        else:
            identifier_type, upi_conf = ("NAME_ONLY" if name else "UNKNOWN"), None
        return {
            "grammar": "upi",
            "reference_id": ref,
            "narrated_direction": Direction.DEBIT if dr_cr == "DR" else Direction.CREDIT,
            "counterparty_name": name,
            "name_source": "BANK_NARRATION" if name else None,
            # SBI prints a fixed-width, often truncated name field.
            "name_confidence": 0.7 if name else None,
            "counterparty_bank_code": bank.upper(),
            "counterparty_upi_id": upi,
            "counterparty_identifier": field or None,
            "identifier_type": identifier_type,
            "upi_confidence": upi_conf,
            "upi_fragment": fragment,
            "payment_method": "UPI",
            "transaction_type": "UPI",
            "suffix": suffix or None,
            "confidence": 0.9,
        }
