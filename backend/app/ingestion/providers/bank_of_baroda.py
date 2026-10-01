"""Bank of Baroda account statement (PDF).

Layout (observed on a real statement, Feb 2024 period):
    DATE | NARRATION | CHQ.NO. | WITHDRAWAL (DR) | DEPOSIT (CR) | BALANCE
Rows are vertically CENTRED: the narration's first line sits above the date,
its continuation below. Narration is hard-wrapped at a fixed width and capped
at 49 characters, so a trailing VPA is often cut ("chittibabudar|am@axl" is
complete; "veerendragan|dham9@o" is not).

UPI narration grammar:
    UPI/<12-digit reference>/<HH:MM:SS>/UPI/<counterparty VPA>[/<suffix>]
BoB prints no counterparty NAME. counterparty_name therefore stays None; the
VPA local part may be offered as a display label, marked VPA_LOCAL_PART.
Other narrations seen: "<account>:Int.Pd:<from> to <to>" (interest credit)
and "Opening Balance" (no amount - not a transaction).
"""

from __future__ import annotations

import re
from datetime import time
from typing import Any

from backend.app.ingestion.providers.base import (
    Detection,
    PdfDoc,
    ProviderAdapter,
    structural_score,
)

_UPI = re.compile(r"^UPI/(\d{12})/(\d{2}):(\d{2}):(\d{2})/UPI/(.*)$")
_GRAMMAR = re.compile(r"UPI/\d{12}/\d{2}:\d{2}:\d{2}/UPI/")


class BankOfBaroda(ProviderAdapter):
    name = "bank_of_baroda"
    columns = {
        "date": ("DATE",),
        "narration": ("NARRATION",),
        "ref": ("CHQ.NO.",),
        "debit": ("WITHDRAWAL",),
        "credit": ("DEPOSIT",),
        "balance": ("BALANCE",),
    }
    date_pattern = re.compile(r"\d{2}-\d{2}-\d{4}")
    date_formats = ("%d-%m-%Y",)
    default_cap = 49
    wrap_rule = "chars"            # hard-wrapped at a fixed 43 characters

    def detect(self, doc: PdfDoc) -> Detection:
        text = doc.text
        low = text.lower()
        return structural_score(doc, [
            (0.35, "narration grammar UPI/<ref>/<time>/UPI/<vpa> (>=3 rows)",
             lambda d: len(_GRAMMAR.findall(text)) >= 3),
            (0.25, "header WITHDRAWAL (DR) / DEPOSIT (CR) / CHQ.NO.",
             lambda d: all(h in text for h in ("WITHDRAWAL (DR)", "DEPOSIT (CR)", "CHQ.NO."))),
            (0.25, "bankofbaroda domain / Bank of Baroda name",
             lambda d: "bankofbaroda" in low or "bank of baroda" in low),
            (0.15, "IFSC prefix BARB0",
             lambda d: re.search(r"\bBARB0[A-Z0-9]{6}\b", text) is not None),
        ], self.name)

    def parse_narration(self, narration: str, at_cap: bool) -> dict[str, Any]:
        m = _UPI.match(narration or "")
        if not m:
            return {"grammar": "other", "confidence": 0.6}
        ref, hh, mm, ss, rest = m.groups()
        vpa_token, separator, suffix = rest.partition("/")
        from backend.app.ingestion.providers.base import complete_vpa

        # A "/" after the VPA means the bank printed all of it.
        upi, fragment = complete_vpa(vpa_token, followed_by_suffix=bool(separator),
                                     at_cap=at_cap)
        token = vpa_token.strip()
        if upi:
            identifier_type, upi_conf = "COMPLETE_VPA", 0.95
        elif "@" in token:
            # "veerendragandham3@o", "bhimrewards.m2p@hdf": the handle was cut
            # by the 49-character narration cap. Kept whole, never matched.
            identifier_type, upi_conf = "PARTIAL_VPA_FRAGMENT", 0.4
        elif token:
            # "sriramchintada2211", "paytmqr2810...": no handle printed.
            identifier_type, upi_conf = "NON_VPA_IDENTIFIER", None
        else:
            identifier_type, upi_conf = "UNKNOWN", None
        info: dict[str, Any] = {
            "counterparty_identifier": token or None,
            "identifier_type": identifier_type,
            "upi_confidence": upi_conf,
            "grammar": "upi",
            "reference_id": ref,
            "time": time(int(hh), int(mm), int(ss)),
            "payment_method": "UPI",
            "transaction_type": "UPI",
            "counterparty_upi_id": upi,
            "upi_fragment": fragment,
            "counterparty_name": None,          # BoB does not print one
            "suffix": suffix or None,
            "confidence": 0.95 if upi else 0.7,
        }
        if upi:
            info["display_label"] = upi.split("@", 1)[0]
            info["display_label_source"] = "VPA_LOCAL_PART"
        return info
