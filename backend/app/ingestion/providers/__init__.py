"""Bank-statement provider adapters.

Detection is evidence-based: each adapter scores structural signals (its
narration grammar, its column headers, its bank name/domain, its IFSC
prefix). An adapter is used only when its confidence reaches
MIN_CONFIDENCE AND at least one STRUCTURAL signal (grammar or header)
fired - a bank name in free text alone never selects a parser. Otherwise the
generic parser runs. Adding a bank = adding a module here; the canonical
pipeline does not change.
"""

from __future__ import annotations

from typing import Optional

from backend.app.ingestion.providers.bank_of_baroda import BankOfBaroda
from backend.app.ingestion.providers.base import Detection, PdfDoc, ProviderAdapter
from backend.app.ingestion.providers.sbi import StateBankOfIndia

ADAPTERS: tuple[ProviderAdapter, ...] = (BankOfBaroda(), StateBankOfIndia())
MIN_CONFIDENCE = 0.6
_STRUCTURAL = ("grammar", "header")


def detect(doc: PdfDoc) -> tuple[Optional[ProviderAdapter], list[Detection]]:
    results = [(a, a.detect(doc)) for a in ADAPTERS]
    ranked = sorted(results, key=lambda r: -r[1].confidence)
    best, detection = ranked[0]
    structural = any(any(k in e for k in _STRUCTURAL) for e in detection.evidence)
    if detection.confidence >= MIN_CONFIDENCE and structural:
        return best, [d for _, d in ranked]
    return None, [d for _, d in ranked]
