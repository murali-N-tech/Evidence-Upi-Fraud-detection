"""The one UPI-identifier normaliser.

Ingestion and every lookup (payee reputation, payee history, graph, QR check)
must agree on what "the same address" means, or a payment stored from a
statement can never be found again at check time. This module is the only
place that decides it; ``vpa.normalise_vpa`` delegates here.

Only unambiguous, formatting-level transformations are applied:

* Unicode NFKC folding (look-alike code points, full-width forms)
* invisible formatting characters (zero-width space/joiners, BOM, bidi marks)
* surrounding whitespace, including non-breaking and other Unicode spaces
* one pair of surrounding quotes
* a leading ``UPI ID:`` / ``VPA:`` style label
* whitespace immediately around the single ``@``
* case

Nothing else is removed. Whitespace INSIDE the local part or handle, a second
``@``, or characters outside the UPI address alphabet make the value
malformed, and a malformed value is returned as ``None`` rather than being
"repaired" into something that looks valid. No fuzzy matching.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Local part: letters, digits, dot, hyphen, underscore (NPCI VPA alphabet).
# Handle: a PSP handle - letters first, then letters/digits/dot/hyphen.
_VALID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,255}@[a-z](?:[a-z0-9.-]{0,62}[a-z0-9])$")
_LABEL = re.compile(
    r"^(?:upi\s*id|upi\s*address|upi|vpa|payee\s*vpa|virtual\s*payment\s*address)"
    r"\s*[:\-=]\s*",
    re.IGNORECASE,
)
_QUOTES = {('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("`", "`")}


def _strip_invisible(text: str) -> str:
    # Category Cf = format characters: U+200B..U+200F, U+2060, U+FEFF, bidi
    # controls. They render as nothing and arrive from PDF text layers and
    # spreadsheet exports; they are never part of a UPI address.
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _clean(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = _strip_invisible(text)
    # str.strip() already covers Unicode whitespace (NBSP folds to a space
    # under NFKC anyway).
    text = text.strip()
    for _ in range(2):  # a label inside quotes, or quotes after a label
        if len(text) >= 2 and (text[0], text[-1]) in _QUOTES:
            text = text[1:-1].strip()
        text = _LABEL.sub("", text, count=1).strip()
    # Spaces around the separator only: "9000000000 @ ybl".
    text = re.sub(r"\s*@\s*", "@", text)
    return text.lower()


def normalize_upi_id(value: object) -> Optional[str]:
    """Canonical UPI ID, or None when the value is not a well-formed one."""
    if value is None:
        return None
    text = _clean(value)
    if not text or text.count("@") != 1:
        return None
    return text if _VALID.match(text) else None


def clean_identifier(value: object) -> str:
    """Formatting-level cleanup for any payee identifier, valid or not.

    Used where a key is needed even for a value that is not a well-formed VPA
    (a pasted phone number, a name-keyed payee). Same transformations as
    ``normalize_upi_id``, minus the validity gate.
    """
    if value is None:
        return ""
    return _clean(value)


def is_phone_style(local_part: str) -> bool:
    """A local part that is essentially a phone/account number.

    Same test the address analyser uses for its "mostly digits" finding, so
    the two can never disagree about which addresses are phone-style.
    """
    local = local_part or ""
    if len(local) < 8:
        return False
    digits = sum(ch.isdigit() for ch in local)
    return digits / len(local) > 0.6
