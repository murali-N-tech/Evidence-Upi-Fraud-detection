"""Strict cell readers: a value is read or it is None. Never a default."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

_CLOCK = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?", re.IGNORECASE)
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]|$)")

_DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y",
    "%d %b %Y", "%d-%b-%Y", "%d %B %Y", "%d-%B-%Y", "%b %d %Y", "%B %d %Y",
    "%d %b %y", "%d-%b-%y", "%d%b%Y",
)
_TIME_FORMATS = ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p", "%I:%M%p", "%I:%M:%S%p")


def has_clock(text: object) -> bool:
    return bool(_CLOCK.search(str(text or "")))


def parse_timestamp(value: object) -> tuple[Optional[datetime], bool]:
    """(timestamp, time_known). (None, False) when the value cannot be read.

    There is deliberately no "now" fallback. An unreadable date used to become
    the moment of upload, which put statement payments in the future relative
    to every check that followed and invented velocity out of nothing.
    """
    if value is None:
        return None, False
    text = re.sub(r"\s+", " ", str(value).replace(",", " ")).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None, False
    text = re.sub(r"(?i)\s*([ap])\.?m\.?$", lambda m: f" {m.group(1).upper()}M", text)

    clock = has_clock(text)
    # ISO first - it is unambiguous, and dayfirst parsing would swap its
    # month and day.
    if _ISO.match(text):
        try:
            return datetime.fromisoformat(text.replace(" ", "T", 1).rstrip("Z")), clock
        except ValueError:
            pass
    global _LAST_FORMAT
    if _LAST_FORMAT:
        try:
            return datetime.strptime(text, _LAST_FORMAT[0]), _LAST_FORMAT[1]
        except ValueError:
            pass
    for dfmt in _DATE_FORMATS:
        for tfmt in ("",) + _TIME_FORMATS:
            fmt = f"{dfmt} {tfmt}".strip()
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            _LAST_FORMAT = (fmt, bool(tfmt))
            return parsed, bool(tfmt)
    return None, False


# The format that matched last. A statement uses one format throughout, so
# trying it first turns ~100 strptime attempts per row into one. A cache, not
# a default: a row that does not match it still goes through the full list.
_LAST_FORMAT: Optional[tuple[str, bool]] = None


def parse_amount(value: object) -> Optional[float]:
    """'1,000.50' / 'Rs. 1000' / '₹ 1,000' / '(1,000)' / '1000 DR' -> float.

    The sign is kept (a bracketed or leading-minus amount is negative); the
    caller decides whether a sign means direction.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"(?i)\b(?:inr|rs\.?|dr|cr)\b|₹", "", text)
    text = text.replace(",", "").replace("(", "").replace(")", "").strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return -abs(number) if negative else number


def clean_text(value: object, limit: int = 120) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None
    return text[:limit]
