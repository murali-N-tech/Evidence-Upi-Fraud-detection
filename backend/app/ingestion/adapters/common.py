from __future__ import annotations

import re
from typing import Optional

# A UPI handle has no dot (okaxis, ybl, paytm, axl); an e-mail domain does.
# Used only to lift an address out of free text, and only when exactly one
# candidate is present.
_VPA_IN_TEXT = re.compile(r"(?<![\w.\-])([A-Za-z0-9][\w.\-]{0,255}@[A-Za-z][A-Za-z0-9]{1,30})(?![\w.@\-])")


def single_vpa_in_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    found = {m.group(1) for m in _VPA_IN_TEXT.finditer(text)}
    return found.pop() if len(found) == 1 else None
