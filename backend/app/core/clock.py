"""One time convention for the whole payment pipeline: Indian Standard Time.

Indian UPI statements (GPay, PhonePe, Paytm, bank exports) print wall-clock
times in IST without a zone. The server, however, may run anywhere - a
developer laptop in IST, Render in UTC. Mixing "the server's local clock"
with "the statement's wall clock" put every check made on Render 5h30m
behind the statement history it was compared with: a 23:30 payment and a
00:15 check the next day looked like the same day, and "strictly before T"
admitted or excluded the wrong rows.

The rule, everywhere:

* A naive timestamp from a statement IS IST wall time. It is never shifted.
* An aware timestamp (with Z or +hh:mm) is converted to IST.
* "Now" is IST now, derived from UTC, independent of the server's zone.
* Everything is stored and compared as naive IST ISO text.

IST has no daylight saving, so a fixed +05:30 offset is exact and needs no
tz database (Windows Python ships without one). APP_TIMEZONE_OFFSET_MINUTES
can override it for a deployment serving another single zone.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

APP_TZ = timezone(timedelta(minutes=int(os.environ.get("APP_TIMEZONE_OFFSET_MINUTES", "330"))),
                  name="IST")


def now_app() -> datetime:
    """Current IST wall time, naive. Independent of the server's timezone."""
    return datetime.now(timezone.utc).astimezone(APP_TZ).replace(tzinfo=None)


def to_app_naive(value: Any) -> Optional[datetime]:
    """Any timestamp -> naive IST datetime. Naive input is taken as IST."""
    if value is None or value == "":
        return None
    try:
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            value = value.to_pydatetime()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip()
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                import pandas as pd

                parsed = pd.to_datetime(text, errors="coerce")
            except Exception:
                return None
            if parsed is None or pd.isna(parsed):
                return None
            moment = parsed.to_pydatetime()
    if moment.tzinfo is not None:
        moment = moment.astimezone(APP_TZ).replace(tzinfo=None)
    return moment


def app_iso(value: Any, timespec: str = "microseconds") -> Optional[str]:
    moment = to_app_naive(value)
    return moment.isoformat(timespec=timespec) if moment else None
