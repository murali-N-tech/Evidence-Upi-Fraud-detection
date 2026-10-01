"""The payer's own recent activity, counted strictly before the payment.

Why this module exists
----------------------
The classifier is trained on three velocity features - the gap since the
payer's last payment, how many they made in the preceding hour, and how many
they have made today against their daily average. On /payee/check the service
never computed them: the model received the parameter defaults (9,000 seconds,
0, 1) on every request, whatever the payer had actually been doing. Three of
the twelve features were therefore inert on the endpoint that matters most,
while /predict computed them properly - one artefact, two input semantics.

The temporal rule, which is the whole point
-------------------------------------------
A pre-payment check may read the payer's past and nothing else. For a request
at time T:

    eligible            timestamp <  T          strictly before, never equal
    last hour           T - 1h <= timestamp < T
    today               same calendar date as T, and timestamp < T
    gap                 T - max(eligible timestamps)

A row stamped exactly at T is the payment being decided, or a duplicate of it,
and is excluded either way. A row stamped after T is the future and cannot
inform a decision made now; an earlier audit found the old code counting both,
which is how a feature that looks like behaviour becomes a feature that looks
like the answer.

`txns_today` returns the count of earlier payments today PLUS ONE. That one is
the payment being scored, not a transaction from the future: the model was
trained on a same-day count that includes the payment under consideration, and
backend/ml/features.build_vector defaults it to 1.0 for exactly that reason.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# No previous payment at all. The value the /predict path has always used, kept
# here so both endpoints describe a cold start the same way.
NO_PRIOR_GAP_SECONDS = 86_400.0


def prepayment_velocity(history: Optional[Any], now: Any,
                        timestamp_column: str = "timestamp") -> dict[str, float]:
    """Velocity features from `history`, using only what precedes `now`.

    `history` is the payer's own ledger as a DataFrame; anything unusable - no
    frame, no timestamps, unparseable values - yields the cold-start answer
    rather than a guess.
    """
    cold = {"seconds_since_last_txn": NO_PRIOR_GAP_SECONDS,
            "txns_last_hour": 0.0, "txns_today": 1.0}

    if history is None or not hasattr(history, "empty") or history.empty:
        return cold
    if timestamp_column not in history.columns:
        return cold

    moment = pd.to_datetime(now, errors="coerce")
    if pd.isna(moment):
        return cold

    stamps = pd.to_datetime(history[timestamp_column], errors="coerce").dropna()
    if stamps.empty:
        return cold

    # Both sides must agree about tz-awareness before they can be compared.
    if getattr(moment, "tzinfo", None) is not None:
        stamps = stamps.dt.tz_localize("UTC") if stamps.dt.tz is None else stamps.dt.tz_convert("UTC")
        moment = moment.tz_convert("UTC")
    elif getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_localize(None)

    past = stamps[stamps < moment]          # strictly before: never equal, never after
    if past.empty:
        return cold

    gap = (moment - past.max()).total_seconds()
    # A clock skew that produces a non-positive gap must not read as the
    # strongest velocity signal there is.
    seconds = float(min(max(gap, 1.0), NO_PRIOR_GAP_SECONDS))

    hour_ago = moment - pd.Timedelta(hours=1)
    last_hour = float(((past >= hour_ago) & (past < moment)).sum())
    today = float((past.dt.date == moment.date()).sum()) + 1.0

    return {"seconds_since_last_txn": seconds,
            "txns_last_hour": last_hour,
            "txns_today": today}
