"""The v2 feature vector at serving, built from the same code as the training matrix.

Train/serve parity is usually a promise that two implementations agree. Here it
is a property of the arrangement: the v1 twelve come from
backend.ml.features.build_vector, and every v2 channel comes from the same
functions backend/ml/dataset_v2.py used to build the training data, which are
themselves the functions the running service calls to analyse a message or a
link. There is one implementation of each feature, so there is nothing for the
two sides to disagree about.

What a caller must supply, and what it means to leave it out
------------------------------------------------------------
Everything here is optional, and absence is carried rather than filled in:

    message text absent   -> the whole message group is NaN, message_available 0
    message present, no   -> link_present 0.0 and a zero score: we read the
    link                     text and there was no link, which is a measurement
    graph view absent     -> the graph group is NaN, graph_available 0
    payee never observed  -> the graph group is NaN, as in training
    fewer payers than the -> overlap_ratio NaN, matching the production gate
    production gate

Nothing in this module reads a transaction that has not happened. The velocity
inputs come from backend/app/services/velocity.prepayment_velocity, which
filters strictly on timestamp < T.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.ml.dataset_v2 import (
    FEATURES_V2,
    GRAPH_FEATURES,
    MIN_PAYERS_FOR_OVERLAP,
    intent_features,
    link_features,
    message_features,
)
from backend.ml.features import build_vector


@dataclass
class FeatureVectorV2:
    values: dict[str, float]
    available: dict[str, bool] = field(default_factory=dict)

    def missing(self) -> list[str]:
        return sorted(name for name, ok in self.available.items() if not ok)

    def frame(self) -> pd.DataFrame:
        """Column order matters to the fitted pipeline; never rely on dict order."""
        return pd.DataFrame([[self.values[name] for name in FEATURES_V2]],
                            columns=FEATURES_V2)


def graph_features(view: Optional[Any], payer_count: Optional[float] = None) -> dict[str, float]:
    """Map a graph_cache.GraphView onto the training columns.

    An unavailable view is the ordinary case, not an error: this deployment
    sees an address only once one of its own users has paid it.
    """
    absent = {name: np.nan for name in GRAPH_FEATURES}
    absent["graph_available"] = 0.0
    if view is None or not getattr(view, "available", False):
        return absent

    payers = payer_count if payer_count is not None else getattr(view, "payer_count", None)
    overlap = getattr(view, "overlap_ratio", None)
    # The production gate, restated so that a value the live graph would refuse
    # to compute is missing here too rather than arriving as a zero.
    if payers is None or float(payers) < MIN_PAYERS_FOR_OVERLAP or overlap is None:
        overlap = np.nan

    def number(value):
        return float(value) if value is not None else np.nan

    return {
        "arrival_span_days": number(getattr(view, "arrival_span_days", None)),
        "lifespan_days": number(getattr(view, "lifespan_days", None)),
        "payments_per_day": number(getattr(view, "payments_per_day", None)),
        "component_size": number(getattr(view, "component_size", None)),
        "shared_payer_peak": number(getattr(view, "shared_payer_peak", None)),
        "overlap_ratio": float(overlap) if overlap is not None and not (
            isinstance(overlap, float) and np.isnan(overlap)) else np.nan,
        "graph_available": 1.0,
    }


def relationship_features(prior_payments: Optional[int],
                          days_since_first: Optional[float]) -> dict[str, float]:
    """The payer's own record with this payee - always readable, so never absent.

    Zero prior payments is a measurement: we looked at the payer's ledger and
    this payee is not in it. The age of the relationship is NaN in that case,
    because there is no relationship to date.
    """
    count = float(prior_payments or 0)
    return {
        "prior_payments_to_payee": count,
        "days_since_first_payment_to_payee": (
            float(days_since_first) if (count > 0 and days_since_first is not None) else np.nan),
        "relationship_available": 1.0,
    }


def build_vector_v2(
    amount: float,
    timestamp: Optional[str],
    profile: Optional[dict[str, Any]] = None,
    reputation: Optional[dict[str, Any]] = None,
    payer_seen_payee_before: Optional[bool] = None,
    seconds_since_last_txn: float = 9000.0,
    txns_last_hour: float = 0.0,
    txns_today: float = 1.0,
    message: Optional[str] = None,
    intent: Optional[str] = None,
    payee_brand: Optional[str] = None,
    payee_is_phone_address: bool = False,
    graph_view: Optional[Any] = None,
    prior_payments_to_payee: Optional[int] = None,
    days_since_first_payment_to_payee: Optional[float] = None,
) -> FeatureVectorV2:
    """One pre-payment request -> the v2 feature vector, plus availability."""
    base = build_vector(
        amount=amount, timestamp=timestamp, profile=profile, reputation=reputation or {},
        payer_seen_payee_before=payer_seen_payee_before,
        seconds_since_last_txn=seconds_since_last_txn,
        txns_last_hour=txns_last_hour, txns_today=txns_today)

    values: dict[str, float] = dict(base.values)
    available: dict[str, bool] = dict(base.available)

    msg = message_features(message)
    values.update(msg)
    for name in msg:
        available[name] = bool(message)
    available["message_available"] = True        # the flag itself is always known

    links = link_features(message, payee_brand)
    values.update(links)
    for name in links:
        available[name] = bool(message)

    stated = intent_features(intent, float(payee_is_phone_address))
    values.update(stated)
    for name in stated:
        available[name] = bool(intent)
    available["intent_stated"] = True

    graph = graph_features(graph_view,
                           payer_count=getattr(graph_view, "payer_count", None))
    values.update(graph)
    for name in graph:
        available[name] = graph["graph_available"] >= 0.5
    available["graph_available"] = True
    if not np.isnan(values["overlap_ratio"]):
        available["overlap_ratio"] = True
    else:
        available["overlap_ratio"] = False

    rel = relationship_features(prior_payments_to_payee, days_since_first_payment_to_payee)
    values.update(rel)
    for name in rel:
        available[name] = True
    available["days_since_first_payment_to_payee"] = not np.isnan(
        values["days_since_first_payment_to_payee"])

    values["payee_is_phone_address"] = float(payee_is_phone_address)
    available["payee_is_phone_address"] = True

    missing = [name for name in FEATURES_V2 if name not in values]
    if missing:
        raise KeyError(f"build_vector_v2 produced no value for {missing}")
    return FeatureVectorV2(values={name: values[name] for name in FEATURES_V2},
                           available={name: available.get(name, True) for name in FEATURES_V2})
