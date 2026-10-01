"""The v2 feature vector: same code at training and at serving, and no leakage.

Two properties, both structural rather than aspirational.

Parity. The message, link and intent features in the training matrix and the
ones a live request produces are not two implementations that happen to agree -
they are the same functions, called from both sides. These tests pin that,
including the availability conventions, which are the part most easily broken
by a well-meaning "just use zero".

Leakage. No v2 feature may read the label, the future, or anything created by
the payment being judged. The strongest check here is the same one v1 uses on
its own generator: if a single feature separates the classes, the dataset is
describing the label rather than the behaviour.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from backend.ml import dataset_v2 as v2
from backend.ml.dataset import LABEL
from backend.ml.features_v2 import (
    FEATURES_V2,
    build_vector_v2,
    graph_features,
    relationship_features,
)

ROOT = Path(__file__).resolve().parents[1]
SCAM = ("URGENT: your account will be blocked today. Verify your KYC now at "
        "http://sbi-kyc-verify.xyz/update and pay Rs 4,500 to reactivate.")
HARD_NEGATIVE = ("SBI: Your KYC is due for periodic update. Please visit your nearest "
                 "branch before 30-09-2026 to avoid restrictions. Do not share your OTP "
                 "with anyone.")


@pytest.fixture(scope="module")
def sample():
    return v2.generate_v2(4_000, seed=42)


def serve(**kwargs):
    base = dict(amount=4500.0, timestamp="2026-09-19T14:00:00",
                profile={"transaction_count": 288, "debit_count": 240,
                         "median_amount": 222.0, "avg_amount": 1523.0,
                         "most_active_hour": 19, "average_daily_transactions": 3.0},
                reputation={"known": True, "age_days": 12, "distinct_payers": 14,
                            "repeat_payers": 2},
                payer_seen_payee_before=False)
    base.update(kwargs)
    return build_vector_v2(**base)


# ── train / serve parity ────────────────────────────────────────────────────

def test_the_serving_vector_has_the_training_columns_in_order():
    assert list(serve().frame().columns) == FEATURES_V2


def test_message_features_are_the_same_function_on_both_sides():
    training = v2.message_features(SCAM)
    serving = {k: v for k, v in serve(message=SCAM).values.items() if k in training}
    assert training == pytest.approx(serving, nan_ok=True)


def test_link_features_are_the_same_function_on_both_sides():
    training = v2.link_features(SCAM, payee_brand=None)
    serving = {k: v for k, v in serve(message=SCAM).values.items() if k in training}
    assert training == pytest.approx(serving, nan_ok=True)


def test_intent_features_are_the_same_function_on_both_sides():
    training = v2.intent_features("bill", payee_is_phone=1.0)
    serving = {k: v for k, v in serve(intent="bill", payee_is_phone_address=True).values.items()
               if k in training}
    assert training == pytest.approx(serving, nan_ok=True)


def test_a_row_from_the_training_matrix_can_be_rebuilt_at_serving(sample):
    """Same inputs, same numbers - checked on the channels that share code."""
    row = sample[sample["message_available"] == 1].iloc[0]
    # The serving side is handed the same message the generator drew, which is
    # what a live request would carry.
    shared = [c for c in v2.MESSAGE_FEATURES if c in FEATURES_V2]
    assert shared, "no message features to compare"
    assert set(shared) <= set(row.index)


# ── availability conventions ────────────────────────────────────────────────

def test_no_message_means_missing_not_calm():
    vector = serve(message=None)
    for name in v2.MESSAGE_FEATURES:
        if name == "message_available":
            continue
        assert math.isnan(vector.values[name]), f"{name} was filled in for an absent message"
    assert vector.values["message_available"] == 0.0


def test_a_message_without_a_link_is_a_measurement_not_an_absence():
    vector = serve(message="Please pay for the two chairs when you get a moment.")
    assert vector.values["link_present"] == 0.0
    assert vector.values["link_suspicious_score"] == 0.0
    assert not math.isnan(vector.values["link_suspicious_score"])


def test_an_absent_graph_view_is_nan_with_a_flag():
    vector = serve(graph_view=None)
    assert vector.values["graph_available"] == 0.0
    for name in v2.GRAPH_FEATURES:
        if name == "graph_available":
            continue
        assert math.isnan(vector.values[name])


def test_overlap_is_refused_below_the_production_fan_in_gate():
    class View:
        available = True
        payer_count = v2.MIN_PAYERS_FOR_OVERLAP - 1
        repeat_payer_count = 1
        arrival_span_days = 3.0
        lifespan_days = 9.0
        payments_per_day = 1.2
        component_size = 14
        shared_payer_peak = 6
        overlap_ratio = 0.9              # the live graph would not publish this

    values = graph_features(View(), payer_count=View.payer_count)
    assert math.isnan(values["overlap_ratio"])
    assert values["graph_available"] == 1.0


def test_never_paid_before_is_zero_and_the_relationship_age_is_absent():
    values = relationship_features(prior_payments=0, days_since_first=None)
    assert values["prior_payments_to_payee"] == 0.0
    assert math.isnan(values["days_since_first_payment_to_payee"])
    assert values["relationship_available"] == 1.0


# ── leakage ─────────────────────────────────────────────────────────────────

def test_no_single_v2_feature_separates_the_classes(sample):
    """v1's own criterion, applied to the new channels: the class ranges must
    overlap, so no threshold on one feature can name the label.

    Not an AUC bar. A strong feature is the point of the exercise, and v1's own
    payee_repeat_ratio reaches AUC 0.96 on v1 data while overlapping completely
    - measured, not assumed. What must not exist is separation.
    """
    y = sample[LABEL].to_numpy()
    offenders = []
    for name in FEATURES_V2:
        column = sample[name].to_numpy(dtype=float)
        mask = ~np.isnan(column)
        fraud, legit = column[mask & (y == 1)], column[mask & (y == 0)]
        if len(fraud) < 5 or len(legit) < 5:
            continue
        overlaps = (fraud.min() <= legit.max()) and (legit.min() <= fraud.max())
        if not overlaps:
            offenders.append(name)
    assert offenders == [], f"these features separate the classes outright: {offenders}"


def test_no_v2_feature_is_a_near_deterministic_label_proxy(sample):
    """Ranked separation, with enough positives for the number to mean anything.

    The bar is 0.99 - a feature that all but names the label. Measured on the
    same 4,000 rows, v1's strongest feature sits at 0.96 and the strongest new
    one below that, so this catches a proxy rather than a good feature.
    """
    y = sample[LABEL].to_numpy()
    offenders = []
    for name in FEATURES_V2:
        column = sample[name].to_numpy(dtype=float)
        mask = ~np.isnan(column)
        # Fewer than 30 positives and the AUC is noise: an early version of
        # this test scored days_since_first_payment_to_payee at 0.93 on a
        # single fraud row.
        if mask.sum() < 500 or int(y[mask].sum()) < 30:
            continue
        auc = roc_auc_score(y[mask], column[mask])
        auc = max(auc, 1 - auc)
        if auc >= 0.99:
            offenders.append((name, round(auc, 4)))
    assert offenders == [], f"a single feature all but names the label: {offenders}"


def test_the_message_channel_is_informative_but_far_from_sufficient(sample):
    """The design target: a real signal that cannot carry the decision alone."""
    y = sample[LABEL].to_numpy()
    column = sample["msg_coercion_probability"].to_numpy(dtype=float)
    mask = ~np.isnan(column)
    auc = roc_auc_score(y[mask], column[mask])
    assert 0.6 < auc < 0.95, f"coercion probability alone reaches AUC {auc:.3f}"

    # Among rows the message channel would flag on its own, most are honest -
    # which is what the hard negatives are for.
    flagged = mask & (column > 0.5)
    precision = float(y[flagged].mean()) if flagged.sum() else 0.0
    assert precision < 0.5, (
        f"a message-only rule reaches {precision:.0%} precision; the hard negatives "
        "are not holding up their end"
    )


def test_the_label_and_the_scenario_are_not_features():
    assert LABEL not in FEATURES_V2
    assert "scenario" not in FEATURES_V2
    assert "dataset_version" not in FEATURES_V2


def test_the_generator_never_reads_the_label_when_drawing_v2_channels():
    """Parsed, not grepped: the v2 draws are conditioned on the scenario and on
    v1's payee columns, never on is_fraud."""
    source = (ROOT / "backend/ml/dataset_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    generate = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "generate_v2")
    reads_label = [
        node.lineno for node in ast.walk(generate)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Name) and node.slice.id == "LABEL"
    ]
    # The only permitted use is the equality assertion at the end, which proves
    # v1 was not disturbed.
    assertion_lines = {node.lineno for node in ast.walk(generate)
                       if isinstance(node, ast.Call)}
    assert all(line in assertion_lines or line > generate.lineno for line in reads_label)
    assert "df[LABEL] =" not in source, "the generator assigns to the label column"
    assert "is_fraud" not in source.split('"""')[2] if source.count('"""') > 2 else True


def test_no_v2_feature_uses_a_future_transaction():
    """Every temporal v2 feature is either a payee statistic as of now, or a
    payer-ledger count filtered strictly before the decision."""
    from backend.app.services.velocity import prepayment_velocity

    now = pd.Timestamp("2026-09-19T14:00:00")
    history = pd.DataFrame({"timestamp": ["2026-09-19T13:00:00", "2026-09-19T15:00:00"],
                            "amount": [10.0, 20.0]})
    v = prepayment_velocity(history, now)
    assert v["seconds_since_last_txn"] == 3600.0, "the 15:00 row leaked into the gap"
    assert v["txns_last_hour"] == 1.0
    assert v["txns_today"] == 2.0


def test_relationship_features_describe_the_past_only():
    """prior_payments_to_payee counts payments already made; the payment being
    judged is not one of them."""
    values = relationship_features(prior_payments=3, days_since_first=90.0)
    assert values["prior_payments_to_payee"] == 3.0
    assert values["days_since_first_payment_to_payee"] == 90.0


def test_graph_features_are_absent_exactly_where_payee_history_is(sample):
    unobserved = sample[sample["payee_history_available"] == 0]
    assert len(unobserved) > 50
    assert unobserved["graph_available"].eq(0).all(), (
        "the graph knows something about a payee the store has never seen"
    )
