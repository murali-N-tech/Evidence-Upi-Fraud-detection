"""Dataset v2 must be v1 plus columns - never v1 rewritten.

The whole V1-versus-V2 comparison rests on one property: the rows, the order
and the labels are the same objects in both datasets, so a difference in
measured performance is a difference in what the model was shown and not a
difference in what it was asked. These tests hold that property in place, and
one of them deliberately disturbs the v2 random stream to check that v1 does
not move with it.

They also check the property the message channel was designed around: that
"a message is present" is close to uninformative about the label. If that ever
stops being true, the channel has become a proxy for the label and every
metric downstream is measuring the generator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from backend.ml import dataset as v1
from backend.ml import dataset_v2 as v2
from backend.ml.dataset import LABEL

N = 6_000          # enough rows for the statistical checks, small enough to be quick
SEED = 42


@pytest.fixture(scope="module")
def pair():
    """The same rows, generated both ways."""
    a = v1.generate(N, seed=SEED)
    b = v2.generate_v2(N, seed=SEED)
    return a, b


# ── v1 is untouched ─────────────────────────────────────────────────────────

def test_v1_feature_columns_are_bit_identical(pair):
    a, b = pair
    pd.testing.assert_frame_equal(b[v1.FEATURES], a[v1.FEATURES], check_exact=True)


def test_v1_labels_are_bit_identical(pair):
    a, b = pair
    assert np.array_equal(b[LABEL].to_numpy(), a[LABEL].to_numpy())
    assert b[LABEL].sum() == a[LABEL].sum()
    assert b[LABEL].mean() == a[LABEL].mean(), "fraud prevalence moved"


def test_row_order_and_scenarios_are_identical(pair):
    a, b = pair
    assert list(b.index) == list(a.index)
    assert np.array_equal(b["scenario"].to_numpy(), a["scenario"].to_numpy())
    assert set(b["scenario"]) == set(a["scenario"]), "a scenario appeared or vanished"


def test_no_fraud_scenario_was_dropped_or_rebalanced(pair):
    a, b = pair
    counts_a = a["scenario"].value_counts().to_dict()
    counts_b = b["scenario"].value_counts().to_dict()
    assert counts_a == counts_b


def test_v1_values_are_not_regenerated_differently(pair):
    """A fresh v1 call must still produce what v2 carries."""
    _, b = pair
    fresh = v1.generate(N, seed=SEED)
    pd.testing.assert_frame_equal(b[v1.FEATURES], fresh[v1.FEATURES], check_exact=True)
    assert np.array_equal(b[LABEL].to_numpy(), fresh[LABEL].to_numpy())


# ── determinism, and the separation of the two streams ──────────────────────

def test_v2_generation_is_deterministic():
    first = v2.generate_v2(1_500, seed=SEED)
    second = v2.generate_v2(1_500, seed=SEED)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_changing_the_v2_stream_leaves_v1_untouched(monkeypatch):
    """The point of a separate stream: move it, and v1 must not follow."""
    base = v2.generate_v2(1_500, seed=SEED)
    monkeypatch.setattr(v2, "V2_SEED_OFFSET", 987_654)
    moved = v2.generate_v2(1_500, seed=SEED)

    pd.testing.assert_frame_equal(moved[v1.FEATURES], base[v1.FEATURES], check_exact=True)
    assert np.array_equal(moved[LABEL].to_numpy(), base[LABEL].to_numpy())
    assert np.array_equal(moved["scenario"].to_numpy(), base["scenario"].to_numpy())

    changed = [c for c in ("msg_coercion_probability", "component_size", "overlap_ratio",
                           "prior_payments_to_payee")
               if not moved[c].fillna(-1).equals(base[c].fillna(-1))]
    assert changed, "the v2 stream did not actually move - the test proves nothing"


def test_a_different_seed_changes_both_datasets():
    other = v2.generate_v2(1_500, seed=7)
    same = v2.generate_v2(1_500, seed=SEED)
    assert not np.array_equal(other[LABEL].to_numpy(), same[LABEL].to_numpy())


# ── the message channel is not a relabelled target ──────────────────────────

def test_message_presence_is_almost_uninformative(pair):
    """If 'a message exists' predicted the label, the channel would be the label."""
    _, b = pair
    auc = roc_auc_score(b[LABEL], b["message_available"])
    assert 0.40 <= auc <= 0.60, f"message_available alone reaches AUC {auc:.3f}"


def test_legitimate_rows_carry_the_vocabulary_a_naive_detector_keys_on(pair):
    """Hard negatives: real bank notices full of KYC, urgency and OTP language."""
    _, b = pair
    legit = b[(b[LABEL] == 0) & (b["message_available"] == 1)]
    assert len(legit) > 200
    for cue in ("msg_authority", "msg_urgency", "msg_fear"):
        assert legit[cue].mean() > 0.05, f"no legitimate message trips {cue}"
    assert legit["msg_coercion_probability"].max() > 0.5, (
        "no legitimate message is ever scored as pressuring - the hard negatives "
        "are not doing their job"
    )


def test_fraud_rows_are_not_all_accompanied_by_a_scam_message(pair):
    _, b = pair
    se = b[(b["scenario"] == "social_engineering")]
    assert len(se) > 20, "not enough social-engineering rows to measure this"
    quiet_share = float((se["message_available"] == 0).mean())
    assert 0.15 < quiet_share < 0.7, (
        f"{quiet_share:.0%} of social-engineering payments arrive without a message; "
        "a scam that always brings text makes 'text present' the label"
    )


def test_a_message_feature_is_missing_rather_than_zero_when_there_is_no_message(pair):
    _, b = pair
    quiet = b[b["message_available"] == 0]
    assert len(quiet) > 100
    for column in ("msg_coercion_probability", "msg_urgency", "msg_length_chars"):
        assert quiet[column].isna().all(), f"{column} is zero where it should be absent"


# ── the graph channel follows the production gates ──────────────────────────

def test_graph_features_are_absent_when_the_payee_was_never_observed(pair):
    _, b = pair
    unobserved = b[b["payee_history_available"] == 0]
    assert len(unobserved) > 100
    assert unobserved["graph_available"].eq(0).all()
    for column in v2.GRAPH_FEATURES:
        if column == "graph_available":
            continue
        assert unobserved[column].isna().all()


def test_overlap_is_refused_below_the_production_fan_in_gate(pair):
    """MIN_PAYERS_FOR_OVERLAP exists because a household paying two new shops
    produced a high overlap ratio over a tiny pool, and was blocked for it."""
    _, b = pair
    thin = b[(b["graph_available"] == 1) &
             (b["payee_distinct_payers"] < v2.MIN_PAYERS_FOR_OVERLAP)]
    assert len(thin) > 50
    assert thin["overlap_ratio"].isna().all()


def test_popular_legitimate_merchants_exist_in_quantity(pair):
    """The hard negatives the graph family must not flag."""
    _, b = pair
    popular = b[(b[LABEL] == 0) & (b["payee_distinct_payers"] >= 60) &
                (b["graph_available"] == 1)]
    assert len(popular) > 100, "no popular legitimate merchants in the data"
    assert popular["component_size"].median() > 50
    assert popular["overlap_ratio"].notna().mean() > 0.5, (
        "popular merchants should have a readable overlap - that is what makes "
        "them a hard negative rather than an absent case"
    )


# ── the feature contract ────────────────────────────────────────────────────

def test_feature_groups_partition_the_v2_feature_set():
    grouped = [f for group in v2.FEATURE_GROUPS.values() for f in group]
    assert sorted(grouped) == sorted(v2.FEATURES_V2), "a feature is in no group, or in two"
    assert len(grouped) == len(set(grouped)), "a feature appears in two groups"


def test_v2_is_a_superset_of_v1_and_keeps_its_order():
    assert v2.FEATURES_V2[: len(v1.FEATURES)] == list(v1.FEATURES)
    assert set(v1.FEATURES) <= set(v2.FEATURES_V2)


def test_every_declared_feature_is_actually_produced(pair):
    _, b = pair
    missing = [f for f in v2.FEATURES_V2 if f not in b.columns]
    assert missing == [], f"declared but not generated: {missing}"
    assert b["dataset_version"].eq("v2").all()


def test_features_for_selects_by_group():
    payer_only = v2.features_for(["payer", "amount", "velocity"])
    assert "amount" in payer_only and "payee_age_days" not in payer_only
    with pytest.raises(KeyError):
        v2.features_for(["not_a_group"])
