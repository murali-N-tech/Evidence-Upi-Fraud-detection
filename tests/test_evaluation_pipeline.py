"""The evaluation pipeline: reproducible, and measured on data the model never saw.

An evaluation script can be wrong in ways its own output cannot show. The two
that matter here:

    the rows it calls "held out" are not the rows the model was held out from

    the operating point it reports was chosen using the very labels it then
    reports against, which turns a false-positive budget into a promise the
    system cannot keep in production

Both are structural, so they are tested structurally rather than eyeballed.
Nothing in this file retrains anything; the heavy work (scoring 30,000 rows,
SHAP, permutation importance) belongs to the script and is exercised by its own
run, not repeated here.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from backend.ml.dataset import FEATURES, LABEL, generate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_model.py"
OUT = ROOT / "evaluation"
SEED = 42


@pytest.fixture(scope="module")
def splits():
    """The same three splits backend/train_model.train() derives."""
    import scripts.evaluate_model as ev
    return ev.rebuild_splits(SEED)


@pytest.fixture(scope="module")
def results():
    if not (OUT / "metrics.json").exists():
        pytest.skip("run scripts/evaluate_model.py first")
    return json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))


# ── the split is the training run's split, and it is deterministic ──────────

def test_the_split_matches_the_training_scripts_own_call(splits):
    df, y, idx_train, idx_val, idx_test = splits
    expected_fit, expected_test = train_test_split(
        np.arange(len(df)), test_size=0.25, random_state=SEED, stratify=y)
    expected_train, expected_val = train_test_split(
        expected_fit, test_size=0.20, random_state=SEED, stratify=y[expected_fit])
    assert np.array_equal(idx_test, expected_test)
    assert np.array_equal(idx_train, expected_train)
    assert np.array_equal(idx_val, expected_val)


def test_rebuilding_twice_gives_the_same_rows(splits):
    """Same seed, same rows - the property the whole evaluation rests on."""
    import pandas as pd
    import scripts.evaluate_model as ev

    df, _, idx_train, idx_val, idx_test = splits
    again_df, _, again_train, again_val, again_test = ev.rebuild_splits(SEED)
    assert np.array_equal(idx_test, again_test), "the test split is not deterministic"
    assert np.array_equal(idx_train, again_train)
    assert np.array_equal(idx_val, again_val)
    # the data itself too, NaN-aware: the payee features are NaN wherever the
    # deployment would not have observed the payee, so == would say False
    pd.testing.assert_frame_equal(df, again_df)


def test_the_test_split_shares_no_row_with_train_or_validation(splits):
    df, _, idx_train, idx_val, idx_test = splits
    assert not (set(idx_test) & set(idx_train))
    assert not (set(idx_test) & set(idx_val))
    assert len(set(idx_train) | set(idx_val) | set(idx_test)) == len(df)


def test_the_split_is_stratified_so_prevalence_is_comparable(splits):
    _, y, idx_train, idx_val, idx_test = splits
    whole = y.mean()
    for idx in (idx_train, idx_val, idx_test):
        assert abs(y[idx].mean() - whole) < 0.002


def test_a_different_seed_would_be_a_different_split(splits):
    import scripts.evaluate_model as ev
    other = ev.rebuild_splits(7)
    assert not np.array_equal(splits[4], other[4]), (
        "if the seed does not change the split, the split is not seed-derived"
    )


# ── the threshold comes from validation, never from test ────────────────────

def test_the_reported_threshold_is_the_one_selected_on_validation(results):
    op = results["operating_point"]
    assert op["test_labels_used_in_selection"] is False
    assert op["test_features_used_in_selection"] is False
    assert "validation" in op["source"]
    assert op["rederivation_agrees_to_stored_precision"] is True
    assert op["difference"] <= 5e-5, "the stored threshold is not reproducible from validation"


def test_the_script_never_selects_a_threshold_from_test_data():
    """Read the script: recall_at_fpr - the threshold chooser - may be called
    on the validation split and nowhere else."""
    source = SCRIPT.read_text(encoding="utf-8")
    calls = re.findall(r"recall_at_fpr\(([^)]*)\)", source)
    assert calls, "the threshold re-derivation has gone missing"
    for call in calls:
        assert "idx_val" in call, f"recall_at_fpr called on something other than validation: {call}"
        assert "y_test" not in call and "idx_test" not in call


def test_the_script_fits_nothing():
    """Parsed, not grepped: prose about train_model.train() is not a call to it."""
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    banned = {"fit", "fit_transform", "partial_fit", "train", "dump"}
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in banned:
            offenders.append(f"line {node.lineno}: {name}()")
    assert offenders == [], f"an evaluation script must not fit or write a model: {offenders}"


# ── every reported number comes from the held-out rows ──────────────────────

def test_support_counts_and_prevalence_describe_the_test_split(results, splits):
    _, y, _, _, idx_test = splits
    support = results["test_metrics"]["support"]
    assert support["total"] == len(idx_test)
    assert support["fraud"] == int(y[idx_test].sum())
    assert support["legitimate"] == support["total"] - support["fraud"]
    assert results["test_metrics"]["fraud_prevalence"] == pytest.approx(y[idx_test].mean())
    assert results["data"]["splits"]["test_rows"] == len(idx_test)


def test_the_confusion_matrix_adds_up_to_the_test_split(results):
    cm = results["test_metrics"]["confusion_matrix"]
    support = results["test_metrics"]["support"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == support["total"]
    assert cm["tp"] + cm["fn"] == support["fraud"]
    assert cm["tn"] + cm["fp"] == support["legitimate"]


def test_the_operating_point_respects_its_false_positive_budget(results):
    point = results["recall_at_fpr_budget"]
    assert point["realised_false_positive_rate"] <= point["budget"]
    assert point["within_budget"] is True


def test_every_fraud_scenario_is_reported_with_its_support(results):
    rows = {r["scenario"]: r for r in results["scenario_recall"]}
    for name in ("social_engineering", "mule_collection", "account_takeover", "card_testing"):
        assert name in rows, f"{name} is missing from the scenario breakdown"
        assert rows[name]["n"] > 0
        assert 0.0 <= rows[name]["recall"] <= 1.0
        assert rows[name]["detected"] <= rows[name]["n"]
    counted = sum(r["n"] for k, r in rows.items() if k != "legitimate (false alarms)")
    assert counted == results["test_metrics"]["support"]["fraud"]


# ── the CSVs reproduce the headline numbers ─────────────────────────────────

def read_csv(name):
    with open(OUT / "csv" / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_predictions_csv_regenerates_the_confusion_matrix(results):
    if not (OUT / "csv" / "test_predictions.csv").exists():
        pytest.skip("run the script first")
    rows = read_csv("test_predictions.csv")
    threshold = results["operating_point"]["threshold"]
    tp = fp = fn = tn = 0
    for r in rows:
        actual = int(r["is_fraud"])
        flagged = float(r["predicted_probability"]) >= threshold
        tp += actual and flagged
        fn += actual and not flagged
        fp += (not actual) and flagged
        tn += (not actual) and not flagged
    cm = results["test_metrics"]["confusion_matrix"]
    assert (tp, fp, fn, tn) == (cm["tp"], cm["fp"], cm["fn"], cm["tp"] and cm["tn"] or cm["tn"])
    assert len(rows) == results["test_metrics"]["support"]["total"]


def test_the_predictions_csv_regenerates_roc_auc_and_average_precision(results):
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = read_csv("test_predictions.csv")
    y = np.array([int(r["is_fraud"]) for r in rows])
    p = np.array([float(r["predicted_probability"]) for r in rows])
    assert roc_auc_score(y, p) == pytest.approx(results["test_metrics"]["roc_auc"], abs=1e-6)
    assert average_precision_score(y, p) == pytest.approx(
        results["test_metrics"]["average_precision"], abs=1e-6)


def test_the_threshold_sweep_contains_the_operating_point(results):
    rows = read_csv("threshold_sweep.csv")
    threshold = results["operating_point"]["threshold"]
    match = [r for r in rows if abs(float(r["threshold"]) - threshold) < 1e-9]
    assert match, "the sweep does not include the threshold actually used"
    cm = results["test_metrics"]["confusion_matrix"]
    assert int(match[0]["tp"]) == cm["tp"] and int(match[0]["fp"]) == cm["fp"]


def test_the_scenario_csv_matches_the_json(results):
    rows = {r["scenario"]: r for r in read_csv("scenario_recall.csv")}
    for r in results["scenario_recall"]:
        assert int(rows[r["scenario"]]["n"]) == r["n"]
        assert float(rows[r["scenario"]]["recall"]) == pytest.approx(r["recall"])


def test_the_run_is_documented_well_enough_to_repeat():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == SEED
    assert manifest["reproduce"].startswith("python scripts/evaluate_model.py")
    for key in ("models/risk_model.pkl", "backend/ml/dataset.py", "backend/train_model.py"):
        assert len(manifest["artefact_sha256"][key]) == 64
    for lib in ("numpy", "scikit-learn", "pandas"):
        assert manifest["libraries"][lib]


def test_the_feature_set_reported_is_the_feature_set_trained(results):
    names = [r["feature"] for r in results["feature_importance"]["permutation"]]
    assert names == list(FEATURES)
