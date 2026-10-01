"""V2's evaluation: same rows as V1, chosen on validation, reproducible from its own CSVs.

The comparison in the report is only meaningful if three things hold, so each
is a test rather than a claim:

    V2 was measured on exactly the rows V1 was measured on, with the same labels
    nothing about V2 was chosen using those rows
    V1's artefacts are exactly as they were before V2 existed
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "evaluation"
V2 = ROOT / "evaluation_v2"
FROZEN = ROOT / "models" / "metrics_v2.json"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def v2_metrics():
    if not (V2 / "metrics.json").exists():
        pytest.skip("run scripts/evaluate_model_v2.py --phase main first")
    return json.loads((V2 / "metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v1_metrics():
    return json.loads((V1 / "metrics.json").read_text(encoding="utf-8"))


# ── V1 is where it was ──────────────────────────────────────────────────────

def test_v1_artefacts_are_untouched(v1_metrics):
    """V1's own manifest recorded the model's hash. It must still match."""
    manifest = json.loads((V1 / "manifest.json").read_text(encoding="utf-8"))
    from scripts.evaluate_model import sha256

    for path, recorded in manifest["artefact_sha256"].items():
        assert sha256(ROOT / path) == recorded, f"{path} changed since the V1 evaluation"


def test_v1_and_v2_results_live_in_separate_directories():
    assert (V1 / "metrics.json").exists() and (V2 / "metrics.json").exists()
    assert (ROOT / "models" / "risk_model.pkl").exists()
    assert (ROOT / "models" / "risk_model_v2.pkl").exists()


def test_v1_metrics_still_say_what_they_said(v1_metrics):
    assert v1_metrics["test_metrics"]["average_precision"] == pytest.approx(0.48599, abs=1e-5)
    assert v1_metrics["test_metrics"]["recall"] == pytest.approx(0.49296, abs=1e-5)


# ── the same rows, the same labels ──────────────────────────────────────────

def test_v2_was_measured_on_exactly_the_rows_v1_was(v1_metrics, v2_metrics):
    a = read_csv(V1 / "csv" / "test_predictions.csv")
    b = read_csv(V2 / "csv" / "test_predictions.csv")
    assert len(a) == len(b) == v2_metrics["test_metrics"]["support"]["total"]
    assert [r["dataset_row_index"] for r in a] == [r["dataset_row_index"] for r in b]
    assert [r["is_fraud"] for r in a] == [r["is_fraud"] for r in b], "the labels moved"
    assert [r["scenario"] for r in a] == [r["scenario"] for r in b]


def test_support_and_prevalence_match_v1(v1_metrics, v2_metrics):
    assert v2_metrics["test_metrics"]["support"] == v1_metrics["test_metrics"]["support"]
    assert v2_metrics["test_metrics"]["fraud_prevalence"] == pytest.approx(
        v1_metrics["test_metrics"]["fraud_prevalence"])


# ── nothing about V2 was chosen on the test split ───────────────────────────

def test_the_frozen_configuration_declares_no_test_involvement():
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert frozen["test_rows_used_in_any_decision"] is False
    assert frozen["validation_results"]["threshold_selected_on"] == "validation split"
    assert frozen["version"] == "v2"


def test_the_operating_threshold_came_from_validation(v2_metrics):
    point = v2_metrics["operating_point"]
    assert point["test_labels_used_in_selection"] is False
    assert "validation" in point["source"]
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert point["threshold"] == frozen["validation_results"]["threshold"]


def test_the_development_script_never_reads_a_test_label():
    """Parsed: y[idx_test] may be sliced nowhere in train_model_v2."""
    source = (ROOT / "scripts/train_model_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Name):
            target = getattr(node.value, "id", "")
            if node.slice.id == "idx_test" and target in ("y", "probs", "scen"):
                offenders.append(node.lineno)
    assert offenders == [], f"a test label is read at line(s) {offenders}"


def test_the_frozen_threshold_respects_the_budget_on_test(v2_metrics):
    point = v2_metrics["recall_at_fpr_budget"]
    assert point["realised_false_positive_rate"] <= point["budget"] + 1e-9
    assert point["within_budget"] is True


# ── the numbers regenerate from the published CSVs ──────────────────────────

def test_the_predictions_csv_reproduces_the_headline_metrics(v2_metrics):
    rows = read_csv(V2 / "csv" / "test_predictions.csv")
    y = np.array([int(r["is_fraud"]) for r in rows])
    p = np.array([float(r["predicted_probability"]) for r in rows])
    assert roc_auc_score(y, p) == pytest.approx(v2_metrics["test_metrics"]["roc_auc"], abs=1e-6)
    assert average_precision_score(y, p) == pytest.approx(
        v2_metrics["test_metrics"]["average_precision"], abs=1e-6)

    threshold = v2_metrics["operating_point"]["threshold"]
    flagged = p >= threshold
    cm = v2_metrics["test_metrics"]["confusion_matrix"]
    assert int((flagged & (y == 1)).sum()) == cm["tp"]
    assert int((flagged & (y == 0)).sum()) == cm["fp"]


def test_every_scenario_is_reported_with_its_support(v2_metrics):
    rows = {r["scenario"]: r for r in v2_metrics["scenario_recall"]}
    for name in ("social_engineering", "mule_collection", "account_takeover",
                 "card_testing", "legitimate (false alarms)"):
        assert name in rows
    counted = sum(r["n"] for k, r in rows.items() if k != "legitimate (false alarms)")
    assert counted == v2_metrics["test_metrics"]["support"]["fraud"]


def test_the_ablation_and_baseline_tables_exist_and_cover_the_ladder():
    ablation = read_csv(V2 / "ablation_results.csv")
    labels = {r["feature_set"] for r in ablation}
    for rung in ("A. payer only", "B. payer + payee", "C. payer + payee + velocity",
                 "D. C + graph", "E. C + message/context", "F. full v2",
                 "full v2 without graph", "full v2 without message/context",
                 "v1 twelve (reference)"):
        assert rung in labels, f"{rung} missing from the ablation"
    baselines = {r["model"] for r in read_csv(V2 / "baseline_comparison.csv")}
    assert {"RandomForest", "ExtraTrees", "LogisticRegression"} <= baselines


def test_the_comparison_table_reports_losses_as_losses():
    rows = read_csv(V2 / "comparison_v1_vs_v2.csv")
    assert rows, "no comparison table"
    by_metric = {r["metric"]: r for r in rows}
    assert "False-positive rate" in by_metric
    for row in rows:
        change = float(row["change"])
        better = row["higher_is_better"] == "True"
        if change == 0:
            assert row["direction"] == "-"
        else:
            expected = "better" if (change > 0) == better else "WORSE"
            assert row["direction"] == expected, f"{row['metric']} is mislabelled"


# ── the manifest is enough to repeat the run ────────────────────────────────

def test_the_manifest_records_what_reproduction_needs():
    manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    from backend.ml.dataset_v2 import FEATURES_V2

    assert manifest["seed"] == 42
    assert manifest["features"] == FEATURES_V2
    assert manifest["threshold_selection_method"].startswith("highest recall")
    assert manifest["splits"] == {"train": 72000, "validation": 18000, "test": 30000}
    for key in ("models/risk_model_v2.pkl", "backend/ml/dataset_v2.py",
                "data/coercion_corpus.jsonl"):
        assert len(manifest["artefact_sha256"][key]) == 64
    assert any("train_model_v2.py --phase freeze" in step for step in manifest["reproduce"])
    assert manifest["dataset"]["dataset_version"] == "v2"
