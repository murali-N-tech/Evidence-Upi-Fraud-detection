"""Shared, protocol-level helpers for the four research experiments.

Everything here REUSES the existing pipeline rather than re-implementing it:

* the operating-threshold rule is backend.train_model.recall_at_fpr (the same
  function the V1 trainer and, by copy, the V2 trainer use);
* the learner is the V1 configuration
  (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
  max_leaf_nodes=31, class_weight="balanced", random_state=seed)) wrapped in
  CalibratedClassifierCV(method="isotonic", cv=5), exactly as in
  backend/train_model.py and scripts/train_model_v2.py --phase freeze;
* the split is the two-stage stratified split used by train()/splits():
  25% test, then 20% of the remainder as validation.

Nothing in this module writes to models/ or evaluation*/ - the original seed-42
artefacts are never touched.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.calibration import CalibratedClassifierCV            # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier       # noqa: E402
from sklearn.metrics import (                                     # noqa: E402
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import train_test_split              # noqa: E402

from backend.train_model import recall_at_fpr                     # noqa: E402

FPR_BUDGET = 0.01
SCENARIOS = ("social_engineering", "mule_collection", "account_takeover", "card_testing")
RESULTS = ROOT / "results"


def boosted(seed: int) -> HistGradientBoostingClassifier:
    """The V1 configuration, unchanged (backend/train_model.py `boosted`)."""
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        class_weight="balanced", random_state=seed)


def calibrated(seed: int) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(boosted(seed), method="isotonic", cv=5)


def stratified_splits(y: np.ndarray, seed: int):
    """train()/splits(): test 25%, then validation = 20% of the remainder."""
    idx_fit, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y)
    idx_train, idx_val = train_test_split(
        idx_fit, test_size=0.20, random_state=seed, stratify=y[idx_fit])
    assert not (set(idx_train) & set(idx_test)) and not (set(idx_val) & set(idx_test))
    assert not (set(idx_train) & set(idx_val))
    return idx_train, idx_val, idx_test


def operating_metrics(y: np.ndarray, probs: np.ndarray, threshold: float,
                      scen=None, prefix: str = "") -> dict:
    """Threshold-dependent and threshold-free metrics on ONE split.

    `threshold` must have been chosen elsewhere (validation); this function
    never searches for one."""
    flagged = probs >= threshold
    tp = int((flagged & (y == 1)).sum()); fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum()); tn = int((~flagged & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    base = float(y.mean())
    brier = float(brier_score_loss(y, probs))
    ref = float(np.mean((y - base) ** 2))
    row = {
        "pr_auc": float(average_precision_score(y, probs)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "threshold": float(threshold),
        "recall": recall, "precision": precision,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "fpr": fp / max(fp + tn, 1),
        "brier": brier, "brier_ref": ref, "bss": (1 - brier / ref) if ref else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": int(len(y)), "n_fraud": int(y.sum()), "n_legit": int((y == 0).sum()),
    }
    if scen is not None:
        scen = np.asarray(scen)
        for name in SCENARIOS:
            mask = (scen == name) & (y == 1)
            row[f"n_{name}"] = int(mask.sum())
            row[f"recall_{name}"] = float(flagged[mask].mean()) if mask.sum() else float("nan")
    return {prefix + k: v for k, v in row.items()}


def fit_select_evaluate(X_tr, y_tr, X_va, y_va, X_te, y_te, scen_te, seed: int,
                        scen_va=None) -> dict:
    """Fit on train, choose the threshold on validation ONLY, measure test once."""
    t0 = time.time()
    model = calibrated(seed)
    model.fit(X_tr, y_tr)
    val_probs = model.predict_proba(X_va)[:, 1]
    _, threshold = recall_at_fpr(y_va, val_probs, FPR_BUDGET)
    # Stored to 4 d.p., as both trainers store it and as the service applies
    # it (models/metrics.json, models/metrics_v2.json). Rounding happens
    # before the test split is read.
    threshold = round(float(threshold), 4)
    test_probs = model.predict_proba(X_te)[:, 1]
    row = operating_metrics(y_te, test_probs, threshold, scen_te)
    row.update(operating_metrics(y_va, val_probs, threshold, scen_va, prefix="val_"))
    row["fit_seconds"] = round(time.time() - t0, 1)
    row["_model"] = model
    row["_test_probs"] = test_probs
    return row


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_frame(df) -> str:
    import pandas as pd
    h = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def environment() -> dict:
    import pandas, scipy, sklearn, joblib
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def source_state() -> dict:
    """Source-code identity. The snapshot is not a git checkout, so the code
    is identified by hashing every Python file the experiments import."""
    files = sorted(list((ROOT / "backend").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")))
    digest = hashlib.sha256()
    for f in files:
        if "__pycache__" in f.parts:
            continue
        digest.update(str(f.relative_to(ROOT)).encode())
        digest.update(f.read_bytes())
    state = {"python_source_sha256": digest.hexdigest(), "n_files": len(files)}
    base = ROOT / "SNAPSHOT_GIT_HEAD"
    if base.exists():
        state["git_head_of_snapshot"] = base.read_text().strip()
    return state


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
