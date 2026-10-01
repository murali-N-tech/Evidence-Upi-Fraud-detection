"""Experiments 1 and 2: 10-seed stability and feature-family ablation.

For every seed the existing V2 generator (backend.ml.dataset_v2.generate_v2)
is run. It calls the V1 generator first and keeps V1's rows, labels and
twelve columns bit-for-bit, so V1 and V2 are always trained and tested on the
same rows (asserted below).

Per seed and per configuration:
  * two-stage stratified split with random_state = seed (as train(seed)),
  * CalibratedClassifierCV(HGB V1-config, isotonic, cv=5) fitted on train,
  * operating threshold = highest validation recall with validation FPR <= 1%,
  * one measurement on the held-out test split.

The configuration (model family, calibration, feature definitions, threshold
rule) is the one frozen at seed 42. It is NOT re-selected per seed; the
multi-seed run measures the frozen protocol's variability, it does not tune.

    python scripts/experiments/exp1_2_multiseed_ablation.py --seeds 42 43 ... 51
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (RESULTS, SCENARIOS, environment, fit_select_evaluate,  # noqa: E402
                    sha256_frame, source_state, stratified_splits, write_json)

from backend.ml.dataset import LABEL                                  # noqa: E402
from backend.ml.dataset_v2 import (FEATURES_V1, FEATURES_V2,          # noqa: E402
                                   FEATURE_GROUPS, features_for, load_or_generate_v2)

CORE = ["payer", "amount", "velocity", "payee"]
# Experiment-2 ladder, as specified. "V1 baseline" is the 12 V1 features; the
# V2 "core" is the four V2 groups payer timing, amount, velocity and payee
# (payee includes payee_is_phone_address and NOT payee_is_new, which V2 files
# under relationship - group definitions are the repository's, unchanged).
CONFIGS = {
    "A_v1_baseline":                 None,
    "B_core":                        CORE,
    "C_core+relationship":           CORE + ["relationship"],
    "D_core+graph":                  CORE + ["graph"],
    "E_core+message":                CORE + ["message"],
    "F_core+link":                   CORE + ["link"],
    "G_core+intent":                 CORE + ["intent"],
    "H_core+rel+graph":              CORE + ["relationship", "graph"],
    "I_core+rel+graph+msg":          CORE + ["relationship", "graph", "message"],
    "J_core+rel+graph+msg+link":     CORE + ["relationship", "graph", "message", "link"],
    "K_full_v2":                     list(FEATURE_GROUPS),
}

OUT_MS = RESULTS / "multiseed"
OUT_AB = RESULTS / "ablation"


def columns_for(name: str) -> list[str]:
    groups = CONFIGS[name]
    return list(FEATURES_V1) if groups is None else features_for(groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = ap.parse_args()

    OUT_MS.mkdir(parents=True, exist_ok=True); OUT_AB.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_AB / "ablation_results.csv"
    done = set()
    if rows_path.exists():
        prev = pd.read_csv(rows_path)
        done = set(zip(prev["seed"], prev["config"]))
    config_log = {}
    cfg_path = OUT_MS / "seed_configurations.json"
    if cfg_path.exists():
        config_log = json.loads(cfg_path.read_text())

    for seed in args.seeds:
        t0 = time.time()
        df, fingerprint = load_or_generate_v2(seed=seed)
        y = df[LABEL].to_numpy()
        scen = df["scenario"].to_numpy()
        idx_tr, idx_va, idx_te = stratified_splits(y, seed)
        # V1 columns inside the V2 frame are the V1 dataset, bit for bit
        # (dataset_v2 asserts it at generation time); re-check the prefix here.
        assert list(FEATURES_V2[:len(FEATURES_V1)]) == list(FEATURES_V1)
        config_log[str(seed)] = {
            "seed": seed, "split_random_state": seed, "hgb_random_state": seed,
            "dataset_fingerprint": fingerprint,
            "dataset_sha256_v2_frame": sha256_frame(df[FEATURES_V2 + [LABEL, "scenario"]]),
            "dataset_sha256_v1_columns": sha256_frame(df[list(FEATURES_V1) + [LABEL, "scenario"]]),
            "rows": int(len(df)), "fraud_rows": int(y.sum()),
            "fraud_prevalence": float(y.mean()),
            "train": int(len(idx_tr)), "validation": int(len(idx_va)), "test": int(len(idx_te)),
            "train_fraud": int(y[idx_tr].sum()), "validation_fraud": int(y[idx_va].sum()),
            "test_fraud": int(y[idx_te].sum()),
            "test_fraud_by_scenario": {s: int(((scen[idx_te] == s) & (y[idx_te] == 1)).sum())
                                       for s in SCENARIOS},
            "model": "CalibratedClassifierCV(HistGradientBoosting v1 config, isotonic, cv=5)",
            "threshold_rule": "highest validation recall with validation FPR <= 0.01",
            "generated_seconds": round(time.time() - t0, 1),
        }
        write_json(cfg_path, config_log)
        print(f"seed {seed}: {len(df):,} rows, fraud {int(y.sum())}, "
              f"test fraud {int(y[idx_te].sum())}  ({time.time()-t0:.0f}s)", flush=True)

        for name in args.configs:
            if (seed, name) in done:
                continue
            cols = columns_for(name)
            X = df[cols]
            row = fit_select_evaluate(
                X.iloc[idx_tr], y[idx_tr], X.iloc[idx_va], y[idx_va],
                X.iloc[idx_te], y[idx_te], scen[idx_te], seed, scen_va=scen[idx_va])
            row.pop("_model"); row.pop("_test_probs")
            row.update({"seed": seed, "config": name, "n_features": len(cols),
                        "groups": "v1" if CONFIGS[name] is None else ",".join(CONFIGS[name])})
            pd.DataFrame([row]).to_csv(rows_path, mode="a", index=False,
                                       header=not rows_path.exists())
            print(f"  {name:<28} n={len(cols):>2} PR-AUC {row['pr_auc']:.4f} "
                  f"ROC {row['roc_auc']:.4f} rec {row['recall']:.4f} FPR {row['fpr']:.4f} "
                  f"SE {row['recall_social_engineering']:.3f} MC {row['recall_mule_collection']:.3f} "
                  f"({row['fit_seconds']}s)", flush=True)
    meta = {"environment": environment(), "source": source_state(),
            "finished_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
            "seeds": args.seeds, "configs": {k: v for k, v in CONFIGS.items()}}
    write_json(OUT_AB / "run_metadata.json", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
