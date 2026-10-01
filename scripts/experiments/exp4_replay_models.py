"""Experiment 4 (predictive part): do chronologically replayed graph features
carry fraud signal, and how does that compare with the simulator's graph
channel?

All four model variants are trained and tested on THE SAME replay rows with a
CHRONOLOGICAL split - the first 60% of payments by time for training, the next
15% for threshold selection, the last 25% for the single test measurement -
using the frozen learner configuration (common.fit_select_evaluate).

  A  V1 features computed from the replayed pre-payment state
  B  V2 features from the replayed state, but the 7 graph columns replaced by
     the row-level simulator's graph channel (dataset_v2._graph_row), driven
     by the replayed payer count / age / repeat ratio / observation flag and
     the row's scenario - i.e. "simulator-generated graph features" on the
     same rows
  C  V2 features entirely from the replayed state (graph via graph_view)
  D  V2 without the graph group

The seed-42 row-level V1/V2 figures are reported alongside for reference only;
they come from a different corpus and are not a like-for-like comparison.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import (RESULTS, environment, fit_select_evaluate, sha256_frame,  # noqa: E402
                    source_state, write_json)

from backend.ml.dataset import FEATURES as FEATURES_V1                       # noqa: E402
from backend.ml.dataset_v2 import FEATURES_V2, FEATURE_GROUPS, _graph_row     # noqa: E402

OUT = RESULTS / "graph_replay"
GRAPH_COLS = list(FEATURE_GROUPS["graph"])


def chronological_split(n: int, train=0.60, val=0.15):
    a, b = int(n * train), int(n * (train + val))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def simulator_graph(ev: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 20_000)
    rows = []
    for r in ev.itertuples(index=False):
        payers = float(r.state_distinct_payers or 0)
        age = float(r.state_age_days) if pd.notna(r.state_age_days) else 0.0
        rep = float(r.state_repeat_ratio) if pd.notna(r.state_repeat_ratio) else 0.0
        rows.append(_graph_row(r.scenario, payers, age, rep, bool(r.state_payee_observed), rng))
    return pd.DataFrame(rows, index=ev.index)[GRAPH_COLS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input", default=None, help="evidence CSV (default: the seed's replay output)")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()
    tag = f"seed{args.seed}"
    ev = pd.read_csv(args.input or (OUT / f"replay_evidence_{tag}.csv"), low_memory=False)
    tag = args.out_tag or tag
    ev = ev.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    assert ev["timestamp"].is_monotonic_increasing
    y = ev["is_fraud"].to_numpy(); scen = ev["scenario"].to_numpy()
    tr, va, te = chronological_split(len(ev))
    assert ev["timestamp"].iloc[tr].max() < ev["timestamp"].iloc[va].min() < ev["timestamp"].iloc[te].min()

    X1 = ev[[f"v1__{c}" for c in FEATURES_V1]].copy(); X1.columns = FEATURES_V1
    X2 = ev[[f"v2__{c}" for c in FEATURES_V2]].copy(); X2.columns = FEATURES_V2
    Xsim = X2.copy(); Xsim[GRAPH_COLS] = simulator_graph(ev, args.seed)
    Xno = X2[[c for c in FEATURES_V2 if c not in GRAPH_COLS]]

    variants = {
        "A_v1_replay_state": X1,
        "B_v2_simulator_graph_channel": Xsim,
        "C_v2_replayed_graph": X2,
        "D_v2_without_graph": Xno,
    }
    results = {}
    rows = []
    for name, X in variants.items():
        r = fit_select_evaluate(X.iloc[tr], y[tr], X.iloc[va], y[va], X.iloc[te], y[te],
                                scen[te], args.seed, scen_va=scen[va])
        r.pop("_model"); probs = r.pop("_test_probs")
        r.update({"variant": name, "n_features": X.shape[1]})
        results[name] = r; rows.append(r)
        print(f"{name:<32} PR-AUC {r['pr_auc']:.4f} ROC {r['roc_auc']:.4f} rec {r['recall']:.4f} "
              f"FPR {r['fpr']:.4f} P {r['precision']:.4f} SE {r['recall_social_engineering']:.3f} "
              f"MC {r['recall_mule_collection']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"replay_model_results_{tag}.csv", index=False)

    graph_signal = {
        "position_finding_rate_fraud": float(ev.loc[ev.is_fraud == 1, "graph_position_fired"].mean()),
        "position_finding_rate_legit": float(ev.loc[ev.is_fraud == 0, "graph_position_fired"].mean()),
        "position_finding_rate_by_scenario": ev.groupby("scenario")["graph_position_fired"].mean().to_dict(),
        "graph_available_rate_by_scenario": ev.groupby("scenario")["v2__graph_available"].mean().to_dict(),
        "graph_feature_medians_by_scenario": {
            c: ev.groupby("scenario")[f"v2__{c}"].median().to_dict() for c in GRAPH_COLS},
        "simulator_graph_feature_medians_by_scenario": {
            c: pd.Series(Xsim[c].to_numpy()).groupby(ev["scenario"].to_numpy()).median().to_dict()
            for c in GRAPH_COLS},
    }
    # non-stationarity of replayed graph features: training vs test period (legitimate payments)
    lg = (y == 0)
    graph_signal["train_vs_test_median_legit"] = {
        c: {"train": float(np.nanmedian(X2[c].to_numpy()[tr][lg[tr]])) if np.isfinite(X2[c].to_numpy()[tr][lg[tr]]).any() else None,
            "test": float(np.nanmedian(X2[c].to_numpy()[te][lg[te]])) if np.isfinite(X2[c].to_numpy()[te][lg[te]]).any() else None}
        for c in GRAPH_COLS}
    graph_signal["mule_collection_fraud_by_split"] = {
        "train": int(((scen[tr] == "mule_collection") & (y[tr] == 1)).sum()),
        "validation": int(((scen[va] == "mule_collection") & (y[va] == 1)).sum()),
        "test": int(((scen[te] == "mule_collection") & (y[te] == 1)).sum())}
    write_json(OUT / f"replay_metrics_{tag}.json", {
        "what": "Chronological replay predictive comparison. Same replay rows for A-D; "
                "chronological 60/15/25 split; threshold from validation (FPR <= 1%); one test pass.",
        "seed": args.seed,
        "split": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te)),
                  "train_fraud": int(y[tr].sum()), "validation_fraud": int(y[va].sum()),
                  "test_fraud": int(y[te].sum()),
                  "train_end": ev["timestamp"].iloc[tr].max(), "validation_end": ev["timestamp"].iloc[va].max(),
                  "test_start": ev["timestamp"].iloc[te].min()},
        "results": results,
        "graph_signal": graph_signal,
        "reference_row_level_seed42_not_comparable": {
            "V1": {"pr_auc": 0.4860, "roc_auc": 0.9652, "recall": 0.4930, "fpr": 0.00945},
            "V2_frozen": {"pr_auc": 0.7326, "roc_auc": 0.9900, "recall": 0.7803, "fpr": 0.00995},
            "note": "different corpus (row-level simulator); shown for orientation only",
        },
        "evidence_sha256": sha256_frame(ev.astype(str)),
        "environment": environment(), "source": source_state(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
