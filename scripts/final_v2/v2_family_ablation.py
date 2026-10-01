"""V2-only leave-family-out ablation over seeds 42-51.

Reference configuration: the full 47-feature V2 representation. Each other
configuration removes the columns of one evidence family (or a union of
families) and is refitted with the identical protocol used everywhere else
in this project (scripts/experiments/common.fit_select_evaluate):

  * data: backend.ml.dataset_v2.load_or_generate_v2(seed) (fingerprint-checked cache)
  * split: 25% test, then 20% of the rest as validation, stratified, random_state = seed
  * learner: CalibratedClassifierCV(HistGradientBoostingClassifier(V1 config,
    random_state = seed), isotonic, cv = 5), fitted on the training split only
  * threshold: highest validation recall with validation FPR <= 1%, rounded to
    4 d.p.; applied once to the test split.

The full-V2 row per seed is refitted here as well (not copied), so every
configuration of one seed comes from the same run of this script.

    python scripts/final_v2/v2_family_ablation.py --seeds 42 43 44 45 46 47 48 49 50 51
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from common import fit_select_evaluate, stratified_splits  # noqa: E402

from backend.ml.dataset_v2 import FEATURES_V2, load_or_generate_v2  # noqa: E402

from families import FAMILY_OF, FAMILY_ORDER  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "final_v2" / "ablation"


def cols_without(*families):
    drop = {f for f, fam in FAMILY_OF.items() if fam in families}
    return [c for c in FEATURES_V2 if c not in drop]


CONFIGS = {
    "A. Full V2": list(FEATURES_V2),
    "B. - Relationship": cols_without("Relationship"),
    "C. - Graph": cols_without("Graph"),
    "D. - Message pressure": cols_without("Message pressure"),
    "E. - Link safety": cols_without("Link safety"),
    "F. - Stated intent": cols_without("Stated intent"),
    "G. - Contextual (message+link+intent)": cols_without("Message pressure", "Link safety", "Stated intent"),
    "H. - Relational (relationship+graph)": cols_without("Relationship", "Graph"),
    "I. - Payee history": cols_without("Payee history"),
    "J. - Payer behaviour": cols_without("Payer behaviour"),
    "K. - Amount context": cols_without("Amount context"),
    "L. - Address and QR": cols_without("Address and QR"),
    "M. - All payee-side (history+relationship+graph+address)": cols_without(
        "Payee history", "Relationship", "Graph", "Address and QR"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "v2_family_ablation_results.csv"
    done = set()
    rows = []
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = {(int(r["seed"]), r["config"]) for r in rows}
    total = len(args.seeds) * len(CONFIGS)
    k = 0
    for seed in args.seeds:
        t0 = time.time()
        df, _ = load_or_generate_v2(seed=seed)
        y = df["is_fraud"].to_numpy(); scen = df["scenario"].to_numpy()
        tr, va, te = stratified_splits(y, seed)
        for name, cols in CONFIGS.items():
            k += 1
            if (seed, name) in done:
                continue
            r = fit_select_evaluate(df.iloc[tr][cols], y[tr], df.iloc[va][cols], y[va],
                                    df.iloc[te][cols], y[te], scen[te], seed, scen_va=scen[va])
            r.pop("_model"); r.pop("_test_probs")
            r.update({"seed": seed, "config": name, "n_features": len(cols)})
            rows.append(r)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[{k}/{total}] seed {seed} {name:<58} F={len(cols):>2} PR-AUC {r['pr_auc']:.4f} "
                  f"rec {r['recall']:.4f} FPR {r['fpr']:.4f}", flush=True)
        print(f"  seed {seed} completed in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
