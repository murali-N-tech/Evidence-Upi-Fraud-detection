# REPRODUCIBILITY MANIFEST — final V2 results

Evaluated code: snapshot of git HEAD `9022a9feb65effa32779207e32458c0d37e9bb61` of the project repository (working tree with uncommitted changes; file hashes below identify the exact code and artefacts).

## Environment

- Experiments: Python 3.11.15 (Linux-6.18.44-fc-v37-x86_64-with-glibc2.39); numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, joblib 1.6.0, matplotlib 3.11.2, shap 0.49.1
- Frozen V2 model built with: scikit-learn 1.7.2 (the pickle loads under scikit-learn 1.7.2 with no InconsistentVersionWarning); the remaining build-environment versions are not recorded in models/metrics_v2.json
- Software tests: project machine: Python 3.10.12, pytest 9.1.1, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2

## Seeds

- frozen_v2_corpus_and_split: 42
- ten_seed_and_ablation: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
- replay_streams: [42, 43, 44]
- hgb_random_state: = seed
- split_random_state: = seed
- permutation_importance: 42
- shap_sample: 42

## Dataset configuration

- n_transactions: 120000
- nominal_fraud_rate: 0.012
- n_payers: 2000
- dataset_version: v2
- seed-42 dataset fingerprint (generator, corpus and coercion-weight hashes): see reproducibility_manifest.json

## Model and threshold configuration

- model: HistGradientBoostingClassifier (max_iter 300, learning_rate 0.08, max_leaf_nodes 31, class_weight balanced)
- calibration: isotonic, 5-fold
- features: 47
- hgb: max_iter 300, learning_rate 0.08, max_leaf_nodes 31, class_weight balanced
- threshold rule: highest validation recall with validation FPR <= 1%, rounded to 4 d.p.
- threshold frozen_threshold: 0.1123
- threshold full_precision: 0.1122651497549468

## Feature configuration

- 47 V2 features in 9 implementation groups: payer (2); amount (2); velocity (3); payee (5); relationship (4); graph (7); message (18); link (3); intent (3)

## Runtime substitutions (replay only; none changes a prediction)

- backend.app.core.clock.now_app -> replay time t
- re._MAXCACHE = 8192 (performance)
- OMP_NUM_THREADS=1 (performance)

## Key artefact hashes (SHA-256)

| File | SHA-256 |
|---|---|
| models/risk_model_v2.pkl | `ebab764adf427970bc6de5d397ab340e0156341b0afee809c84dc43e984c8385` |
| models/metrics_v2.json | `88c5f945a7725fc397849ba06c41d1bff660b307100448b7917be6d795e555b4` |
| models/coercion_weights.json | `558a966a777607393e250adcb1617e397955c14d0594238705f43ba7ae90940e` |
| backend/ml/dataset_v2.py | `e1a26e15589b9f2450db913d5a8091ae68d8e10c4f81865d71df520a0cc8461d` |
| backend/ml/dataset.py | `c890ba4a3cc28c8bd151ce812d9940e3a5d4eeef9b4d27f05c0c734c79cfe8ac` |
| backend/ml/features_v2.py | `3dbecdee62d230c53706d1569f4aa4dee828268df31900fac0b51b1411a75807` |
| backend/ml/replay_corpus.py | `5726ef5a1bbc6c98e8999dec003eda4d287be689c4aa4368bb191d7612f2bbc0` |
| data/coercion_corpus.jsonl | `e5cf37f089c1bc820d48138c59908d76ab9614d3c2c7655990483b8d58abf23a` |
| scripts/train_model_v2.py | `10a6b1290527eceeaa7861d39e3f45a58249f9739f9b8c5653286518b02b9177` |
| backend/app/services/payee_check.py | `d399ade83fe50f04aaddf0364d63cbcf35c359cf9f02ca51283a5c6ef3efac24` |
| backend/app/services/evidence.py | `6596b274a65007af0c8cd81f041b8187e479f9303228b2b466306d343e35fe4b` |
| backend/app/services/graph_cache.py | `d6206794406ebf091c53f5acdd366c9f2abbd2b83959136dde4d6b103e5de327` |
| backend/app/services/payee_reputation.py | `ef3f61356f312cb871100ac345815d70e27a4447ee453d51ec8d737fa6e45dc5` |

Hashes of all 24 experiment scripts and 118 result files: `reproducibility_manifest.json`.

## Commands

```bash
# from the repository root; JWT_SECRET must be set to any throwaway value
export OMP_NUM_THREADS=1                       # timing only
python scripts/final_v2/v2_classifier_analysis.py          # frozen V2, seed 42: tables 1-6, 12; figures 1-8, 11
python scripts/final_v2/v2_family_ablation.py --seeds 42 43 44 45 46 47 48 49 50 51   # 130 fits (~9 min on 2 vCPU)
python scripts/final_v2/v2_ablation_multiseed_summary.py   # tables 7, 8; figures 9, 10
python scripts/experiments/exp2b_template_disjoint.py --seeds 42 43 44 45 46 47 48 49 50 51   # template-disjoint control
for S in 42 43 44; do                                       # chronological replay (~95 min per stream)
  python scripts/experiments/exp3_4_replay.py --seed $S --n 60000 --db /tmp/replay_seed$S.db
  python scripts/experiments/exp4_rebuild_validation.py --seed $S --sample 60 --final-db /tmp/replay_seed$S.db
done
python scripts/final_v2/v2_end_to_end.py                    # ten-family re-aggregation with V2 in the Classifier family
python scripts/final_v2/v2_graph_and_replay_example.py      # graph statistics, before/assess/after example (reads the replay DBs)
python scripts/final_v2/v2_e2e_report.py                    # tables 9, 11; figures 12, 13, 15
python scripts/final_v2/v2_leakage_audit.py                 # LEAKAGE_AUDIT.md
python scripts/final_v2/v2_tests_graph_repro.py             # tables 10, 13; this manifest
python scripts/final_v2/polish_figures.py                   # final layout pass: figures 0, 2, 7-11 redrawn from saved results (no refit)
python scripts/final_v2/render_table_images.py              # table_images/*.png (needs node + playwright)
python scripts/final_v2/make_final_report.py                # FINAL_PROJECT_RESULTS.md
python -m pytest tests -q                                   # software tests (820)
```
