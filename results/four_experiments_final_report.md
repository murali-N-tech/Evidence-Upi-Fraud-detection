# Four Experiments — Internal Report

Evidence-Aware Pre-Payment Risk Assessment for UPI Transactions (synthetic evaluation)
Snapshot git HEAD: `9022a9feb65effa32779207e32458c0d37e9bb61` · Python 3.11.15, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, joblib 1.6.0

All numbers below are copied from the artefacts under `results/`. Every result is synthetic. None of them describes real UPI traffic.

---

## A. Part 0: repository audit (summary)

- **Running prototype.** The prototype serves V1: 12 features, `models/risk_model.pkl`, threshold 0.1174.
- **V2.** V2 has 47 features. It is frozen and offline only (`models/risk_model_v2.pkl`, threshold 0.1123).
- **Live assessment function.** It is `backend/app/services/payee_check.check_payee`.
  - The observation path is `payee_reputation.record_payment`, which updates the ledger, the aggregates and the graph in one transaction.
- **Architectural finding.** Payee age is computed from `backend.app.core.clock.now_app`, which is wall-clock time. A historical replay therefore needs clock injection.
  - This is the only runtime substitution made. It is a clock, not an evaluator.
  - The row-level corpora (`dataset.py`, `dataset_v2.py`) have no identities and no time order. For that reason a separate replay generator was added (`backend/ml/replay_corpus.py`). It is a new module, and the existing generators are unchanged.
- **Untouched artefacts.** Nothing in `models/`, `evaluation/` or `evaluation_v2/` was modified.

## B. Experiments executed

| # | Experiment | Scale | Status |
|---|---|---|---|
| 1 | 10-seed stability, V1 vs V2 | seeds 42–51 | done |
| 2 | Feature-family ablation (A–K) | 11 configs × 10 seeds = 110 fits | done |
| 2b | Template-disjoint message control (leakage fix) | configs E, I, K × 10 seeds | done |
| 3 | End-to-end ten-family evaluation | 3 replay streams × 60,000 payments, through `check_payee` | done |
| 4 | Chronological graph replay | 10 validation tests per stream; model variants A–D | done |
| 5 | Leakage audit | 17 items | done: 11 PASS, 0 FAIL, 5 WARNING, 1 INFO |

Additional replicates were added beyond the original request, for the following reasons:

- **Replay seeds 43 and 44.** In seed 42, every mule-collection ring fell outside the training window. Seeds 43 and 44 were pre-committed and run, and the results are reported as mean ± sd over three streams.
- **Template-disjoint control (2b).** It is the fix required once the audit found message-template overlap.

## C. Results

### Experiment 1: ten-seed stability

Test split: 30,000 payments per seed. Values are mean ± sd over seeds.

| Metric | V1 | V2 | Paired Δ | V2 better (seeds) |
|---|---|---|---|---|
| PR-AUC | 0.485 ± 0.022 (0.456–0.513) | 0.720 ± 0.023 (0.683–0.769) | +0.235 ± 0.027 | 10/10 |
| Recall | 49.4 ± 2.2% | 75.7 ± 3.2% | — | — |
| FPR | 0.99 ± 0.08% | 0.99 ± 0.08% | — | — |
| Brier | 0.00813 | 0.00573 | — | 10/10 |
| SE recall | 20.5 ± 2.7% | 62.4 ± 4.5% | — | — |
| MC recall | 61.2 ± 5.5% | 83.0 ± 3.7% | — | — |
| AT recall | ≈93% | ≈93% | no difference | 5/10 |
| CT recall | ≈97% | ≈97% | no difference | 2/10 |

**Seed 42 reproduction.** Seed 42 reproduces the frozen artefacts exactly:

- V1: PR-AUC 0.4860; TP/FP/FN/TN = 175/280/180/29,365.
- V2: PR-AUC 0.7326; TP/FP/FN/TN = 277/295/78/29,350.

### Experiment 2: feature-family ablation

Figures are the PR-AUC gain over the V2 core B, paired by seed, in percentage points.

| Configuration | ΔPR-AUC vs B (pp) |
|---|---|
| C +relationship | +2.8 ± 0.7 |
| D +graph | +11.2 ± 1.5 |
| E +message | +7.5 ± 1.9 |
| F +link | +1.7 ± 0.7 |
| G +intent | +4.0 ± 1.0 |
| H rel+graph | +14.0 ± 0.9 |
| I +message | +20.1 ± 2.6 |
| J +link | +20.2 ± 2.4 |
| K full V2 | +23.7 ± 2.7 |

- Every single-family addition is positive on 10/10 seeds.
- Graph is associated with the largest mule-collection recall gain (63.5% → 77.1%).
- Message is associated with the largest social-engineering recall gain (22.8% → 40.5%).
- **Template-disjoint control:**
  - Full V2 changes by −0.4 ± 0.7 pp (0.715 ± 0.022 vs 0.720 ± 0.023).
  - E changes by +0.0 ± 0.9 pp.
  - Exact-template memorisation therefore does not explain the message gain.

### Experiment 3: end-to-end ten-family evaluation

Three streams; last 25% of each; mean ± sd.

**Consistency audit.** For all 180,000 payments, the risk, coverage, level, bonus and verdict were recomputed from the recorded evidence. There were 0 mismatches.

**Risk ranking.**
- Ten-family risk R: PR-AUC 0.115 ± 0.018, ROC-AUC 0.848 ± 0.014.
- Live V1 alone, same rows: 0.084 ± 0.004 and 0.840 ± 0.015.

**Verdicts for legitimate payments:**

| Verdict | Share |
|---|---|
| APPROVE | 65.3 ± 0.7% |
| WARN | 22.1 ± 0.6% |
| STEP_UP | 5.8 ± 0.5% |
| BLOCK | 6.8 ± 0.6% |

**BLOCK rates by fraud scenario:**

| Scenario | Outcome |
|---|---|
| Social engineering | 71.2 ± 15.0% blocked |
| Mule collection | 57.6 ± 3.8% blocked |
| Account takeover | 19.5 ± 2.1% blocked |
| Card testing | 82.5 ± 7.3% **approved** |

**Operating points of R.**
- WARN or above: recall 82.2 ± 2.6%, FPR 34.7 ± 0.7%.
- BLOCK: recall 50.3 ± 5.7%, FPR 6.8 ± 0.6%.
- No threshold on R reaches FPR ≤ 1%, because 2.4 ± 1.1% of legitimate payments already sit at R = 99.

**Causes of legitimate BLOCKs.** The leading family is:
- stated intent (refund purpose) in 50 ± 5% of cases;
- message pressure in 26 ± 3%;
- payee history in 13 ± 9%.

**Cold start.** The share of legitimate payments blocked falls from 30.9% in days 0–30 to 6.5% in days 120–150. The cause is that payee age is measured from the store's first observation.

**Per-family findings.**
- Payer behaviour most often leads fraud assessments. Removing it lowers the verdict for 36% of fraudulent and 21% of legitimate payments.
- The positional graph finding fired on only 18 of 2,160 fraudulent payments.
- Corroboration bonus is paid for 65% of fraud and 18% of legitimate payments.
- The distinct-fact rule changed the bonus on only 3 of 180,000 payments.
- No unavailable family ever scored.

**Transfer.** The live V1 model, trained on the row-level corpus, recalls 5.5 ± 0.2% of replay fraud at its stored threshold. This is simulator-to-simulator shift.

### Experiment 4: chronological graph replay

**Validation tests.** All 10 tests passed on every stream:

| Tests | Reads | Violations |
|---|---|---|
| 1–3 and 10 | 180,000 each | 0 |
| 4–6 | 142,634 | 0 |
| 7 | 37,366 | 0 |
| 8: independent rebuild | 165 of 165 identical | 0 |
| 9: earlier observations unchanged | 90,000 of 90,000 | 0 |

Power probe: every sampled assessment would change if the future were visible.

**Predictive comparison.** Chronological 60/15/25 split; PR-AUC.

| Variant | PR-AUC |
|---|---|
| A: V1 on replayed state | 0.586 ± 0.037 |
| B: V2 + simulator graph channel (contrast; scenario-conditioned) | 0.749 ± 0.038 |
| C: V2 + replayed graph | 0.604 ± 0.117 |
| D: V2 without graph | 0.689 ± 0.045 |

- C − D = −8.5 ± 9.9 pp, and it is negative on every stream (−0.9, −4.9, −19.7).
- Replayed graph features are non-stationary. The median arrival span of legitimate payees is 27–30 days in the training period and 104–112 days in the test period.
- The replayed graph also departs from the simulator:
  - there is one giant component (median size 11,409);
  - the median mule overlap is 0.57, below the 0.6 gate.
- There are 3–4 mule rings per stream, and seed 42 has none in its training window.

## D. Leakage audit

File: `results/audit/leakage_audit.json`.

**PASS items:**
- split disjointness (all 10 seeds);
- threshold chosen on validation only (110/110 fits have validation FPR ≤ 1%; 56 fits exceed 1% on test, which shows the test set was not used to tune);
- no test-derived preprocessing;
- no best-seed selection;
- no future transactions in the store at assessment;
- the current payment is not in its own state;
- no future graph edges or payee history;
- the independent rebuild matches;
- historical records are not rewritten;
- the replay-model split is chronological;
- the live V1 model is not refitted on the replay.

**WARNING items:**
- **Message-template overlap.** The row-level overlap is 99.9–100%. Its effect is quantified by 2b and is negligible.
- **Payer overlap.** Random splits in Experiments 1–2 share payers across train and test.
- **Contrast variant B.** It is scenario-conditioned, which is intended.
- **In-sample templates.** The live message family scores texts from the same pool its coercion weights were fitted on.
- **Non-stationarity.** Replayed graph features are not stationary. This is a validity issue, not leakage.

**Result.** No leakage was found, so no result had to be withdrawn.

## E. Files generated

- **Scripts** (`scripts/experiments/`):
  - `common.py`
  - `exp1_2_multiseed_ablation.py`, `exp1_2_analysis.py`
  - `exp2b_template_disjoint.py`
  - `exp3_4_replay.py`, `exp3_e2e_analysis.py`, `exp3_4_combine.py`
  - `exp4_replay_models.py`, `exp4_rebuild_validation.py`
  - `leakage_audit.py`, `make_manifest.py`
  - plus `backend/ml/replay_corpus.py`
- **Multiseed** (`results/multiseed/`): `multiseed_results.csv`, `multiseed_summary.json`, `seed_configurations.json`.
- **Ablation** (`results/ablation/`): `ablation_results.csv`, `ablation_summary.json`, `ablation_paired_deltas.csv`, `template_disjoint_results.csv`, `template_disjoint_summary.json`, `run_metadata.json`.
- **End to end** (`results/end_to_end/`):
  - combined: `e2e_results.csv`, `e2e_summary.json`, `evidence_family_metrics.csv`, `e2e_audit_sample.jsonl`;
  - per seed: `*_seed{42,43,44}.*`, including the full audit logs `e2e_audit_full_seed*.jsonl`.
- **Graph replay** (`results/graph_replay/`):
  - combined: `replay_metrics.json`, `replay_validation.json`;
  - per seed: `replay_transactions_seed*.csv`, `replay_evidence_seed*.csv` (compressed as `.csv.gz` on the device), `replay_metrics_seed*.json`, `replay_model_results_seed*.csv`, `replay_validation_seed*.json`, `replay_run_seed*.json`.
- **Other:**
  - `results/audit/leakage_audit.json`
  - `results/four_experiments_manifest.json` (SHA-256 of every output, script and model)
  - `results/tables/`: Tables A–E, per seed where applicable
  - `results/figures/`
  - `results/paper/`: the exact tables, figures and text facts used in the manuscript

## F. Reproduction commands

Run from the repository root, using the project virtual environment. `JWT_SECRET` must be set to any value; use a throwaway, never a real secret.

```bash
export OMP_NUM_THREADS=1        # timing only
python scripts/experiments/exp1_2_multiseed_ablation.py --seeds 42 43 44 45 46 47 48 49 50 51
python scripts/experiments/exp1_2_analysis.py
python scripts/experiments/exp2b_template_disjoint.py --seeds 42 43 44 45 46 47 48 49 50 51
for S in 42 43 44; do
  python scripts/experiments/exp3_4_replay.py --seed $S --n 60000 --db /tmp/replay_seed$S.db     # ~95 min each
  python scripts/experiments/exp3_e2e_analysis.py --seed $S
  python scripts/experiments/exp4_replay_models.py --seed $S
  python scripts/experiments/exp4_rebuild_validation.py --seed $S --sample 60 --final-db /tmp/replay_seed$S.db
done
python scripts/experiments/exp3_4_combine.py --seeds 42 43 44
python scripts/experiments/leakage_audit.py
python scripts/experiments/make_manifest.py
```

Replay runtime was about 5,650–5,840 s per stream on 2 vCPUs.

## G. Runtime substitutions (none change a prediction)

- `clock.now_app` is set to the replay time; replay only.
- `re._MAXCACHE = 8192`, because the message analysers recompile regexes otherwise.
- `OMP_NUM_THREADS=1` for single-row prediction.

## H. Performance observations (not changed)

- The live classifier family reloads the model pickle on every call (`load_model()` inside `_ml_evidence`). This is about 24% of assessment time in profiling.
- Counterfactual fact probes call `predict_proba` several times per assessment.

## I. Remaining limitations

- **Synthetic data only.** The V2 channels, the replay design and the message pool are all author-specified.
- **Message pool.** It is English-only and in one authorial style.
- **Replay coverage.** The replay covers 3 streams of 150 days with 3–4 mule rings each.
- **Fixed rules.** Rule weights and verdict bands are fixed and untuned. This produces the high intervention rate.
- **Cold start.** Payee evidence depends on how long the store has been observing.
- **Graph features.** Replayed graph features need time normalisation.
- **No deployment.** There is no deployment, no provider integration and no latency benchmark.
- **Reference papers.** The two published UPI papers supplied as style references were not accessible in this session. They are therefore not cited, and no bibliographic details were invented.

## J. What changed in the manuscript

- **Restructured.** The paper now has five sections (Introduction, Methodology, Experimentation, Results and Discussion, Conclusion) plus References.
- **New title:** "Evidence-Aware Pre-Payment Risk Assessment for UPI Transactions Using Behavioural, Contextual, and Payer–Payee Graph Evidence".
- **Tables and figures.** Six tables and five figures. Tables III–VI and Figs. 4–5 are generated only from the artefacts (`ieee8/make_tables.py`). The running-text numbers for Sections IV-D/E come from `tabs/facts.json`.
- **References.** 18 references, all cited. Five uncited references were dropped from the previous version: CARE-GNN, Weber, Zadrozny, SHAP, Breiman.
- **Length.** About 8 pages of text and floats, plus a reference tail on page 9, in A4 IEEE two-column format.

## K. Verification

- **Seed-42 reproduction:** exact.
- **Consistency audit:** 0/180,000 mismatches.
- **Replay tests:** all pass.
- **Leakage audit:** 0 FAIL.
- **Manuscript numbers:**
  - Hand-typed numbers in Sections IV-A to IV-C, III and the abstract were checked against `multiseed_summary.json`, `ablation_summary.json`, `template_disjoint_summary.json` and the seed-42 row of `ablation_results.csv`.
  - All other result numbers are substituted programmatically.
