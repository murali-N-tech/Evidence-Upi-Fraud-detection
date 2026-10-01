# STAGE 6 — FINAL AUDIT

**Manuscript audited:** Stage 2 (Abstract, §1–2), Stage 3 (§3–5 and the Fig. 1–2 / Table 1–4 specifications), Stage 4 (§6–9) and Stage 5 (§10–13), about 28,000 words in total.

**Artifacts checked (22 Sep 2026):**
- `evaluation/metrics.json`, `evaluation_v2/metrics.json`, `models/metrics.json`, `models/metrics_v2.json`
- `evaluation_v2/ablation_results.csv`, `baseline_comparison.csv`, `development/*`
- both manifests, with SHA-256 re-checked
- `backend/` source code
- a fresh run of five test modules
- the Spade+, PGLTSE and Secure UPI text extracts
- Crossref records and publisher pages for reference metadata

**Status labels:** A = CONFIRMED · B = INCONSISTENT · C = UNSUPPORTED · D = AMBIGUOUS · E = NEEDS EXTERNAL VERIFICATION · F = CORRECT BUT SHOULD BE REPHRASED

---

## 1. Overall Status

**REQUIRES SUBSTANTIAL REVISION**

All numbers in the manuscript match the artifacts, and the separations between V1 and V2, calibrated and uncalibrated models, and ML evaluation and software validation hold in Sections 6–13. Four problems stop the paper from being submittable as it stands:

1. The manuscript has no reference list.
2. Section 1.6 uses the old section numbering.
3. Section 4.7 contradicts the running code.
4. Several statements about missing evidence are contradicted by the implementation:
   - an unseen payee does add a small amount of risk;
   - the V1 classifier learns a response to missing payee history;
   - the MINIMAL evidence level cannot occur for any valid request.

The evaluated code is also not committed. Each fix is small, but together they touch claims in the Abstract and in Sections 1, 2, 3, 4 and 8.

---

## 2. Critical Issues

**CR-1**
- **Location:** Whole manuscript (end matter)
- **Problem:** There is no References section. Citations [1]–[22] appear in the text, but no bibliography is assembled. [21] and [22] exist only in a "New references" note at the end of the Stage 4 file.
- **Evidence:** `grep` of all four stage files finds no reference list.
- **Required correction:** Add an IEEE-formatted list [1]–[22] using the verified metadata in §9 of this audit. Remove the stage notes and checklists from the manuscript body.

**CR-2**
- **Location:** §1.6
- **Problem:** B. The organisation paragraph uses the old numbering: "Section 5 explains how the synthetic data were constructed… Section 12 concludes".
- **Evidence:** Stage 2 text. The final structure has 13 sections.
- **Required correction:** Rewrite §1.6 to match §1–§13 as listed in Audit 4.

**CR-3**
- **Location:** §4.7, second paragraph
- **Problem:** B. The paragraph says model-level attributions "are computed offline… not shown to the payer".
- **Evidence:** `backend/main.py` exposes `GET /explain/{tx_id}`, which uses `shap.Explainer` on the calibrated V1 `predict_proba` with `models/shap_background.json`. `src/components/Explainability.tsx` renders the result. §7.7 already describes this correctly, so §4.7 and §7.7 contradict each other.
- **Required correction:** Rewrite §4.7 to say the following, and align §2.7 ("complements these with model-level attributions") with it:
  - offline analysis uses TreeExplainer on the first fold's *uncalibrated* booster, so its magnitudes are on the raw booster scale;
  - the running app also offers an explanation screen that computes model-agnostic SHAP on the *calibrated* V1 probability;
  - neither feeds the verdict.

**CR-4**
- **Location:** §3.3 C3, §3.6, §8.2 (sentence "A first payment to an unobserved payee… yields a lower evidence level and lower coverage, not a higher risk"), §8.4 example, §11.4
- **Problem:** B. When an address parses but has never been observed, the payee-history family is still available. It raises `payee_unseen` (severity warn, weight 6), which contributes to risk. It does not lower coverage or the evidence level. Two consequences follow:
  - The evidence level can only be MINIMAL when no address could be read, and that request is already hard-blocked (`no_payee` / `empty`). So MINIMAL does not occur for any valid request.
  - "Unseen" is treated in code as a measured observation of weight 6, below the WARN threshold of 22. That is defensible, but it is not what the paper says.
- **Evidence:** `payee_check.py` lines 530 and 712–720 (family unavailable only when `reputation is None`, i.e. no address). `payee_reputation.py` line 451 (`payee_unseen`, weight 6). `evidence.py` evidence-level rule.
- **Required correction:**
  - State that "never observed by this deployment" is treated as a low-weight measured finding (6 points, below the WARN band) that must be corroborated by other evidence to change the verdict.
  - Reword C3 as "absence must not by itself change the verdict".
  - Remove or correct the §8.2 sentence.
  - Say in §3.6 that MINIMAL applies only when no address can be read.
  - Do not change the code to match the paper unless you intend to.

**CR-5**
- **Location:** Contribution 2 (§1.5), §3.3 C2/C3, §8.2 ("Unavailable evidence never contributes to risk in either direction"), §10.5 ("the architecture does not treat absence as a risk factor")
- **Problem:** F/B. The guarantee holds at the aggregation level only: an unavailable family carries no score. Inside the classifier family, however, `payee_history_available` is an input feature and missing payee features are NaN values the model learns a response to. The training data have a higher fraud rate where history is hidden (1.729% vs 1.005%), so V1's probability can rise when history is missing.
- **Evidence:** `backend/ml/features.py` (V1 feature list), `dataset.apply_observation_mask`, `models/metrics.json` → `evidence_availability`.
- **Required correction:** Scope each claim to aggregation-level behaviour. Add one sentence each to §6.2 and §8.2 explaining that, inside the classifier, missingness is an explicit input whose learned effect is monitored through the availability-bias check and tests, and is limited by the classifier's 66-point ceiling.

**CR-6**
- **Location:** Abstract
- **Problem:** F/B. Four problems:
  1. The last sentence says the results "characterise the architecture". The predictive results characterise the *classifier component*.
  2. The Abstract does not say that V2 graph features are simulated and that the incremental graph is not executed in the ML evaluation.
  3. It does not say there is no end-to-end evaluation of the ten-family system.
  4. Its rounding (1.18%, 0.733, 0.486) differs from the body (1.183%, 0.7326, 0.4860).
- **Evidence:** Stage 2 abstract text; §9.4, §12.2 and §12.4.
- **Required correction:** Replace the last two sentences with an ML-component statement along these lines:
  > "In an offline evaluation of the classifier component… the V2 representation reaches a PR-AUC of 0.7326… Graph, message, link and purpose channels in V2 are simulator-generated; the incremental graph and the evidence-aggregation logic are validated separately by software tests, and the full ten-family verdict system is not benchmarked end to end."

  Use one precision convention throughout.

**CR-7**
- **Location:** §2.1, third paragraph ("The present study follows the second approach… trained on data with the natural fraud rate")
- **Problem:** B. The classifier uses `class_weight="balanced"`. Class weighting shifts the effective class prior in the same way the paragraph criticises resampling for doing. This contradicts §7.4.
- **Evidence:** `train_model.py` line 255 and `train_model_v2.py candidates()`.
- **Required correction:** Rephrase as: "trained without resampling, with balanced class weights, and then isotonically calibrated so that probabilities return to the observed base rate."

**CR-8**
- **Location:** §2.2 ("computes them strictly from payments that precede"), §2.4 ("graph and relationship features are computed only from observations that precede"), Contribution 3
- **Problem:** F. These claims hold for the **live path** only. In the offline corpus, features are sampled per row, not computed from a ledger. Historical replay is not guaranteed (§6.5, §12.6).
- **Evidence:** `dataset.py` and `dataset_v2._graph_row`; Stage 4 §6.5.
- **Required correction:** Add "in the live assessment path" to each, and cross-reference §6.5.

**CR-9**
- **Location:** Reproducibility (§9.10, §12.11 "fully reproducible")
- **Problem:** C. The evaluated code is not in a committed state:
  - 81 working-tree changes are uncommitted, including `payee_check.py`, `payee_reputation.py`, `vpa.py`, `upi_qr.py` and `main.py`;
  - `backend/app/core/clock.py` is untracked;
  - `scripts/evaluate_model_v2.py` does not match its manifest hash.

  All *model and data* artefact hashes do match, as the Evidence line lists.
- **Evidence:** `git status` (HEAD 9022a9f, 19 Sep 2026); manifest SHA-256 re-check.
- **Required correction:**
  - Commit and tag the evaluated state.
  - Regenerate or annotate the V2 manifest entry for `evaluate_model_v2.py`.
  - Add a code-and-data availability statement giving the commit hash.
  - Change "fully reproducible" to "reproducible from the tagged commit, seed and hashed artefacts".

**Note on the audit brief:** it gives the V1 confusion matrix as FN = 175, TP = 180. That is **swapped**. The artifacts give **TP = 175, FN = 180**, and the manuscript (Table 5 and §10.2) already uses the correct values. Do not "correct" the manuscript to the brief's values.

---

## 3. Numerical Consistency Audit

| Metric | Value | Where used | Correct? | Notes |
|---|---|---|---|---|
| Transactions | 120,000 | Abstract, §6.1, §9.1, §10.1, §13 | A | `metrics.json` rows |
| Payers | 2,000 | Abstract, §6.1, §9.1, §10.1 | A | fingerprint `users` |
| Fraud prevalence | 1.183% (1,419) | §6.1, §7.4, §9.1, §10.1, §12.1, §13 | A | 0.011825. The Abstract's "1.18%" is correct but uses different precision (CR-6). |
| Train / Val / Test | 72,000 / 18,000 / 30,000 | §9.1, §10.1 | A | Fraud counts 851 / 213 / 355 also A |
| Scenario mix | 50 / 22 / 20 / 8% | §6.1, §12.1 | A | `SCENARIO_MIX`. Stated as a design choice, not a prevalence estimate. |
| Test scenario counts | 182 / 80 / 73 / 20 | §9.1, Table 6 | A | |
| Hidden-history rows | 29,376 | §6.2, §10.5 | A | |
| Hidden share | 24.48% | §6.2, §9.7, §10.5 | A | 0.2448 |
| Fraud rate, hidden | 1.729% | §6.2, §10.5, §11.4 | A | |
| Fraud rate, visible | 1.005% | §6.2, §10.5, §11.4 | A | |
| Legitimate share among hidden | 98.3% | §10.5, §11.4 | A | 1 − 0.01729 |
| V1 PR-AUC | 0.4860 | Table 5, §10.2, §11.2, §13 | A | Abstract "0.486" (CR-6) |
| V1 ROC-AUC | 0.9652 | Table 5 | A | |
| V1 recall @ ≈1% FPR | 49.3% (0.4930) | Table 5, §10.2, §11.2 | A | Realised FPR 0.00945 |
| V1 precision / F1 | 0.3846 / 0.4321 | Table 5 | A | Recomputed |
| V1 SE recall | 19.23% (35/182) | Table 6, §10.2, §11.1, §13 | A | |
| V1 mule recall | 63.75% → 63.8% (51/80) | Table 6, Table 7, §10.2 | A | 63.7% appears nowhere |
| V1 ablation SE (payer / payee / both) | 3.85 / 11.54 / 19.23% | §10.2, §11.1 | A | `models/metrics.json` ablation |
| SE ablation difference | 7.69 pp → ≈ +7.7 pp | §9.6, §10.2, §11.1, §13 | A | Always labelled as social-engineering-specific |
| V1 uncalibrated ablation reference | 0.4996 | Table 7, §10.4 | A | Labelled uncalibrated |
| V1 Brier / BSS | 0.00798 / 0.317 | Table 5, §10.6 | A | BSS 0.31738 in the artifact |
| V1 confusion matrix | TN 29,365 · FP 280 · FN 180 · TP 175 | Table 5, §10.2 | A | Sums to 30,000; TP + FN = 355. The brief's values are swapped (see note in §2). |
| V1 threshold | 0.1174 | §7.5, Table 5 | A | |
| V2 PR-AUC (frozen) | 0.7326 | Table 5, §10.3, §11.2, §13 | A | Abstract "0.733" (CR-6) |
| V2 ROC-AUC | 0.9900 | Table 5, §10.3 | A | 0.98997 |
| V2 recall | 0.7803 | Table 5, §10.3, §13 | A | 277/355 |
| V2 FPR | 0.00995 | Table 5, §10.3 | A | 295/29,645 |
| V2 precision | 0.4843 | Table 5, §10.3 | A | 277/572 |
| V2 F1 | 0.5976 | Table 5, §10.3 | A | Recomputed |
| V2 Brier / BSS | 0.00555 / 0.525 | Table 5, §10.3, §10.6 | A | Reference = constant base-rate predictor, 0.01169; defined in §7.6 |
| V2 confusion matrix | TN 29,350 · FP 295 · FN 78 · TP 277 | Table 5, §10.3 | A | Sums to 30,000 |
| V2 threshold | 0.1123 | §7.5, Table 5 | A | |
| V2 validation (PR-AUC / recall) | 0.6995 / 0.746 | §10.3 | A | `operating_point.validation_results` |
| V2 scenario recall | 65.93 / 85.00 / 94.52 / 100% | Table 6, §10.3 | A | 120/182, 68/80, 69/73, 20/20 |
| Alerts V1 / V2 | 455 / 572 | Table 5, §11.6 | A | |
| Ablation full V2 (uncalibrated) | 0.7234 | Table 7, §10.4 | A | Never called the V2 result |
| Model families (uncalibrated) | GB 0.7234 · RF 0.6915 · ET 0.5145 · LR 0.5055 | §10.4 | A | `baseline_comparison.csv`, `calibrated = False` |
| Ablation rows A–F, leave-one-out | All values in Table 7 | Table 7 | A | Rounded values re-checked row by row |
| Permutation importance (V2 top 5) | 0.366 / 0.246 / 0.206 / 0.138 / 0.121 | §10.6 | A | payee_is_new, arrival_span_days, amount_over_user_avg, msg_coercion_probability, component_size |
| SHAP (V2 order) | payee_is_new > arrival_span_days > amount_over_user_avg > component_size | §10.6 | A | |
| Calibration bins (V2, uniform) | e.g. 0.347 vs 0.435 (n = 46); top bin 71/71 | §10.6 | A | |
| Tests | 91 passed | §10.7 | A | Re-run 22 Sep 2026: 91 passed in 27.6 s. Add date/commit (CR-9). |
| Query-count bound | ≤ +2 queries after 400 extra payments | §9.9, §10.7 | A | `test_the_read_path_stays_bounded…` |
| Coercion corpus | 1,022 (364 / 557 / 101) | §6.4, §12.3 | A | `coercion_weights.json` |
| Isotonic folds, ≈170 positives per fold | 5 folds, 851 / 5 | §12.9 | A | Approximation, labelled "roughly" |

**Precision note:** the manuscript mixes 2-, 3- and 4-decimal precision (e.g. 49.3% beside 0.7803). **F** — choose one convention per metric type, e.g. 4 d.p. for AUCs and rates in tables, and one decimal in prose percentages.

---

## 4. V1/V2 Consistency Audit

| Item | Status | Finding |
|---|---|---|
| Running V1 | A | Consistently called the model the prototype serves (§4.5, §6.4, §7.2, §8.1, Table 4, Table 5). |
| Frozen offline V2 | A | Never described as deployed or live (§4.5, §6.4, §7.3, §9.10, §10.3, §13). |
| Calibrated vs uncalibrated | A | Table 5 is calibrated; Table 7 and the model-family comparison are uncalibrated, with explicit captions (§9.6, §10.4). |
| 0.4860 | A | Frozen, calibrated V1 on the test split. |
| 0.7326 | A | Frozen, calibrated V2 on the test split. |
| 0.7234 | A | Uncalibrated full-V2 ablation row, explicitly "not the V2 result". |
| 0.4996 | A | Uncalibrated V1 reference row, explicitly distinguished from 0.4860. |
| Table 7, row C | F | "C. B + velocity" has 12 features but is **not** V1 (it has `payee_is_phone_address` and lacks `payee_is_new`). This is explained in prose only; add a table footnote. |
| §4.5 V1 description | F | The prose summary omits the raw `amount` feature. Either list all 12 or refer to Table 4. |
| §4.5 "each group carrying an explicit availability indicator where its source can be absent" | D | Payer, amount and velocity groups have none; the link group uses `message_available`. Rephrase to "groups whose source can be absent (payee, relationship, graph, message, intent) carry an availability indicator; link features share the message indicator." |

---

## 5. Graph Evaluation Audit

| Check | Status | Finding |
|---|---|---|
| Graph implementation described as incremental, updated in the observation path, read-only in the check path | A | §4.4, §5.6. Matches `graph_cache.py` and the tests. |
| Graph never rebuilt during a check | A | §4.4, §9.8, §10.7; `test_a_pre_payment_check_does_not_rebuild_the_graph`. |
| V2 graph features simulator-generated | A | §6.3, §9.8, §10.3, §10.4, §11.3, §12.2. |
| Graph implementation validated by software tests only | A | §9.8, §10.7. |
| No end-to-end graph predictive evaluation claimed | A | §9.8 and §12.2 explicitly. **Exception:** the Abstract is silent on this (CR-6). |
| Rule-based graph family not evaluated on labelled data | A | §9.8, §12.2. |
| MIN_FAN_IN = 10, ≥ 4 shared payers, overlap ≥ 0.6, repeat < 0.1, 30-day burst, ≥ 5 per day | A | §5.6 matches `graph_cache.py` constants. |
| Popular-merchant (peer repeat-ratio) discount | A | §5.6, §6.3. |
| No GNN claim for this system | A | "GNN" appears only for other literature. **Risk:** the repository contains `backend/app/services/temporal_gnn.py`, imported by `main.py`. It is an hour-of-day histogram, not a GNN, but a reviewer reading released code may take it for one. **SHOULD FIX:** rename the module, or add a code-availability note. |
| Title term "Incremental Payment Graph" | D | Accurate for the architecture. It does not imply that the experiments executed the graph, provided CR-6 is fixed. Optional wording: "Incremental Payer–Payee Graph". |

---

## 6. Evidence Aggregation Audit

| Check | Status | Finding |
|---|---|---|
| Damped max (β = 0.4, cap 99) | A | §5.8 = `combine_scores`. |
| Participation threshold ζ = 20; strength k = min(\|facts\|, \|families\|); bonus table 6/14/22/28/32 | A | §5.9 = `agreement_bonus`. |
| Distinct-fact counting (no double counting) | A | §5.9, §8.3. |
| ML participation (the stale `ML_CASTS_AGREEMENT_VOTE` comment) | A | §8.3 describes actual behaviour: `NON_VOTING` is empty and facts are counterfactually attributed. |
| Counterfactual attribution (15% relative, 0.01 floor; payee probe → NaN, others → benign values) | A | §5.7 = `_ml_facts`, `ML_PROBES`. |
| ML ceiling of 66, score mapping 0–18 / 30–66 | A | §5.7. |
| Payer-behaviour mapping (< 25 → 0; else 12–52) | A | §5.4. |
| Coverage weights (sum 100) | A | §5.10 = `CONFIDENCE_WEIGHT`. |
| Risk vs confidence separation | A/F | §8.4 is correct. **Terminology:** §3–5 say "coverage score C(x)", §8.4 says "Confidence, reported as the coverage score". Define once in §3.1 ("coverage, which we also refer to as evidence confidence") and use "coverage" afterwards. |
| Availability semantics | **B** | CR-4 (unseen payee adds weight 6; MINIMAL cannot occur for a valid request) and CR-5 (classifier-internal missingness). |
| Verdict bands 0–21 / 22–44 / 45–69 / ≥ 70 (max 99) and hard-block set | A | §5.11, §8.6 = `core/verdict.py` and `HARD_BLOCK`. |
| APPROVE ≠ safe | A | §3.1, §5.11, §8.6, §11.6. |
| EntityStatus not scored; VERIFIED needs an external record | A/F | §3.4, §8.2 are correct. **But** §4.8 and §12.7 ("no connection to any bank or PSP") omit that `backend/app/services/upi_verify.py` provides an *optional, unconfigured* PSP verification interface (via environment variables). Rephrase: "includes an optional provider-verification interface that was not configured in this study." |
| Cross-reference | B | §4.3 (address family) cites "(Section 5.10)" for hard-block findings. It should be **§5.11**. |

---

## 7. Figure Audit

| Figure | Issue | Severity | Correction |
|---|---|---|---|
| Fig. 1 (architecture) | Exists only as a specification. The spec is consistent with the code (read-only check path, dashed observation path, V2 outside the live path). | Must fix (render) | Render it. Add the `/explain` explanation screen as a read-only consumer of the V1 model (CR-3). Do not draw an NPCI/PSP box. If the optional verifier is shown, mark it "not configured". |
| Fig. 2 (graph illustration) | Spec only. Arithmetic checked: ω(X, Y) = 1/min(3, 2) = 0.5, then 2/min(4, 2) = 1.0. The caption rightly says "illustrative" and that no finding fires. | Must fix (render) | Render it. Keep "illustrative — not experimental data". |
| Result figures (PR curve, ROC curve, confusion matrix, scenario recall, permutation importance, SHAP summary, calibration curve, threshold sweep, recall vs FPR) | They exist in `evaluation/figures` (V1) and `evaluation_v2/figures` (V2) but **no figure is cited in §10**. | Should fix | Include a selected subset: PR curves V1 vs V2, scenario recall, V2 calibration (uniform bins), V2 permutation importance. Label each figure V1 or V2, calibrated or uncalibrated, and add "synthetic held-out test split (n = 30,000)". The SHAP summary caption must say "uncalibrated first-fold booster". |
| Any graph figure | No result figure depicts the running graph. | — | NO ISSUE FOUND. |
| Missing-evidence figure | None exists; values are given in prose. | Optional | A two-bar chart of fraud rate (hidden vs visible) could be added, captioned "simulated class distributions". |
| Figure numbering vs table numbering | Fine. | — | NO ISSUE FOUND. |

---

## 8. Table Audit

| Table | Issue | Severity | Correction |
|---|---|---|---|
| Order of first citation | Tables are cited out of order: Table 2 (§4.3), Table 4 (§4.5), Table 1 (§5), Table 3 (§6.2). IEEE requires numbering by first citation. | Should fix | Renumber: Notation → I, Families → II, V1/V2 → III, Feature groups → IV. Alternatively, cite Table 1 in §3.1 and Table 3 in §4.5. |
| Tables 1–4 | Specifications only; not typeset. | Must fix | Render them. Remove the working notes ("Rows ordered by…", "Table note: …") or turn them into footnotes. |
| Table 1 | The `a` (amount) vs `a_i` (availability) clash is handled by a footnote. | Optional | Readable as is. If a reviewer objects, use α_i for availability. |
| Table 2 | Fact assignments checked against code. Availability wording for payee history ("address can be looked up") is correct, but it should state that an unseen payee is *available* with a weight-6 finding (CR-4). | Must fix | Add "unseen → `payee_unseen`, weight 6". |
| Table 3 | Content correct. "V2 group counts" = 2 + 2 + 3 + 5 + 4 + 7 + 18 + 3 + 3 = 47. | — | NO ISSUE FOUND (after rendering). |
| Table 4 | The "Reported results" row says "Section on results". | Should fix | Change to "§10, Table 5". |
| Table 5 | All values verified. Caption states the calibration status implicitly ("calibrated classifiers"). | Optional | Add "isotonic-calibrated" and "threshold from validation (1% FPR budget)" to the caption. |
| Table 6 | Values and counts verified. | — | NO ISSUE FOUND. |
| Table 7 | Values verified; the caption states "uncalibrated". The model-family comparison is prose only. | Should fix | Add a footnote for row C ≠ V1. Optionally add a separate Table 8 for the model-family comparison, captioned "uncalibrated, same protocol as Table 7". |
| IEEE caption style | Markdown captions sit below the tables. | Should fix | IEEE: "TABLE I" plus a caption *above* each table; "Fig. 1." *below* each figure. |

---

## 9. Reference Audit

### Confirmed (A)

| # | Reference | Verified metadata |
|---|---|---|
| [1] | S. Rastogi, C. Panse, A. Sharma, V. M. Bhimavarapu, "Unified Payment Interface (UPI): A digital innovation and its impact on financial inclusion and economic development" | *Universal J. Accounting and Finance*, 9(3), 518–530, 2021, **doi:10.13189/ujaf.2021.090326** (Crossref). Use all four authors. |
| [3] | A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, G. Bontempi, "Credit card fraud detection: A realistic modeling and a novel learning strategy" | *IEEE TNNLS*, 29(8), 3784–3797, 2018 |
| [4] | J. Braithwaite, "'Authorized push payment' bank fraud: What does an effective regulatory response look like?" | *J. Financial Regulation*, 10(2), 174–193, 2024, doi:10.1093/jfr/fjae006 |
| [5] | J. Jiang, Y. Chen, B. He, M. Chen, J. Chen, "Spade+: A generic real-time fraud detection framework on dynamic graphs" | *IEEE TKDE*, 36(11), 7058–7073, 2024, doi:10.1109/TKDE.2024.3394155 |
| [6] | B. Hooi, H. A. Song, A. Beutel, N. Shah, K. Shin, C. Faloutsos, "FRAUDAR: Bounding graph fraud in the face of camouflage" | *Proc. 22nd ACM SIGKDD*, 2016, pp. **895–904**, doi:10.1145/2939672.2939747 |
| [7] | J. Ma, S. Xiang, Q. Li, L. Yuan, D. Cheng, C. Jiang, "Parallel graph learning with temporal stamp encoding for fraudulent transactions detections" | *IEEE Trans. Big Data*, 11(4), 1945–1958, 2025, doi:10.1109/TBDATA.2024.3499338 |
| [8] | R. Rani, A. Alam, A. Javed, "Secure UPI: Machine learning-driven fraud detection system for UPI transactions" | *2024 2nd Int. Conf. Disruptive Technologies (ICDT)*, 2024, pp. **924–928**, doi:10.1109/ICDT61202.2024.10489682. **Conference paper.** Page range read from the PDF footers (924 on the first page, 928 on the last), which settles the earlier "924 or 925" question. |
| [11] | T. Saito, M. Rehmsmeier | *PLoS ONE*, 10(3), e0118432, 2015 |
| [12] | J. Davis, M. Goadrich | *Proc. 23rd ICML*, 2006, pp. **233–240**, doi:10.1145/1143844.1143874 |
| [15] | N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE" | *JAIR*, 16, 321–357, 2002, doi:10.1613/jair.953 |
| [16] | E. A. Lopez-Rojas, A. Elmir, S. Axelsson, "PaySim: A financial mobile money simulator for fraud detection" | *Proc. European Modeling and Simulation Symp. (EMSS)*, 2016, pp. **249–255** (ISBN 978-88-97999-76-8; no DOI) |
| [17] | A. Niculescu-Mizil, R. Caruana | *Proc. 22nd ICML*, 2005, pp. **625–632**, doi:10.1145/1102351.1102430 |
| [19] | G. W. Brier | *Monthly Weather Review*, 78(1), 1–3, 1950 |
| [22] | L. Breiman, "Random forests" | *Machine Learning*, 45(1), 5–32, 2001, doi:10.1023/A:1010933404324 |

### Needs verification (E)

| # | What to check |
|---|---|
| [2] | Bolton & Hand, *Stat. Sci.* 17(3), 2002, doi:10.1214/ss/1042727940. Crossref has no page field; confirm pp. 235–255 on Project Euclid. |
| [9] | CARE-GNN (Dou, Liu, Sun, Deng, Peng, Yu), *CIKM 2020*, doi:10.1145/3340531.3411903. The DOI resolves; confirm pp. 315–324 (Crossref was rate-limited). |
| [10] | Weber et al., arXiv:1908.02591. Decide whether to cite it as the arXiv preprint or as the KDD 2019 Workshop on Anomaly Detection in Finance paper, and verify that venue if you use it. |
| [13] | LightGBM (Ke et al.), *NIPS 30*, 2017. Confirm pp. 3146–3154 in the proceedings. |
| [14] | Pedregosa et al., *JMLR* 12, 2011. Confirm pp. 2825–2830. |
| [18] | Zadrozny & Elkan, *KDD 2002*, doi:10.1145/775047.775151. The DOI resolves; confirm pp. 694–699. |
| [20] | Lundberg & Lee, "A unified approach to interpreting model predictions," *NIPS 30*, 2017. Confirm pp. 4765–4774. |
| [21] | Friedman, *Ann. Statist.* 29(5), 2001, doi:10.1214/aos/1013203451. Crossref has no page field; confirm pp. 1189–1232. |

### Incorrect

- None found in the metadata recorded in Stages 2 and 4. The only earlier uncertainties (Secure UPI pages, PaySim pages, Rastogi DOI) are now resolved above.

### Missing

- **The entire bibliography (CR-1).**
- No in-text citation lacks a corresponding reference number.

### Uncited

- None. All of [1]–[22] are cited. First-use order is monotonic ([1] in §1.1 through [22] in §7.7), and there are no duplicates.

### Content checks against the reference texts

| Claim | Status | Finding |
|---|---|---|
| Spade+ handles insertions **and deletions** of edges; industrial setting (Grab) | A | Confirmed in `spade.txt`. |
| PGLTSE uses a continuous-time dynamic graph with gang-aware risk propagation | A | Confirmed in `pg.txt`. |
| PGLTSE evaluated on healthcare-claims, e-commerce and credit-card datasets | E | The extract confirms a credit-card dataset (Sparkov) and multiple "real-world datasets". Re-check the full dataset list in its experiments section before submission. |
| Secure UPI uses SMOTE, PCA and XGBoost and reports accuracy (XGBoost 98.2%) | A | Confirmed. |
| Secure UPI "compares several alternatives using accuracy" | A | Its Table 2 is titled "Comparison of accuracy". |
| Secure UPI dataset source and size not reported | D | The extract only says "a labelled dataset". Re-read the full PDF once before asserting this. |
| §1.3 describes Secure UPI as "gradient-boosted ensembles with resampling" | F | Say "an XGBoost classifier with SMOTE oversampling" for consistency with §2.5. |

---

## 10. IEEE Style Audit

1. **Section numbering:** IEEE Transactions use Roman numerals for sections (I., II., …) and letters for subsections (A., B., …). Convert when typesetting. The §1.6 content must also be fixed (CR-2).
2. **Equation numbering:** none of the display equations in §5 are numbered. Number them (1)…(n) and refer to them by number in §6–§8.
3. **Abbreviations to define at first use:**
   - PR-AUC (first used in the Abstract)
   - ROC-AUC (§1.3)
   - SHAP (§2.7; SHapley Additive exPlanations)
   - NPCI (§4.8; National Payments Corporation of India)
   - PSP (write it out once)
   - FPR (define in §7.4 before §9.3)
   - IST (§4.2 ✓)
   - UPI, VPA, APP and GNN are already defined correctly.
4. **Figure and table captions:** use "Fig. n." below figures and "TABLE N" above tables.
5. **References:** IEEE style needs initials before surnames, abbreviated journal names (e.g., *IEEE Trans. Knowl. Data Eng.*), "vol. x, no. y, pp. a–b, Mon. Year", and DOIs in the form "doi: 10.xxxx/…".
6. **Working material in the body:** the "New references…" block and all stage checklists must be removed. Code identifiers such as `address_and_qr` and `ML_CASTS_AGREEMENT_VOTE` are acceptable in monospace but should be kept to a minimum; §8.3 could describe the stale constant as "a residual configuration flag".
7. **Spelling:** the paper uses British spelling throughout (behaviour, authorised, normalise). IEEE accepts it if it is consistent, and it is.
8. **Units and symbols:** amounts in "Rs" should be written consistently ("Rs 270" or "₹270"). Time units are correct. Percentages and percentage points are used correctly.
9. **Keywords:** IEEE journals typically expect alphabetical order with "Index Terms" as the label.

---

## 11. Reproducibility Audit

These items are missing and are needed for independent reproduction:

1. **Code availability:** there is no repository URL or commit hash, and the evaluated state is uncommitted (CR-9).
2. **Software environment:** Python 3.10.12, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, joblib 1.6.0 and shap (version recorded in the manifest) are not stated in the paper.
3. **Simulator parameters:**
   - §6.1 describes the payer model only qualitatively; the log-normal parameters (μ = 5.6, σ = 1.1), usual-hour range, daily-rate gamma and known-payee range are not given.
   - The per-scenario generating processes are not specified.
   - The V2 channel probabilities are given only by example (message 55% vs 62%).

   Add an appendix table, or state that these are defined in the released generator code.
4. **Coercion model:**
   - Its type (a logistic model over 10 cue indicators), training/held-out split (307 held-out messages) and threshold (0.16) are not described.
   - The message corpus's availability is not stated.
5. **Baseline and model-family hyperparameters:** the configurations behind the numbers in §10.4 are not given:
   - RF: 300 trees, min_samples_leaf 2, balanced_subsample;
   - ET: 300 trees, min_samples_leaf 2, balanced;
   - LR: max_iter 2000, balanced, with standardisation;
   - all three with median imputation.
6. **SHAP and permutation settings:** 200 SHAP rows drawn with seed 42 and 5 permutation repeats are stated. **NO ISSUE FOUND.**
7. **Threshold rule:** stated precisely (highest validation recall with FPR ≤ 1%). **NO ISSUE FOUND.**
8. **Split procedure:** stated (test_size 0.25, then 0.20, stratified, random_state 42). **NO ISSUE FOUND.**
9. **Commands:** the manifests record the commands (e.g. `scripts/evaluate_model_v2.py --phase main`), but the paper does not name them. Add them to a reproducibility note.
10. **Test provenance:** "91 passed" needs the commit hash and date.

---

## 12. Claim-Strength Audit

| Location | Phrase | Justified? | Suggested wording |
|---|---|---|---|
| §1.2 | "A system that reasons only about the payer is therefore structurally limited for this class of fraud, however well its classifier is trained." | Partly (argument, uncited) | "…is likely to be limited for this class of fraud, since the most informative observations may concern the recipient and context." |
| §1.3 | "Contextual signals… are seldom integrated…" | Unsupported generalisation | "…are, in the works reviewed here, seldom integrated…" |
| §1.3 / §2.8 | "the reviewed approaches do not generally describe…" | Acceptable, because it is scoped to "reviewed" | Keep. |
| §2.8 | "Existing UPI-specific work has concentrated on transaction-feature classifiers evaluated by accuracy [8]." | One citation for "existing work" | "The UPI-specific study reviewed here [8] uses a transaction-feature classifier evaluated by accuracy." |
| §2.3 | "These approaches demonstrate that relational structure carries information…" | Acceptable with [5]–[10] | "These studies show…" |
| §1.5 C2 | "…prevents unavailable evidence from acting as either reassurance or suspicion…" | Overstated (CR-4, CR-5) | "…is designed so that, at the aggregation level, an unavailable family contributes neither reassurance nor suspicion…" |
| §1.5 C3 | "…using only information available before the payment being assessed." | Live path only (CR-8) | Add "in the live assessment path". |
| §1.5 C4 | "a reproducible evaluation of the approach" | Evaluates the classifier component | "a reproducible evaluation of the classifier component…, together with software validation of the graph and aggregation logic" |
| §8.2 | "Unavailable evidence never contributes to risk in either direction." | Overstated (CR-4, CR-5) | Scope to "an unavailable family contributes no score". |
| §10.5 | "…which is why the architecture does not treat absence as a risk factor." | Partly contradicted (CR-4, CR-5) | "…which is why the aggregation does not treat an unavailable family as a risk factor; an unobserved payee receives only a low-weight finding below the WARN band." |
| §12.11 | "The simulator makes the study fully reproducible." | Not yet (CR-9) | "…reproducible from the tagged code, seed and hashed artefacts." |
| §13 | "These properties hold by construction and are checked by tests; they do not depend on the simulator" | Mostly true, but "availability" is affected by CR-4 | Qualify the availability property as in CR-4 and CR-5. |
| §9.1 | "this does not expose labels" (payer overlap across splits) | Mild overclaim | "no payer identifier is used as a feature; generalisation to unseen payers is not measured." |
| "real-time" (§1.1) | Describes UPI itself | A | Keep. |
| "guarantee" (§3.1, §8.6, §11.6) | Always used in negation ("not a guarantee") | A | Keep. |
| "production", "prove", "state-of-the-art", "superior", "novel", "first", "robust" (as claims) | None found as a claim about this system | — | NO ISSUE FOUND. |

### Originality risk (no plagiarism detector was run; phrases are flagged by closeness to known source wording)

| Phrase (short) | Source | Risk | Suggested rewrite |
|---|---|---|---|
| "one curve dominates another in ROC space if and only if it does so in precision–recall space" (§2.7) | Davis & Goadrich [12] abstract | Close paraphrase of the main theorem statement | "…showed that dominance between two classifiers' curves is preserved when moving between ROC and precision–recall space, yet a higher ROC area need not imply a higher precision–recall area [12]." |
| "prevents long-standing legitimate communities from gradually becoming denser than fraudulent ones" (§2.4) | Spade+ [5] abstract ("…might inadvertently lead to densely connected legitimate communities") | Moderate | "…so that the graph does not accumulate stale edges that would make established, honest groups appear anomalously dense [5]." |
| "combines a local view of entity interactions with a global view of transaction relations" (§2.4) | PGLTSE [7] ("local interaction and global association information") | Low–moderate | "…learns from both the immediate interactions of each entity and the wider relation structure among transactions [7]." |
| General | Your earlier project report (`UPI_Project_Report_SRKREC*.pdf`, `UPI_Research_Paper_IEEE.docx`) | Self-overlap may flag in a similarity check if those documents were submitted to a repository | Run an institutional similarity check (Turnitin/iThenticate) before submission; cite the report if it was published. |

---

## 13. Security/Deployment Audit

All four were re-verified against the current working tree on 22 Sep 2026. **None has been fixed**, so §12.10 remains accurate.

| Finding | Current code | Status |
|---|---|---|
| `/fraud-graph` returns every user's payer-to-payee edges to any authenticated caller | `main.py:731` — `Depends(get_current_user)`, returns `load_edges()` for all edges with no per-user filter | A |
| `/health` is unauthenticated and exposes library versions | `main.py:499` — no auth dependency; `trained_with` = `{python 3.10.12, numpy 2.2.6, scikit-learn 1.7.2, scipy 1.15.3, joblib 1.6.0}` | A |
| `/chat` returns provider error text via `detail` | `chat.py` — `ChatReply.as_dict()` includes `"detail"`, which carries `provider_said` for 401, 403 and 429 | A |
| `/payee/report` counts distinct reporters but has no rate limit | `main.py:1205`; `report_payee` counts `DISTINCT reporter`; no rate limiter (the only limiter is in `chat.py`) | A |

**Additional note (not a vulnerability, a claim-accuracy issue):** §4.8 and §12.7 should mention the optional, unconfigured provider-verification interface in `upi_verify.py` (see §6 of this audit).

---

## 14. Required Corrections Before Submission

### MUST FIX

1. Add a complete IEEE bibliography [1]–[22] with the verified metadata. Remove working notes and checklists from the body (CR-1).
2. Rewrite §1.6 to the final 13-section structure (CR-2).
3. Correct §4.7 and align §2.7: offline SHAP uses the uncalibrated booster, while the live `/explain` screen uses model-agnostic SHAP on the calibrated V1; neither feeds the verdict (CR-3).
4. Correct the missing-evidence statements (§3.3 C3, §3.6, §8.2, §8.4, §10.5, §11.4, §13, Table 2):
   - an unseen payee has payee history *available* with a weight-6 finding;
   - MINIMAL cannot occur for a valid request;
   - classifier-internal missingness is a learned input (CR-4, CR-5).
5. Revise the Abstract so that:
   - results are identified as the ML component;
   - simulated graph and context channels are disclosed;
   - there is no end-to-end ten-family benchmark;
   - precision matches the body (CR-6).
6. Fix the §2.1 contradiction with the balanced class weighting (CR-7).
7. Qualify the temporal claims in §2.2, §2.4 and Contribution 3 to the live path (CR-8).
8. Commit and tag the evaluated code, including `clock.py`; resolve the `evaluate_model_v2.py` manifest mismatch; add a code-availability statement (CR-9).
9. Render Figures 1–2 and Tables 1–4 (currently specifications only).
10. Fix the cross-reference "(Section 5.10)" → "(Section 5.11)" in §4.3.

### SHOULD FIX

1. Renumber the tables in order of first citation.
2. Number the equations in §5.
3. Define PR-AUC, ROC-AUC, SHAP, NPCI and PSP at first use.
4. Add selected result figures (PR curves, scenario recall, calibration, importance) with V1/V2 and calibration labels.
5. Add a footnote to Table 7 (row C ≠ V1). Put the model-family comparison in its own table with an "uncalibrated" caption.
6. Mention the optional, unconfigured `upi_verify` interface in §4.8 and §12.7.
7. Rename `temporal_gnn.py`, or note in the code-availability statement that it is a histogram utility.
8. Add reproducibility details: library versions, simulator parameters (appendix), coercion-model description, and RF/ET/LR hyperparameters.
9. Soften the generalisations flagged in §12 (§1.2, §1.3, §2.8, C2–C4, §9.1, §12.11).
10. Rewrite the three close paraphrases flagged in §12 (Davis & Goadrich, Spade+, PGLTSE).
11. Verify the pages for [2], [9], [13], [14], [18], [20] and [21], the venue for [10], and the full PGLTSE dataset list and Secure UPI dataset statement.
12. Use one term, "coverage", defined once as evidence confidence.
13. Correct the §4.5 V1 summary (it omits `amount`) and the claim about group availability indicators.
14. Change Table 4's "Section on results" to "§10, Table 5".

### OPTIONAL

1. Consider "Incremental Payer–Payee Graph" in the title.
2. Unify numeric precision (4 d.p. in tables, 1 d.p. for prose percentages).
3. Use Roman-numeral section style and "Index Terms" when typesetting.
4. Consider α_i instead of a_i for availability if reviewers find the notation confusing.
5. Add a missing-evidence bar chart captioned "simulated class distributions".
6. Reduce the use of code identifiers in prose (§4.3, §8.3).

---

## 15. Final Submission Checklist

- [ ] Section numbering — §1.6 still uses the old numbering (CR-2)
- [x] Numerical consistency — every value matches the artifacts. The brief's V1 TP/FN values are swapped; the manuscript is correct.
- [x] V1/V2 distinction
- [x] Calibrated/uncalibrated distinction (0.4860 / 0.7326 calibrated; 0.4996 / 0.7234 uncalibrated)
- [ ] Graph evaluation limitation — present in §6–§13, **missing from the Abstract** (CR-6)
- [x] Message-template limitation
- [ ] Synthetic-data disclosure — present throughout, but the Abstract should also disclose the simulated graph and context channels (CR-6)
- [ ] Missing-evidence disclosure — contradicted by the implementation in §3.3, §3.6, §8.2 and §10.5 (CR-4, CR-5)
- [ ] SHAP wording — §4.7 is wrong (CR-3); §7.7 and §10.6 are correct
- [x] Security disclosure — all four findings re-verified as still present
- [ ] Citation correctness — no bibliography yet (CR-1); 8 entries still need page or venue verification
- [x] DOI correctness — all DOIs checked resolve, including Rastogi 10.13189/ujaf.2021.090326 and Secure UPI 10.1109/ICDT61202.2024.10489682
- [ ] IEEE formatting — equation numbering, captions, Roman numerals, abbreviations
- [ ] Figure/table consistency — figures and Tables 1–4 not yet rendered; table order
- [ ] Reproducibility — uncommitted code; missing environment and simulator details (CR-9)
- [ ] No unsupported claims — see §12 of this audit
- [x] No production claims
- [x] No GNN claim for this system (rename `temporal_gnn.py` before releasing code)
- [x] Conclusion contains no new results — every figure in §13 appears earlier in §10

---

**Sources used for reference verification:**

- [Crossref: Rastogi et al., doi:10.13189/ujaf.2021.090326](https://api.crossref.org/works/10.13189/ujaf.2021.090326)
- [Crossref: Spade+](https://api.crossref.org/works/10.1109/TKDE.2024.3394155)
- [Crossref: PGLTSE](https://api.crossref.org/works/10.1109/TBDATA.2024.3499338)
- [Crossref: Braithwaite](https://api.crossref.org/works/10.1093/jfr/fjae006)
- [Crossref: Dal Pozzolo et al.](https://api.crossref.org/works/10.1109/TNNLS.2017.2736643)
- [Crossref: FRAUDAR](https://api.crossref.org/works/10.1145/2939672.2939747)
- [Crossref: Davis & Goadrich](https://api.crossref.org/works/10.1145/1143844.1143874)
- [Crossref: Niculescu-Mizil & Caruana](https://api.crossref.org/works/10.1145/1102351.1102430)
- [Crossref: Saito & Rehmsmeier](https://api.crossref.org/works/10.1371/journal.pone.0118432)
- [Crossref: SMOTE](https://api.crossref.org/works/10.1613/jair.953)
- [Crossref: Brier](https://api.crossref.org/works/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)
- [Crossref: Breiman](https://api.crossref.org/works/10.1023/A:1010933404324)
- [Crossref: Friedman](https://api.crossref.org/works/10.1214/aos/1013203451)
- [Crossref: Bolton & Hand](https://api.crossref.org/works/10.1214/ss/1042727940)
- [PaySim, EMSS 2016 PDF](https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf)
- [IEEE Xplore: Secure UPI](https://ieeexplore.ieee.org/document/10489682/)
- [ACM DL: CARE-GNN](https://dl.acm.org/doi/10.1145/3340531.3411903)
- [ACM DL: Zadrozny & Elkan](https://dl.acm.org/doi/10.1145/775047.775151)
