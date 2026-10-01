
## D. End-to-End Ten-Family Evaluation

Values are mean ± sd over the three replay streams, for the last 25% of each stream unless stated otherwise. For all [[F:rows_total]] assessed payments, the recorded risk, coverage, evidence level, corroboration bonus and verdict were reproduced exactly from the recorded per-family evidence ([[F:consistency_mismatches]] mismatches), so the results describe the aggregation of Section II-F as executed.

*Risk and verdicts.* The ten-family risk $R$ reaches a PR-AUC of [[F:risk_pr]] and a ROC-AUC of [[F:risk_roc]] at a prevalence of about [[F:prev]]% (Table V); on the same payments the live V1 probability alone reaches [[F:lv_pr]] and [[F:lv_roc]]. Aggregation thus ranks fraud better than the classifier it contains. Its verdicts, however, differ sharply from the classifier's 1% budget (Fig. 5): legitimate payments receive APPROVE in [[F:legit_approve]]%, WARN in [[F:legit_warn]]%, STEP_UP in [[F:legit_stepup]]% and BLOCK in [[F:legit_block]]% of cases. Social-engineering and mule-collection payments are blocked in [[F:se_block]]% and [[F:mc_block]]% of cases, account-takeover payments in [[F:at_block]]%, and card-testing probes — small payments to shared addresses — are approved in [[F:ct_approve]]%. As operating points, WARN or above recalls [[F:warn_rec]]% of fraud at an FPR of [[F:warn_fpr]]%, and BLOCK recalls [[F:block_rec]]% at [[F:block_fpr]]%. No threshold on $R$ meets a 1% FPR budget on the window preceding the evaluation slice, because [[F:r99]]% of legitimate payments there already reach the cap $R=99$.

*Why legitimate payments are blocked.* The leading family of a blocked legitimate payment is stated intent in [[F:lead_intent]]% of cases, message pressure in [[F:lead_msg]]% and payee history in [[F:lead_payee]]%. The dominant finding is a refund purpose, which the stated-intent rule treats as decisive although the simulator, like everyday life, also generates honest refunds; bank messages written with pressure vocabulary act similarly on the message family. Early in each stream a third cause dominates: payee age is measured from the first observation in the store, so popular merchants initially look young and busy and trigger the collection-account rule. The share of legitimate payments blocked falls from [[F:block_legit_first]]% in the first 30 days to [[F:block_legit_last]]% in the last 30, while fraud BLOCK recall per block stays within [[F:block_fraud_min]]–[[F:block_fraud_max]]% (Fig. 5c).

*Families, corroboration and coverage.* Payer behaviour is active for nearly all fraud and most often leads a fraudulent assessment; removing it lowers the verdict of [[F:pb_loo_fraud]]% of fraudulent and [[F:pb_loo_legit]]% of legitimate payments (Table VI). Payee history is active for [[F:ph_act_fraud]]% of fraud but [[F:ph_act_legit]]% of legitimate payments. Address and QR evidence fires almost only on fraud, because the replay gives some collectors brand-lure handles and mismatched QR names. The relationship family never scores by design, and the positional graph finding fired on only [[F:gp_fraud]] of [[F:fraud_total]] fraudulent and [[F:gp_legit]] legitimate payments. Over the full streams a corroboration bonus is paid for [[F:bonus_fraud]]% of fraudulent and [[F:bonus_legit]]% of legitimate payments; counting distinct facts rather than families changed the bonus for only [[F:dedup_rows]] payments, so double counting is rare under this design rather than frequently corrected. No unavailable family ever carried a score, and coverage is high for both classes (median [[F:cov_fraud]] and [[F:cov_legit]]) and only weakly related to risk (Spearman [[F:cov_rho]]).

::: {custom-style="Marker"}
[[WIDE-START]]
:::

![Fig. 5. End-to-end ten-family evaluation on the chronological replay. (a) Risk distribution by label in the last 25% of stream 42, with the WARN, STEP_UP and BLOCK boundaries. (b) Advisory verdict mix by group, stream 42. (c) Share of legitimate and fraudulent payments flagged per 30-day block as the store matures (stream 42 bold, streams 43–44 thin).](fig/fig5_e2e.png){width=6.9in}

Table: TABLE V. Replay Streams: Complete System vs. Classifier Variants (Last 25% of Each Stream; %, Mean ± sd Over Streams)

[[TAB6]]

::: {custom-style="TableNote"}
System rows: the ten-family risk $R$ at the fixed verdict bands (PR-AUC and ROC-AUC of $R$ itself). Classifier rows A–D: fitted on the first 60% of each stream, threshold chosen on the next 15% (FPR ≤ 1%). *B uses the simulator's scenario-conditioned graph channel and is a contrast, not a deployable model. SE, MC, AT, CT: scenario recall.
:::


Table: TABLE VI. Evidence Families in the Complete System (Replay, Last 25% of Each Stream; %, Mean ± sd Over Streams)

[[TAB5]]

::: {custom-style="TableNote"}
Active: family scored above zero. AUC: family score vs. label among payments where the family was available. Leads: family has the highest score in a fraudulent assessment. LOO ↓: share of payments whose verdict would be lower if the family were removed and (5)–(8) re-applied.
:::

::: {custom-style="Marker"}
[[WIDE-END]]
:::


## E. Chronological Graph Replay

*Temporal correctness.* All checks passed on every stream: tests 1–3 and 10 at [[F:reads_1]] reads, tests 4–6 at [[F:reads_4]] reads of observed payees and test 7 at [[F:reads_7]] reads of unobserved payees, with no violation. The independent rebuild reproduced the recorded assessment exactly for all [[F:t8]] sampled payments (test 8), and all [[F:t9a]] observations written before each midpoint were unchanged at the end (test 9). Re-assessing the same payments against the final store, which contains their future, changed at least one recorded quantity for [[F:t9b]] of them, so the tests have power.

*Predictive value.* V1 refitted on the replayed state reaches a PR-AUC of [[F:A_pr]] (Table V). V2 with replayed graph features (C, [[F:C_pr]]) does not improve on V2 without them (D, [[F:D_pr]]): the paired difference is [[F:CD]] pp and negative on every stream ([[F:CD_min]] to [[F:CD_max]] pp), whereas the simulator's graph channel (B, [[F:B_pr]]) exceeds C by [[F:BC]] pp. Two properties explain the gap. The raw replayed features are non-stationary — the median arrival span of legitimate payees is [[F:arr_train]] days in the training period but [[F:arr_test]] days in the test period — so splits learned early do not transfer. And the replayed graph departs from the simulator's assumptions: nearly all payees fall in one giant component (median size [[F:comp_legit]] for legitimate payees, against 18–120 in the simulator channel), and the median overlap ratio of mule accounts ([[F:mc_overlap]]) lies below the 0.6 gate of the positional finding. Mule collection is also represented by only [[F:rings]] rings per stream, none of which falls in the training period on [[F:mc_train_zero]] stream.

## F. Discussion

*System versus classifier.* The classifier results of Sections IV-A to IV-C do not describe the system. With fixed rule weights and verdict bands, the complete pipeline warns on about a third of legitimate payments in a mature store and blocks [[F:legit_block]]%, far above the 1% budget under which the classifier was tuned, because single contextual cues (a refund purpose, pressure vocabulary, a young busy payee) can decide a verdict alone. It blocks most social-engineering payments, the scenario that motivated it, but approves most card-testing probes. The rule weights and bands were never fitted to labelled data; these results indicate that they should be — for example by learning the family-to-verdict mapping on labelled replays, or by capping single-cue families below BLOCK as the classifier already is — and that cold-start payee evidence needs explicit handling.

*Graph evidence.* The ablation associates the simulated graph channel with the largest single gain, but graph features computed by the running graph on a replayed ledger lowered PR-AUC on every stream, and the positional finding almost never fired. The replay therefore supports the graph's correctness, not the predictive value the row-level simulator attributes to it; raw counts and spans that grow with the store need normalising by observation time, and whether real collection accounts show the assumed overlap patterns can be settled only on real data.

*Transfer.* The live V1 model recalls only [[F:lv_rec]]% of replay fraud at its stored threshold, whereas the same representation refitted on the replay reaches a PR-AUC of [[F:A_pr]]. A shift between two simulators written by the same authors already breaks a stored operating point, which cautions against expecting any synthetic threshold to hold on real traffic.

## G. Limitations and Validity

All data are synthetic; no UPI, PSP, bank or NPCI data were used, and no result measures real-world performance. The V2 channels, the replay design and the message pool were specified by the authors, so the evaluation cannot reveal where their assumptions are wrong; the template-disjoint control removes exact-template reuse but not the stylistic homogeneity of an author-written, English-only pool. Precision and PR-AUC depend on prevalence, and isotonic calibration fitted at about 1.2% need not hold at other prevalences or under drift {{dalpozzolo}}. The replay covers three 150-day streams with few mule rings. The leakage audit found no leakage (11 items PASS, none FAIL) and five warnings: template overlap (quantified above), payers shared across random splits in Experiments 1–2, the scenario-conditioned contrast variant B, in-sample templates for the live message family, and non-stationary replayed graph features. The prototype is not deployed or integrated with any provider, and latency was not benchmarked. The new experiments reduce three earlier limitations — single-seed results, the lack of an end-to-end evaluation and the lack of a temporal test of the graph — but not the dependence on synthetic data.
