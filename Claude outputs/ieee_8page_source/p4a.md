
# IV. Results and Discussion

All results are synthetic. Sections IV-A to IV-C concern the classifier on the row-level corpus; IV-D concerns the complete ten-family system and IV-E the graph, both on the replay stream.

## A. Original Baseline: V1 and V2 at Seed 42

On the original test split (30,000 payments, 355 fraudulent), V1 reaches a PR-AUC of 0.4860 and a ROC-AUC of 0.9652 and, at its validation threshold (0.1174), a recall of 49.3% at an FPR of 0.945% (precision 0.3846, F1 0.4321; TP = 175, FN = 180, FP = 280). The frozen V2 model reaches a PR-AUC of 0.7326, a ROC-AUC of 0.9900 and, at 0.1123, a recall of 0.7803 at an FPR of 0.00995 (precision 0.4843, F1 0.5976; TP = 277, FN = 78, FP = 295), with a Brier score of 0.00555 (BSS 0.525). Social-engineering recall is 19.23% for V1 and 65.93% for V2. Refitting both models with the experiment code reproduces these values exactly, which ties the new experiments to the original artefacts.

## B. Ten-Seed Stability

Table III summarises the ten independently generated corpora. The V1–V2 ordering holds on every seed: V2 has the higher PR-AUC, recall, precision, F1 and BSS and the lower Brier score in 10 of 10 paired comparisons, and the paired PR-AUC difference is +0.235 ± 0.027 (range +0.189 to +0.271). The seed-to-seed standard deviation of PR-AUC (≈0.02) is the scale against which any single-split difference should be read. The original V2 value (0.7326) lies above the V2 mean of 0.720 but inside its range (0.683–0.769); the original V1 value (0.4860) is close to its mean of 0.485. Both models operate at the same realised FPR (0.99 ± 0.08%), so the recall gain (+26.2 ± 3.2 pp) is not bought with extra false alarms. Because thresholds are chosen on validation, the test FPR exceeds the 1% budget on five V1 and three V2 seeds, by at most 0.17 pp.

The gain is scenario-specific. Social-engineering recall rises from 20.5 ± 2.7% to 62.4 ± 4.5% and mule-collection recall from 61.2 ± 5.5% to 83.0 ± 3.7% on every seed, whereas account-takeover (≈93%) and card-testing (≈97%) recall do not differ systematically (V2 higher on only 5 and 2 seeds). The additional channels matter where the payer behaves normally.

::: {custom-style="Marker"}
[[WIDE-START]]
:::

Table: TABLE III. Ten-Seed Stability of V1 and V2 (Row-Level Corpus, Seeds 42–51, Held-Out Test Splits of 30,000 Payments)

[[TAB3]]

::: {custom-style="TableNote"}
Statistics are over the ten seed-level values (sample sd); test sets are not pooled. Δ is the paired per-seed difference. SE, MC, AT, CT: social engineering, mule collection, account takeover, card testing.
:::

Table: TABLE IV. Feature-Family Ablation (Mean Over Seeds 42–51; Recall, FPR, Precision, F1 and Scenario Recall in %)

[[TAB4]]

::: {custom-style="TableNote"}
*F*: number of features. ΔPR-AUC is paired by seed against the V2 core (B). † Template-disjoint control: test messages come from templates never seen in training and the coercion model is refitted on the training half.
:::

::: {custom-style="Marker"}
[[WIDE-END]]
:::

## C. Feature-Family Ablation

Table IV and Fig. 4 report the ablation. The 12-feature V2 core (B) matches V1 (A) in PR-AUC (0.483 vs. 0.485); its lower ROC-AUC reflects that payer-relative payee novelty belongs to the relationship group. Every single-family addition to the core raises PR-AUC on all ten seeds. The graph group is associated with the largest single gain (+11.2 ± 1.5 pp) and with the largest increase in mule-collection recall (63.5% → 77.1%). The message group adds +7.5 ± 1.9 pp and is associated with the largest single increase in social-engineering recall (22.8% → 40.5%) but leaves mule-collection recall unchanged. Stated intent (+4.0 pp), relationship (+2.8 pp) and link (+1.7 pp) contribute less.

The cumulative sequence shows complementarity rather than redundancy between the graph and message groups: relationship and graph together (H) reach +14.0 pp, message adds a further +6.1 ± 2.0 pp (I) with social-engineering recall rising from 39.4% to 56.8%, link adds nothing measurable once messages are present (+0.1 ± 0.6 pp; J), and intent adds +3.5 ± 1.4 pp (K). Account-takeover and card-testing recall stay within 91–98% in every configuration. These are associations under the simulator: the ablation measures how much of each designed channel's signal the learner recovers, not how informative the channel would be on real payments.

*Template-disjoint control.* With the standard protocol, 99.9–100% of test messages reuse a template that also occurs in training rows. When test messages are restricted to held-out templates and the coercion model is refitted on the training half (E†, I†, K† in Table IV), the full-V2 PR-AUC changes by −0.4 ± 0.7 pp (0.715 ± 0.022 vs. 0.720 ± 0.023) and social-engineering recall by −0.1 ± 1.5 pp; the core + message configuration changes by +0.0 ± 0.9 pp. Exact-template memorisation therefore does not explain the message-channel gain. Both halves of the pool were, however, written by the same authors, so generalisation to real scam language remains untested.

![Fig. 4. Classifier robustness on the row-level corpus. (a) Per-seed test PR-AUC of V1 and V2 for seeds 42–51 (lines pair seeds; bars show means). (b) Paired PR-AUC gain over the V2 core for each ablation configuration (mean ± sd over seeds); † marks the template-disjoint controls.](fig/fig4_robustness.png){width=3.35in}
