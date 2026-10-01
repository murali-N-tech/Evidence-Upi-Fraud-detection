# Editorial Compression Report — UPI Pre-Payment Risk-Assessment Paper

## PART 1 — Editorial Summary

| Item | Original | Compressed |
|---|---|---|
| Pages (same A4 IEEE two-column template, 10-pt Times, unchanged margins, column gap and spacing) | 19 | 10 (page 10 holds references only, about 60% full) |
| Words in rendered PDF (body + tables + captions + references) | ≈13,860 | ≈8,190 |
| Body prose (excluding equations, tables, references) | ≈11,500 | ≈7,200 |
| Sections | 13 (plus 57 subsections) | 9 (plus 17 subsections) |
| Figures / tables | 9 / 9 | 5 / 6 |
| Numbered equations | 20 | 11 |
| References | 23 | 23 (same numbering, same entries) |

**Sections compressed most:**
- Related Work: nine subsections became four thematic paragraphs, a gap statement and a compact Table I.
- Architecture and mathematical formulation: two sections merged into Sections III and V.
- Data and features: the V1 feature bullet list and the V2 group list became prose.
- Methodology: the result-category list and the metric equations are now short paragraphs.
- Discussion and limitations: eight Discussion and ten Threats subsections merged into one section with three parts.

**Removed or merged:**
- Four figures dropped:
  - Fig. 2, the workflow (its content is in the Fig. 1 caption and the text).
  - Fig. 4, the aggregation pipeline (Fig. 1 already shows the aggregation boxes).
  - Fig. 5, the evaluation workflow.
  - Fig. 7, recall versus FPR and the threshold sweep (the ROC/PR figure and the text cover them).
- Tables merged:
  - Table III (dataset) with Table IV (V1/V2 representations).
  - Table VI (confusion matrix) into Table IV (results), together with the V1 confusion counts.
- Table IX (software tests) became one prose paragraph that keeps every module count and property.
- Removed as implementation detail:
  - endpoint names;
  - the explainability UI;
  - the `ML_CASTS_AGREEMENT_VOTE` constant;
  - the entity-status hierarchy (only the "no payee verified" fact is kept);
  - the payer-quantity equations;
  - the Brier formula as a display equation;
  - the paper-organisation paragraph (now two sentences).
- Condensed:
  - the repository-state disclosure;
  - the list of four security issues, now two examples plus the need to resolve them;
  - the future-work list, now one sentence naming the priority next steps.

**Deliberately preserved:**
- **Contribution framing:**
  - pre-payment and advisory scope;
  - ten evidence families with availability and typed facts;
  - risk kept separate from coverage;
  - fact-aware corroboration, so there is no double counting;
  - the incremental graph as explicit analytics, not a GNN.
- **Model versions:** the V1/V2 separation, with V1 served and V2 frozen and offline only.
- **Evaluation protocol:**
  - validation-only selection and a single test measurement;
  - imbalance handling without resampling;
  - isotonic calibration.
- **All primary results:**
  - V1 and V2 metrics, including both confusion matrices;
  - PR and ROC curves;
  - scenario recall and the V1 ablation (+7.69 pp);
  - calibration bins;
  - permutation and SHAP rankings;
  - the full V2 ablation and model-family table;
  - missing-history statistics;
  - software validation (91 cases).
- **All limitations:**
  - synthetic data;
  - simulated V2 channels and template overlap;
  - the graph not exercised predictively;
  - no end-to-end evaluation;
  - no time-indexed replay;
  - distribution shift;
  - the V1 hours-from-usual encoding;
  - the deployment and security caveats;
  - no latency benchmark.


---

## PART 2 — Final Manuscript (text)

The typeset IEEE two-column version is `UPI_Graph_PrePayment_Risk_IEEE_10page.docx`, with a PDF preview. The text below is identical; figures are shown as file references.

### Graph-Based Pre-Payment Fraud Risk Assessment for UPI Transactions with Scam Link and QR Analysis

Chinthada Murali Naga Raju, Chigurupati Vanya Kishore, Gudivada Yaswanth Naga Duga Prabhas, Bethapudi Vivek, and K. Kishore Raju

Department of Artificial Intelligence and Data Science, Sagi Rama Krishnam Raju Engineering College (A), Bhimavaram, Andhra Pradesh, India (affiliated to Jawaharlal Nehru Technological University, Kakinada)

*Abstract*—Payments on India's Unified Payments Interface (UPI) settle within seconds, so loss can be prevented only before a payment is submitted. This is hardest for authorised push payment (APP) fraud, in which a deceived account holder makes the payment personally and the payer's own behaviour may look ordinary. We present an advisory pre-payment risk-assessment architecture that evaluates ten evidence families: payer behaviour, observed payee history, the payer–payee relationship, an incrementally maintained payer–payee graph, payment-address and QR analysis, amount context, message pressure, link safety, stated purpose, and a calibrated gradient-boosting classifier. Every family declares whether it could observe anything and which typed facts its score rests on. The aggregation therefore reports risk separately from evidence coverage and credits corroboration only across distinct observations. Because labelled UPI fraud data are not public, the classifier is evaluated on a synthetic corpus of 120,000 payments from 2,000 simulated payers (fraud prevalence 1.183%). The 12-feature model served by the prototype (V1) reaches a held-out precision–recall area (PR-AUC) of 0.4860. An expanded 47-feature research representation (V2), frozen and evaluated offline, reaches a PR-AUC of 0.7326, a ROC-AUC of 0.9900, a recall of 0.7803 at a false-positive rate of 0.00995 and a Brier score of 0.00555; its social-engineering recall is 65.93% against 19.23% for V1. The additional V2 channels are produced by the same simulator used for evaluation, so these figures describe the classifier under controlled synthetic conditions, not real-world UPI performance, and the complete ten-family verdict system was not benchmarked end to end.

*Index Terms*—Authorised push payment fraud, evidence aggregation, explainable machine learning, fraud risk assessment, pre-payment assessment, probability calibration, transaction graph, Unified Payments Interface (UPI).


#### I. Introduction

The Unified Payments Interface (UPI) lets a payer reach a recipient through a virtual payment address (VPA), a mobile number or a QR code, and moves funds between bank accounts within seconds [1]. Near-instant, hard-to-reverse settlement changes where fraud control can act: once money has moved onward, detection supports investigation but rarely recovery. The useful decision is taken earlier, between the payer's intention to pay and the submission of the payment, and it concerns whether that payment should proceed, proceed with a warning, or be verified first.

Fraud detection has conventionally been framed as classification of completed transactions against the history of the originating account [2], and operational studies add that fraud is rare, labels arrive late, patterns drift, and analysts can review only a limited number of alerts [3]. These methods are effective when an intruder controls the account. They are less effective against APP fraud, in which the legitimate owner authenticates the payment after being deceived by an impersonator, a fake seller, a false refund or an urgent message; because the owner's own authentication makes such payments look legitimate, the problem has also drawn regulatory attention [4]. A deceived payer may pay at a usual hour, from a usual device, for an ordinary amount. The more informative signals then describe the recipient: how other payers have paid it, whether its address or QR payload is consistent, and what the accompanying message asks for.

Graph methods capture such relational structure [5], [6], and recent work emphasises that payment graphs evolve and should be maintained or modelled over time [5], [7]. These methods, however, are usually posed as detectors over a whole graph rather than as evidence about one payment that has not yet been made. The UPI-specific study we reviewed applies XGBoost with SMOTE oversampling to transaction features and reports accuracy [8]. In the literature summarised in Table I, contextual signals such as address integrity, QR consistency, coercive message content and link reputation are seldom combined with payer and payee evidence in a single assessment, and two practical questions are rarely addressed: how unavailable evidence is represented, and how correlated signals are prevented from counting as independent corroboration.

This paper addresses that gap with an advisory decision-support layer that assesses a payment request before submission. It is not a payment processor or payment service provider (PSP), does not hold or settle funds, and is not integrated with the National Payments Corporation of India (NPCI) or any bank. The contributions are organised around the pre-payment setting and the treatment of evidence rather than a new learning algorithm:

1. an integrated pre-payment architecture that combines payer behaviour with payee history, relational evidence from a payer–payee graph, and context from the address, QR payload, amount, message, links and stated purpose;
2. an evidence-aware aggregation in which each family declares availability and typed facts, so that risk is reported separately from coverage, unavailable families receive no rule-based score, and one observation reported by several families is not counted as several corroborations;
3. an incrementally maintained payer–payee graph that supplies explicit, interpretable relational quantities without rebuilding the graph per request;
4. an imbalance-aware, reproducible evaluation of the served 12-feature model (V1) and an offline 47-feature research representation (V2), with validation-only model selection and a single held-out test measurement, complemented by software tests of the graph and aggregation logic.

Section II reviews related work, Section III formulates the problem and outlines the system, Section IV describes the data and features, and Section V the model and aggregation. Sections VI and VII present the methodology and results, Section VIII discusses findings and limitations, and Section IX concludes.

#### II. Related Work

*Transaction-level and behavioural detection.* Statistical treatments established the core difficulties of fraud detection: extreme class imbalance, asymmetric error costs, and adversaries who adapt to controls [2]. Behavioural models profile an account's usual amounts, hours, velocity and counterparties and score deviations from that profile; evaluations on operational card data show that alert capacity, delayed feedback and concept drift must be modelled for results to be realistic [3]. Such features respond well to account takeover but carry little signal when a manipulated payer behaves normally.

*Graph-based and temporal methods.* Dense-subgraph methods such as FRAUDAR flag unusually dense groups while bounding the effect of camouflage edges [6]. Graph neural networks (GNNs) learn node representations from neighbourhoods; CARE-GNN learns which neighbours to trust under camouflage [9], graph convolutional networks have classified illicit flows in cryptocurrency graphs [10], and APGNN learns several behaviour prototypes per class to obtain finer-grained and more interpretable detection [11]. For evolving graphs, Spade+ maintains the peeling order of dense-subgraph detection under edge insertions and deletions in an industrial setting [5], and PGLTSE models transactions as a continuous-time dynamic graph with gang-aware risk propagation [7]. We share the motivation for incremental and temporally aware processing but reproduce none of these methods: our graph supplies a few explicit quantities to a per-payment assessment and is neither mined for dense subgraphs nor used for representation learning.

*UPI and APP fraud.* UPI research has largely examined adoption and financial inclusion [1], while APP fraud has been analysed mainly from legal and regulatory viewpoints [4]. Peer-reviewed detection work on UPI is limited; Rani et al. combine SMOTE, principal component analysis and XGBoost and compare algorithms by accuracy [8], a measure that stays high at low prevalence even for a classifier that misses most fraud.

*Evaluation, calibration and explainability.* Under severe imbalance, precision–recall analysis is more informative than receiver operating characteristic (ROC) analysis [12], and a larger ROC area does not imply a larger precision–recall area [13]; recall at a fixed false-positive budget and the precision of the resulting alerts are directly interpretable [3]. Boosted trees often produce poorly calibrated scores, which sigmoid or isotonic post-processing can correct [14], [15]; the Brier score summarises probabilistic accuracy [16]. Calibration matters here because probabilities are mapped to evidence scores through fixed breakpoints. Our learner builds on gradient boosting [17]. Where labelled payment data are unavailable, agent-based simulation such as PaySim is an accepted substitute [18]. Class imbalance is often handled by interpolation-based oversampling [19]; we instead keep the data distribution, weight classes during training and recalibrate, using histogram-based tree construction for speed [20]. Additive attribution methods such as SHAP explain individual predictions [21], standard implementations are available in scikit-learn [22], and permutation importance originates with random forests [23]. For a payer-facing tool, we treat checkable evidence findings as the primary explanation and model attributions as a secondary view.

*Research gap.* Table I contrasts the reviewed work qualitatively. Transaction-level models describe the payer well but weaken under persuasion [2], [3], [4]; graph, temporal and prototype methods capture relational structure [5], [6], [7], [9], [10], [11] but not as explicit, explainable evidence about a single pending payment; and the UPI study reports accuracy [8]. What remains open is a pre-payment assessment that combines complementary evidence families, keeps relational quantities current incrementally, represents missing evidence explicitly, avoids double counting, and is evaluated with imbalance-aware metrics [12], [13], [14], [15], [16].

Table: TABLE I. Qualitative Summary of Related Work

| Work | Graph use | Context evidence | Pre-payment | Evaluation |
|---|---|---|---|---|
| Dal Pozzolo et al. [3] | None | Not reported | No | Alert precision |
| FRAUDAR [6] | Dense subgraphs | No | No | Fraud blocks found |
| CARE-GNN [9] | GNN | No | No | AUC, recall |
| APGNN [11] | Prototype GNN | No | No | Detection, interpretability |
| Spade+ [5] | Incremental peeling | No | Real-time community detection | Detection, update cost |
| PGLTSE [7] | Temporal GNN | No | No | AUC, AUPRC, recall, F1 |
| Secure UPI [8] | None | No | Not reported | Accuracy |
| This work | Incremental, explicit | Address/QR, amount, message, link, purpose | Yes (advisory) | PR-AUC, recall at 1% FPR, calibration, per-scenario recall, ablation |

*Note:* Datasets and tasks differ, so no numerical comparison is implied.

#### III. Problem Formulation and System Overview

##### A. Payment Request and Available Information

A payer about to authorise a push payment issues a request

$$x=\{u,p,a,t,q,m,l\} \qquad (1)$$

where $u$ is the payer, $p$ the payee identifier (VPA, mobile number, or address decoded from a QR payload), $a$ the amount, $t$ the assessment time, $q$ the raw QR payload, $m$ the message that prompted the payment and $l$ its links. Any component other than $p$ and $t$ may be absent; for example, a QR code without an embedded amount leaves $a$ unknown. The system returns, before submission, a risk score $R\in\{0,\dots,99\}$, a coverage (evidence-confidence) score $C\in\{0,\dots,100\}$ and an advisory verdict $V\in\{$APPROVE, WARN, STEP_UP, BLOCK$\}$. APPROVE means only that risk is below the intervention threshold, not that the payment is safe.

Let $\tau(o)$ be the time of observation $o$ in the ledger $\mathcal{H}$. The assessment may use only

$$\mathcal{H}_{<t}=\{o\in\mathcal{H}:\tau(o)<t\} \qquad (2)$$

comprising the payer's confirmed payments, imported statement rows and payments confirmed by other users, together with the payer profile, the pair history of $(u,p)$, the payee history of $p$ and a payer–payee graph $\mathcal{G}_{<t}$ built from the same observations. Settlement outcomes, later complaints and the fraud label are unavailable at $t$ by construction.

##### B. Design Constraints

The design follows six constraints. (i) No post-decision information: the observation for $x$ is written only after the payer confirms the payment. (ii) Missing is not safe: an unreadable source cannot make an assessment more favourable. (iii) Missing is not fraud: an unavailable family adds no rule-based score, and a measured absence, such as an address never seen paid, is a low-weight finding that cannot reach WARN alone. (iv) No double counting: one observation reported by several components is not independent corroboration. (v) Every non-zero rule-based contribution to risk is a named finding, and risk and availability are reported separately. (vi) The assessment reads maintained state and never recomputes history or the graph per request. No payee is presented as verified by a bank, PSP or NPCI; an optional provider-verification interface exists but was not configured, so no payee in this study is externally verified.

To make (ii) and (iii) operational, each family $i$ reports an availability flag and a score,

$$a_i=1\Rightarrow z_i\in\{0,\dots,100\},\qquad a_i=0\Rightarrow z_i=\bot \qquad (3)$$

so that "looked and found nothing" ($a_i=1$, $z_i=0$) differs from "could not look" ($a_i=0$). The relationship status of $(u,p)$ is ESTABLISHED if the pair history is non-empty, NEW if the payer's history is readable but contains no payment to $p$, and UNKNOWN if no authenticated payer is available; NEW is thus a measured absence and UNKNOWN an absent measurement. Statement rows that show only a truncated VPA or a counterparty name are recorded as partial identity and never promoted to an established relationship.

##### C. Architecture

Fig. 1 shows the architecture. A client decodes a QR code or accepts a typed address and optionally collects the amount, stated purpose and message. A read-only assessment service evaluates the ten families and aggregates them; a persistent store holds the observation ledger, per-payee reputation aggregates, the incremental graph and payer profiles. Writes occur only on a separate observation path, when a payment is confirmed or a statement is imported, so the request being assessed cannot influence the state it is assessed against. Every recorded payment carries a provenance label and a deduplication key, all addresses pass through one normalisation function, and timestamps are normalised to Indian Standard Time (IST).


*[Figure: fig/fig1_arch.png]* Fig. 1. Architecture of the pre-payment assessment. Solid arrows denote the read-only assessment path; the dashed arrow is the observation path, which writes only after confirmation or statement import. The served classifier family uses V1; V2 is frozen and evaluated offline.


Each family returns a record $e_i=(a_i,z_i,\Gamma_i,\Phi_i)$ of availability, score, findings $\Gamma_i$ and facts $\Phi_i$. Facts come from a closed vocabulary of fourteen observations (five payer-side, four payee-side and five contextual, e.g., HOUR_ANOMALY, PAYEE_SHAPE, GRAPH_POSITION) and identify *what* was observed independently of *which* family observed it. Table II summarises the families. Payee history reads reputation aggregates across all observed payers (observation age, distinct and repeat payers, amount uniformity, reports); a never-observed address remains available and receives a 6-point "unseen payee" finding, well below the 22-point WARN threshold. Payer behaviour compares the request with the payer's profile. The relationship family is informational and always scores zero. Address and QR checks address form, confusable characters, whether a QR payload is a payment request, and whether its displayed name matches its address. The remaining contextual families examine the amount against the rail limit, coercive cues in a supplied message, URLs, and the stated purpose.


Table: TABLE II. Evidence Families, Availability, Contribution to Risk and Coverage Weight

| Family | Available when | Contribution to risk | Weight |
|---|---|---|---|
| Payee history | An address can be read | Scored; unseen payee: 6-point finding; blocked payee: hard block | 24 |
| Relationship | Authenticated payer present | Informational (score fixed at 0); fact PAYEE_FAMILIARITY | 18 |
| Payer behaviour | Payer profile exists | 0 if raw score below 25, else 12–52 | 16 |
| Classifier (V1 served) | Amount known, profile, at least 5 prior payments | 0–18 below threshold; 30–66 above; facts attributed counterfactually | 12 |
| Address and QR | Always | Scored; malformed address or non-payment payload: hard block | 11 |
| Graph | Payee present in graph | Position findings only (fact GRAPH_POSITION) | 8 |
| Amount context | Amount known | Scored; above rail limit: hard block | 5 |
| Message pressure | Message supplied | Scored; 0 if only weak cues | 3 |
| Link safety | Always (0 if no link) | Scored | 2 |
| Stated intent | Purpose stated | Scored | 1 |

*Note:* Weights $c_i$ value availability for the coverage score (10) only; they sum to 100 and never enter risk.


##### D. Incremental Payer–Payee Graph

Each recorded payment from $u$ to $p$ updates a directed edge carrying a payment count and first and last timestamps, updates a per-payee record (distinct payers, repeat payers, payment count, first and last arrival and activity), and merges the components of $u$ and $p$ in a union–find structure. These updates share the database transaction that records the payment, so the graph cannot drift from the ledger, and components only merge, matching an append-only ledger. The assessment reads shape quantities through one indexed lookup and overlap through a bounded two-hop query. With $P(p)$ the payers of $p$ in $\mathcal{G}_{<t}$, the overlap with a peer $p'$ is

$$\omega(p,p')=\frac{|P(p)\cap P(p')|}{\min(|P(p)|,|P(p')|)} \qquad (4)$$

For the peer $p^\star$ sharing the most payers, overlap is read only if both addresses have at least ten payers, and it is discounted when the peer's repeat ratio is at least 0.1. A *position* finding fires when $|P(p)\cap P(p^\star)|\ge4$, $\omega(p,p^\star)\ge0.6$ and the repeat ratio of $p$ is below 0.1 (Fig. 2): many payers shared with one other address, almost none returning, which is consistent with collection accounts drawing on a common victim pool rather than with merchants whose customers return. *Shape* findings (many one-time payers, compressed arrivals, a high rate of new payers) restate what payee history already scores, so they carry zero weight and share the PAYEE_SHAPE fact. Component size is recorded but never used as a finding. The component is explicit graph analytics: it learns no embeddings and captures only patterns specified in advance, in exchange for findings that can be named and computed within one request on a small, partially observed graph.

*[Figure: fig/fig3_graph.png]* Fig. 2. Illustrative graph update (not experimental data). (a) Payees X and Y share payer C, so the overlap ratio is 0.5. (b) After D→X the shared set is {C, D} and the ratio is 1.0. No finding fires, because each address needs at least ten payers.

#### IV. Dataset and Feature Representation

##### A. Synthetic Corpus

Labelled UPI fraud data held by banks, PSPs and NPCI are not public, and no such data were used. The classifier is trained and evaluated on a corpus from a purpose-written simulator in the tradition of agent-based payment simulators [18], which gives a known generating process and seed-level reproducibility but cannot establish performance on real UPI traffic. The corpus contains 120,000 transactions from 2,000 simulated payers (nominal fraud rate 0.012, seed 42), of which 1,419 are fraudulent, a prevalence of 1.183% (Table III). Payers have a log-normal typical amount (median about Rs 270), a usual hour, a daily rate and habitual payees; a 12% segment makes large routine payments so that large amounts are not almost exclusively fraudulent, and all amounts respect the UPI per-transaction limit.

Fraud is drawn from four scenarios in fixed proportions: social engineering (SE) 50%, mule collection (MC) 22%, account takeover (AT) 20% and card testing (CT) 8%. These proportions are simulator settings, not estimates of real fraud composition. Labels come from the generating process, and the scenarios deliberately overlap with legitimate behaviour: legitimate payers sometimes send large amounts to new payees, some AT payments go to known payees, and some SE payees are aged accounts resembling small merchants. Two separability artefacts present in earlier generator versions were removed. The corpus is row-level: each row samples the evidence state at the moment one payment is about to be made, rather than replaying a time-ordered ledger through the running system.

Table: TABLE III. Dataset and Feature Representations (Synthetic Corpus, Seed 42)

| Item | Value |
|---|---|
| Transactions / simulated payers | 120,000 / 2,000 |
| Fraudulent transactions | 1,419 (1.183%) |
| Train / validation / test rows | 72,000 / 18,000 / 30,000 |
| Fraudulent rows per split | 851 / 213 / 355 |
| Test fraud by scenario | SE 182, MC 80, AT 73, CT 20 |
| Rows with hidden payee history | 29,376 (24.48%) |
| V1 (served baseline) | 12 features: payer timing, amount, velocity, payee; threshold 0.1174 |
| V2 (research, offline) | 47 features in 9 groups: V1 groups + phone-address flag, relationship, graph, message, link, intent; threshold 0.1123 |

*Note:* Both representations use the same learner and calibration; missing values stay NaN with explicit availability flags. SE, MC, AT, CT: social engineering, mule collection, account takeover, card testing.

##### B. V1 Features and Missing Evidence

V1 has twelve features. Seven describe the payer and payment (amount; amount over the payer's median prior payment; a night indicator for hours before 06:00 or from 22:00; circular distance from the payer's usual hour; seconds since the last payment, capped at one day; payments in the last hour; payments today relative to the daily average). Five describe the payee: payer-relative novelty (payee_is_new), payee age, distinct payers, repeat ratio (payers who paid more than once over distinct payers, measurable only with at least five payers) and a history-availability flag.

A deployment observes a payee only after one of its users, or an imported statement, records a payment to it. An earlier pipeline substituted constants for missing payee values, in effect describing every unknown address as moderately established; the current pipeline substitutes nothing. Unavailable payee history leaves the three history values missing (NaN) with the availability flag set to 0, and the histogram-based learner routes missing values natively [20], [22]. The simulator reproduces unavailability with an observation mask whose visibility probability rises with the payee's distinct-payer count (half-saturation at nine payers). It hides payee history on 24.48% of rows, where simulated prevalence is 1.729% against 1.005% on visible rows. This difference follows from the masking rule, which ties visibility to payer count, and is not evidence that missing history indicates fraud. Because the availability flag is itself a classifier input, missingness can influence the learned probability; the pipeline reports this bias on every run and automated tests bound it. One V1 encoding limitation is retained unchanged: when the usual hour is unknown, the hour-distance feature is 0, indistinguishable from a payment at exactly the usual hour, and V2 inherits this encoding.

##### C. V2 Representation

V2 extends V1 to 47 features in nine groups: payer timing (2), amount (2), velocity (3), payee (5, adding a phone-number-address flag), relationship (4: novelty, prior payments to the payee, days since the first such payment, availability), graph (7: arrival span, lifespan, payments per day, component size, shared-payer peak, overlap ratio, availability), message (18: ten coercion-cue indicators, a coercion probability and score, a weak-cues flag, four text statistics, availability), link (3) and stated intent (3). The V2 generator reproduces the V1 columns bit-for-bit and draws every added column from a separate random stream; an automated check asserts that the V1 columns are unchanged.

The added channels are simulated. Whether a message is present, its kind, whether it carries a link and which purpose is stated are drawn from scenario-conditional probabilities chosen by the designers and overlapping by design (a message accompanies 55% of legitimate and 62% of SE rows). Message texts come from a pool of 1,022 templates written for the project (364 scam, 557 hard negatives, 101 easy negatives), and one message feature comes from a coercion model fitted on the same pool. Graph features are not computed by running the incremental graph over a simulated ledger; they are sampled per row from distributions that mirror the production gates (the group is missing whenever payee history is hidden, and overlap is missing below the ten-payer gate). The V2 experiment therefore measures the contribution of simulated graph-derived and contextual channels, while the graph implementation itself is validated by software tests. V2 gains are evidence about the architecture under the simulator's assumptions, not about production performance.

#### V. Model and Evidence Aggregation

##### A. Calibrated Classifier

Both representations use histogram-based gradient-boosted trees [17], [20] as implemented in scikit-learn [22] (300 iterations, learning rate 0.08, at most 31 leaves, balanced class weights, fixed random state), wrapped in five-fold isotonic calibration, which averages the calibrated outputs of five booster–calibrator pairs. Imbalance handling is kept explicit: the data keep their 1.183% prevalence and each split is stratified; balanced class weights rescale each class's loss contribution; no SMOTE [19], random over- or undersampling, or generative minority sampling is used; and isotonic calibration returns the weighted scores to the observed probability scale. The V2 configuration was fixed on validation only. Four model families were compared on the full V2 features (the V1-configured gradient boosting, a random forest, extremely randomised trees and balanced logistic regression, the last three with median imputation) by validation PR-AUC and then validation recall at 1% FPR, and a validation feature-group ablation selected the full nine-group set. The chosen model was fitted, calibrated, thresholded on validation, and frozen with a fingerprint of the generators, message corpus and coercion weights; its record states that no test rows were used in any decision.

The threshold $\theta$ is the value with the highest validation recall among thresholds whose validation false-positive rate (FPR) does not exceed 1%, applied unchanged to the test split: 0.1174 for V1 and 0.1123 for V2. An independent re-derivation of the V1 threshold agrees with the stored value to four decimal places. The classifier threshold is distinct from the verdict bands below.

##### B. Classifier Evidence Score

Measurements are mapped to scores by the clamped linear map

$$g(v;v_0,v_1,y_0,y_1)=y_0+(y_1-y_0)\,\mathrm{clip}\!\left(\frac{v-v_0}{v_1-v_0},0,1\right) \qquad (5)$$

With $\pi(x)$ the calibrated probability, the classifier family scores

$$z_{\mathrm{ml}}=\begin{cases}\mathrm{round}\,g(\pi;0,\theta,0,18), & \pi<\theta\\ \mathrm{round}\,g(\pi;\theta,0.8,30,66), & \pi\ge\theta\end{cases} \qquad (6)$$

Its ceiling of 66 lies below the BLOCK boundary of 70, so the classifier cannot block a payment on its own. Declaring every fact the classifier reads would let one probability supply up to six corroborations, so facts are attributed counterfactually. A probe $\phi^{(-v)}(x)$ neutralises the inputs behind fact $v$ (payee-history inputs set to missing, payer-side inputs set to the payer's ordinary values), and

$$v\in\Phi_{\mathrm{ml}}\iff\pi(x)-\pi\!\left(\phi^{(-v)}(x)\right)\ge\max\left(0.01,0.15\,\pi(x)\right) \qquad (7)$$

The probes never change $z_{\mathrm{ml}}$.

##### C. Risk, Corroboration, Coverage and Verdict

Let $z_{(1)}\ge\dots\ge z_{(n)}$ be the scores of active families ($a_i=1$, $z_i>0$). The base risk is a damped maximum,

$$R_0=\min\left(99,\ \mathrm{round}\left(z_{(1)}+\beta\sum_{j=2}^{n}z_{(j)}\right)\right),\quad\beta=0.4 \qquad (8)$$

with $R_0=0$ when no family is active. Let $\mathcal{A}=\{i:a_i=1,z_i\ge20\}$, $\Phi^\star=\bigcup_{i\in\mathcal{A}}\Phi_i$, and $\mathcal{A}^\star\subseteq\mathcal{A}$ the families reporting at least one fact. Then

$$k=\min(|\Phi^\star|,|\mathcal{A}^\star|),\qquad R=\min\left(99,R_0+B(k)\right) \qquad (9)$$

with $B(k)=0$ for $k<2$, $B=6,14,22,28$ for $k=2,\dots,5$ and 32 for $k\ge6$. Counting families prevents a family from corroborating itself; counting distinct facts prevents several families reading one observation from appearing to agree. Hence the payer-behaviour rules and the classifier, which both examine relative amount, hour, activity and novelty, and the payee-history family and graph shape findings, which both describe inflow patterns, do not multiply one observation's weight. Coverage depends only on availability,

$$C=\mathrm{round}\left(100\sum_i a_ic_i\Big/\sum_i c_i\right) \qquad (10)$$

with the weights $c_i$ of Table II. An evidence level is also reported: FULL when payer behaviour, payee history and relationship are all available, PARTIAL when payer behaviour or payee history is, and MINIMAL otherwise (which arises only when no address can be read and a hard block already applies). Finally, with hard-block set $\mathcal{B}$ (blocked payee, malformed address, non-payment payload, missing payee, empty request, amount above the rail limit) and $\Gamma=\bigcup_i\Gamma_i$,

$$V=\begin{cases}\text{BLOCK}, & \Gamma\cap\mathcal{B}\ne\emptyset\text{ or }R\ge70\\ \text{STEP\_UP}, & 45\le R\le69\\ \text{WARN}, & 22\le R\le44\\ \text{APPROVE}, & R\le21\end{cases} \qquad (11)$$

Risk and coverage combine freely: a reported address paid by a payer without history yields high risk with limited evidence, and a routine payment to a long-established payee yields low risk with strong evidence. Low coverage is never read as low risk. The verdict bands are fixed design parameters and advisory outputs, not ground-truth classes; no verdict holds or reverses a payment.

##### D. Explanations

Each assessment returns its findings (code, severity, plain-language message, reporting family), the facts that took part in corroboration with the families that reported each, and any evidence that could not be evaluated with the reason. These checkable findings are the primary, user-facing explanation. As a secondary view, the prototype computes model-agnostic SHAP values [21] for the V1 vector against its calibrated probability; the offline V2 analysis uses the tree explainer described in Section VI. Attributions describe the classifier; they are not the verdict and do not establish causality.

#### VI. Experimental Methodology

*Splits and result categories.* A stratified quarter of the corpus is held out for testing and 20% of the remainder for validation (random state 42), giving 72,000/18,000/30,000 disjoint rows; V1 and V2 use the same split. Because the corpus is row-level, the split is random rather than chronological, so one simulated payer's payments can appear in several splits; no payer identifier is a feature, but generalisation to unseen payers is not measured. Three result categories are kept apart. *Validation results* informed decisions (V2 model family and feature set; both thresholds). *Held-out test results* come from a single evaluation of the stored V1 artefact, without refitting, and of the frozen V2 artefact. *Software test results* verify properties of the running code and are not predictive metrics.

*Metrics.* With fraud as the positive class, the primary threshold-independent metric is PR-AUC, computed as average precision, whose no-skill value equals prevalence (about 0.012) [12], [13]. ROC-AUC is secondary. Recall, precision, F1 and FPR are reported at the validation-selected threshold, with recall at about 1% FPR given together with the realised test FPR. Calibration is assessed with the Brier score $\mathrm{BS}=\frac{1}{n}\sum_j(\pi_j-y_j)^2$ [16], the Brier skill score $\mathrm{BSS}=1-\mathrm{BS}/\mathrm{BS}_{\mathrm{ref}}$ with $\mathrm{BS}_{\mathrm{ref}}=0.01169$ for a constant base-rate predictor, and reliability curves with quantile and fixed-width bins. Accuracy is not a headline metric, because labelling every payment legitimate is already about 99% accurate. The ground truth is binary, so the confusion matrix is the 2×2 matrix of the classifier flag against the label; no four-class verdict matrix is constructed.

*Further analyses.* Recall is reported per scenario at the overall threshold. The V1 ablation trains payer-only (seven features), payee-only (five) and full models in the V1 configuration, each thresholded on validation, and reports the full model's SE recall minus that of the stronger half. The V2 ablation re-fits *uncalibrated* gradient boosting on six nested feature sets, two leave-one-group-out sets and a V1-feature reference, each with its own validation threshold and one test measurement; the model-family comparison uses the same protocol. V2 feature attribution uses permutation importance [23] (mean drop in test average precision over five permutations of the calibrated model) and tree-explainer SHAP on 200 test rows of the uncalibrated first-fold booster, whose magnitudes do not decompose the calibrated probability.

*Software validation and reproducibility.* The graph is not executed by the predictive experiments; its behaviour and that of the aggregation are verified by software tests on the running store (Section VII-F). Spade+ and PGLTSE were not re-implemented, so no numerical comparison with them is made. No latency benchmark exists, and no response-time figure is reported. The corpus is deterministic given the seed, with dataset, model and corpus hashes recorded in manifests; the environment was Python 3.10.12 with numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3 and joblib 1.6.0.

#### VII. Results

Unless stated otherwise, results are held-out test results on 30,000 payments (355 fraudulent: 182 SE, 80 MC, 73 AT, 20 CT; 29,645 legitimate), with every threshold chosen on validation under a 1% FPR budget. They describe the classifier component under controlled synthetic conditions, not the ten-family verdict system.

##### A. V1 Baseline and V2 Primary Result

Table IV compares the two calibrated models on identical rows. V1 reaches a PR-AUC of 0.4860 and a ROC-AUC of 0.9652; at $\theta=0.1174$ it recalls 49.3% of fraud at an FPR of 0.945%. The frozen V2 model reaches a PR-AUC of 0.7326 and a ROC-AUC of 0.9900; at $\theta=0.1123$ it recalls 78.03% at an FPR of 0.995%, with precision 0.4843, F1 0.5976 and a Brier score of 0.00555 (BSS 0.525). Recall at the 1% budget therefore rises from 49.3% to 78.03% while false positives rise from 280 to 295. Of V2's 572 alerts, 277 are fraudulent and 295 legitimate; 78 frauds (22.0%) are missed. The validation PR-AUC recorded when the V2 threshold was chosen (0.6995) is close to the test value, which is consistent with the protocol. All figures come from one split and one seed, without confidence intervals or significance tests.

Table: TABLE IV. Classifier Performance on the Held-Out Test Split (N = 30,000; 355 Fraudulent)

| Metric | V1 (served) | V2 (offline) |
|---|---|---|
| PR-AUC | 0.4860 | 0.7326 |
| ROC-AUC | 0.9652 | 0.9900 |
| Threshold (validation) | 0.1174 | 0.1123 |
| Recall | 0.4930 | 0.7803 |
| Precision | 0.3846 | 0.4843 |
| F1 | 0.4321 | 0.5976 |
| FPR | 0.00945 | 0.00995 |
| TP / FN | 175 / 180 | 277 / 78 |
| FP / TN | 280 / 29,365 | 295 / 29,350 |
| Alerts raised | 455 | 572 |
| Brier score | 0.00798 | 0.00555 |
| Brier skill score | 0.317 | 0.525 |

##### B. Ranking Behaviour

Fig. 3(a) shows the V2 precision–recall curve against the no-skill line at the test prevalence of 0.0118. Precision stays near 1.0 at low recall and falls as recall increases; the operating point sits at recall 0.780 and precision 0.484. The ROC curve in Fig. 3(b) rises almost vertically because the negative class is so large, which is why the precision–recall view is the more informative of the two [12], [13].


*[Figure: fig/fig6_pr_roc.png]* Fig. 3. Frozen V2 model on the held-out test split: (a) precision–recall curve with the no-skill baseline at test prevalence; (b) ROC curve. Markers show the validation-selected operating point.


##### C. Scenario-Wise Recall

Aggregate recall hides large differences between scenarios (Table V). V1 detects AT and CT, which alter the payer's own behaviour, at high rates, but only 19.23% of SE payments. In the V1 ablation, the payer-only model recalls 3.85% of SE payments, the payee-only model 11.54% and the full model 19.23%, 7.69 percentage points above the stronger half; this is a scenario-specific complementarity, not a gain in overall recall. V2 raises SE recall to 65.93% and MC recall to 85.00%, leaves AT and CT unchanged, and SE remains the hardest scenario for both models.

Table: TABLE V. Scenario-Wise Recall on the Held-Out Test Split (Detected Cases)

| Scenario | Cases | V1 | V2 |
|---|---|---|---|
| Social engineering | 182 | 19.23% (35) | 65.93% (120) |
| Mule collection | 80 | 63.75% (51) | 85.00% (68) |
| Account takeover | 73 | 94.52% (69) | 94.52% (69) |
| Card testing | 20 | 100.00% (20) | 100.00% (20) |
| Legitimate (FPR) | 29,645 | 0.945% (280) | 0.995% (295) |

##### D. Calibration

Fig. 4 shows V2 reliability on the test split. With quantile bins only four bins are populated, because most predictions lie near zero, and they lie close to the diagonal. Fixed-width bins expose the upper range: the lowest bin (below 0.1) holds 29,383 payments with a mean prediction of 0.0033 and an observed rate of 0.0025; intermediate bins show observed rates close to or somewhat above the mean prediction (e.g., 0.435 against 0.347 in the 0.3–0.4 bin of 46 payments); the top bin holds 71 payments, all fraudulent, with a mean prediction of 0.973. Intermediate bins contain only 33–231 payments, so their rates are imprecise. Both calibrated models improve on the base-rate predictor (BSS 0.525 for V2 and 0.317 for V1), on the simulator only.

*[Figure: fig/calibration_curve.png]* Fig. 4. Calibration of the frozen V2 model on the held-out test split: quantile bins (left) and fixed-width bins (right); Brier score 0.00555, skill score 0.525.

##### E. Feature Attribution and Ablation

By permutation importance (Fig. 5(a)), the most useful V2 inputs are payee_is_new (0.3656), arrival_span_days (0.2455), amount_over_user_avg (0.2059), msg_coercion_probability (0.1379) and component_size (0.1209). Individual message-cue indicators, link features and availability flags score near zero, some slightly negative. Mean absolute SHAP on the uncalibrated booster (Fig. 5(b)) again ranks payee_is_new first, followed by arrival_span_days, amount_over_user_avg and component_size. Both measures reflect predictive usefulness under this protocol; correlated features can share or trade importance, and neither implies causal influence.


*[Figure: fig/fig5_importance.png]* Fig. 5. V2 feature attribution on the held-out test split (ten highest-ranked features): (a) permutation importance; (b) mean absolute SHAP attribution on the uncalibrated first-fold booster (200 rows).

Table: TABLE VI. V2 Ablation and Model-Family Comparison (Held-Out Test; Uncalibrated, Separately Trained Models; Threshold from Validation)

| Configuration | Features | PR-AUC | Recall | FPR | SE recall | MC recall |
|---|---|---|---|---|---|---|
| A. Payer timing + amount | 4 | 0.0970 | 0.163 | 0.0093 | 1.6% | 2.5% |
| B. A + payee | 9 | 0.3156 | 0.377 | 0.0096 | 17.6% | 57.5% |
| C. B + velocity | 12 | 0.4562 | 0.468 | 0.0096 | 17.0% | 61.3% |
| D. C + graph | 19 | 0.5653 | 0.546 | 0.0076 | 28.0% | 70.0% |
| E. C + message, link, intent | 36 | 0.5923 | 0.625 | 0.0106 | 47.8% | 60.0% |
| F. Full V2, gradient boosting | 47 | 0.7234 | 0.766 | 0.0083 | 65.9% | 78.8% |
| Full V2 without graph | 40 | 0.6298 | 0.682 | 0.0104 | 57.1% | 60.0% |
| Full V2 without message, link, intent | 23 | 0.6101 | 0.620 | 0.0092 | 36.8% | 78.8% |
| V1 twelve features (reference) | 12 | 0.4996 | 0.496 | 0.0093 | 19.2% | 63.8% |
| Full V2, random forest | 47 | 0.6915 | 0.744 | 0.0104 | 60.4% | 77.5% |
| Full V2, extremely randomised trees | 47 | 0.5145 | 0.561 | 0.0099 | 49.5% | 38.8% |
| Full V2, logistic regression | 47 | 0.5055 | 0.566 | 0.0103 | 48.9% | 37.5% |

*Note:* Rows are comparable only with each other: the uncalibrated full-V2 row (0.7234) is not the frozen calibrated V2 result (0.7326), and the V1-feature reference (0.4996) is not the served V1 model (0.4860). Configuration C has twelve features but is not V1 (it includes the phone-address flag and omits payee_is_new). Single fits without significance tests.


Table VI reports the V2 ablation and model-family comparison, which should be read only for differences between its own rows. Adding the simulated graph group to configuration C raises PR-AUC from 0.4562 to 0.5653, and adding the message, link and intent groups raises it to 0.5923. Removing either group from the full set lowers PR-AUC, to 0.6298 without graph and 0.6101 without message, link and intent. Removing the message channels reduces SE recall most (to 36.8%), whereas removing the graph group reduces MC recall most (to 60.0%). On the full features, gradient boosting (0.7234) exceeds the random forest (0.6915), extremely randomised trees (0.5145) and logistic regression (0.5055).

##### F. Missing History and Software Validation

Payee history is hidden on 29,376 of 120,000 rows (24.48%), where simulated prevalence is 1.729% against 1.005% on visible rows, and even among hidden-history rows 98.3% of payments are legitimate. The availability flag, although a classifier input, has a V2 permutation importance of 0.0000. The artefacts do not break test performance down by availability.

Five software-test modules with 91 cases, all passing on the evaluated code, verify implementation properties rather than predictive accuracy. The graph-cache tests (21) cover incremental updates, component merging, transactional rollback and concurrent observations, and confirm that a check writes nothing, never rebuilds the graph, and issues a bounded number of queries as the store grows. The graph-semantics tests (18) confirm that merchant or household overlap is not treated as evidence, component size is never a finding, and an unknown payee is unavailable rather than zero. The evidence-availability tests (15) check that no population constants are substituted, that a measured zero differs from unavailability, and that availability does not act as a label proxy. The evidence-model tests (18) check that unavailable evidence has no score or vote, that the damped combination is monotonic and that each family votes once. The V2-dataset tests (19) confirm that V1 columns, labels and order are bit-identical, that generation is deterministic and that the ten-payer overlap gate is respected.

#### VIII. Discussion and Limitations

*Why V2 differs from V1.* The V1 results separate cleanly by scenario because seven of its twelve features describe the payer: AT and CT depart from the payer's habits and are detected, while in SE the payer behaves normally and V1's only payee view is age, payer count and repeat ratio (unavailable for about a quarter of payments) plus novelty. V2 adds relational, graph, message, link and purpose channels and, under the synthetic protocol, raises PR-AUC from 0.4860 to 0.7326, SE recall from 19.23% to 65.93% and MC recall from 63.75% to 85.00% at the cost of 15 additional false positives. Attribution and ablation agree: payee novelty is the most useful single input, simulated graph features (arrival span, component size) and the message-coercion probability also rank highly, SE gains come mainly from the message channels, and MC gains mainly from the graph channels. The direction of this result supports the premise that payee-side, relational and contextual evidence matters for payer-authorised fraud. Its magnitude does not transfer, because the designers set the scenario-conditional context and graph distributions and one message feature is fitted on the templates V2 draws from. The more robust finding concerns the learning setup: a calibrated classical learner can use a heterogeneous representation in which whole groups are often missing, with explicit availability indicators, without missingness dominating.

*Implications for pre-payment assessment.* Explicit missingness avoids the most harmful default, describing an unknown address as established, and also the opposite error of warning on every first payment to a stranger; a never-observed address receives a finding below WARN, and an unavailable family contributes nothing. Separating risk from coverage keeps a low score caused by missing evidence from resembling one caused by reassuring evidence. At the 1% FPR budget, V2 raises 572 alerts per 30,000 payments, slightly more than half of them legitimate. Class weighting and calibration do not change the dependence of precision on prevalence, so at a lower real prevalence the same FPR would yield a larger share of false alerts. In the full architecture the classifier is one bounded input, and how aggregation changes the alert burden was not measured.

*Limitations and threats to validity.* (i) *Synthetic data.* No real UPI transactions or fraud labels were used; the populations, scenarios, overlaps, prevalence and scenario mixture are simulator settings, and an evaluation drawn from the same simulator cannot reveal where its assumptions are wrong. (ii) *Simulated V2 channels and template overlap.* The graph, message, link, intent and relationship channels are simulator-generated, and the message pool of 1,022 templates was written for the project, so generalisation to real scam language is untested. (iii) *Graph not exercised predictively.* The incremental graph was not run in the predictive experiment, and its positional finding was not evaluated on labelled data; only software tests validate it. (iv) *No end-to-end evaluation.* The predictive results cover one family; the rule-based families, aggregation, corroboration and verdict bands were checked for correctness but not for detection performance, so the verdict distribution for fraudulent and legitimate payments is not established. (v) *Temporal replay.* The live path enforces the pre-payment boundary, but aggregates and the graph are not time-indexed, so replaying history against a store that already contains later payments would not reproduce the pre-payment state; the offline boundary holds only by construction. (vi) *Distribution shift and calibration.* Real behaviour, scam language, VPA patterns, graph structure and prevalence may differ from the simulator, fraud adapts over time [3], the isotonic calibrators rest on roughly 170 fraudulent examples per training fold, and all results come from one seed and split. (vii) *Deployment.* The prototype is not deployed, has no NPCI, bank or PSP integration, performs no external verification and has no latency benchmark. It also has unresolved access-control issues (for example, a graph endpoint that exposes other users' payment relationships to any authenticated caller, and payee reports that are not rate-limited), and an advisory layer can itself be probed or manipulated. The evaluated repository state must be committed and tagged before the code accompanies this paper.

#### IX. Conclusion

This study asked whether a pre-payment assessment for UPI-style payments can combine payer behaviour, observed payee history, relational structure and payment context while remaining explicit about what it does not know. The proposed architecture does so through ten evidence families that declare availability and typed facts, an incrementally maintained payer–payee graph that supplies explicit relational quantities, a damped-maximum aggregation with fact-aware corroboration, and separate reporting of risk and coverage; these properties hold by construction and are checked by software tests. On a synthetic corpus, the served V1 model reaches a held-out PR-AUC of 0.4860 but detects only 19.23% of social-engineering payments, whereas the offline V2 representation reaches a PR-AUC of 0.7326 and a recall of 0.7803 at an FPR of 0.00995 with well-calibrated probabilities on the simulator. Because the added channels are simulator-generated, this supports the direction of the premise but not the size of any real-world gain. The next step is evaluation on independent real transaction data and externally validated, multilingual scam-message corpora, together with a chronological end-to-end benchmark of the incremental graph and the complete verdict system using time-indexed aggregates.

#### References

[1] S. Rastogi, C. Panse, A. Sharma, and V. M. Bhimavarapu, "Unified Payment Interface (UPI): A digital innovation and its impact on financial inclusion and economic development," *Universal J. Accounting Finance*, vol. 9, no. 3, pp. 518–530, 2021, doi: 10.13189/ujaf.2021.090326.

[2] R. J. Bolton and D. J. Hand, "Statistical fraud detection: A review," *Statist. Sci.*, vol. 17, no. 3, pp. 235–255, 2002, doi: 10.1214/ss/1042727940.

[3] A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, "Credit card fraud detection: A realistic modeling and a novel learning strategy," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 29, no. 8, pp. 3784–3797, Aug. 2018, doi: 10.1109/TNNLS.2017.2736643.

[4] J. Braithwaite, "'Authorized push payment' bank fraud: What does an effective regulatory response look like?," *J. Financial Regulation*, vol. 10, no. 2, pp. 174–193, 2024, doi: 10.1093/jfr/fjae006.

[5] J. Jiang, Y. Chen, B. He, M. Chen, and J. Chen, "Spade+: A generic real-time fraud detection framework on dynamic graphs," *IEEE Trans. Knowl. Data Eng.*, vol. 36, no. 11, pp. 7058–7073, Nov. 2024, doi: 10.1109/TKDE.2024.3394155.

[6] B. Hooi, H. A. Song, A. Beutel, N. Shah, K. Shin, and C. Faloutsos, "FRAUDAR: Bounding graph fraud in the face of camouflage," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, pp. 895–904, doi: 10.1145/2939672.2939747.

[7] J. Ma, S. Xiang, Q. Li, L. Yuan, D. Cheng, and C. Jiang, "Parallel graph learning with temporal stamp encoding for fraudulent transactions detections," *IEEE Trans. Big Data*, vol. 11, no. 4, pp. 1945–1958, 2025, doi: 10.1109/TBDATA.2024.3499338.

[8] R. Rani, A. Alam, and A. Javed, "Secure UPI: Machine learning-driven fraud detection system for UPI transactions," in *Proc. 2nd Int. Conf. Disruptive Technol. (ICDT)*, 2024, pp. 924–928, doi: 10.1109/ICDT61202.2024.10489682.

[9] Y. Dou, Z. Liu, L. Sun, Y. Deng, H. Peng, and P. S. Yu, "Enhancing graph neural network-based fraud detectors against camouflaged fraudsters," in *Proc. 29th ACM Int. Conf. Inf. Knowl. Manage. (CIKM)*, 2020, pp. 315–324, doi: 10.1145/3340531.3411903.

[10] M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, and C. E. Leiserson, "Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics," arXiv:1908.02591, 2019.

[11] X. Wang, H. Yu, J. Guo, P. Li, and X. Luo, "Towards fraud detection via fine-grained classification of user behavior," *IEEE Trans. Big Data*, vol. 11, no. 4, pp. 1994–2007, 2025, doi: 10.1109/TBDATA.2024.3517313.

[12] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, vol. 10, no. 3, Art. no. e0118432, 2015, doi: 10.1371/journal.pone.0118432.

[13] J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in *Proc. 23rd Int. Conf. Mach. Learn. (ICML)*, 2006, pp. 233–240, doi: 10.1145/1143844.1143874.

[14] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in *Proc. 22nd Int. Conf. Mach. Learn. (ICML)*, 2005, pp. 625–632, doi: 10.1145/1102351.1102430.

[15] B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in *Proc. 8th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2002, pp. 694–699, doi: 10.1145/775047.775151.

[16] G. W. Brier, "Verification of forecasts expressed in terms of probability," *Monthly Weather Rev.*, vol. 78, no. 1, pp. 1–3, 1950.

[17] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001, doi: 10.1214/aos/1013203451.

[18] E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, "PaySim: A financial mobile money simulator for fraud detection," in *Proc. Eur. Model. Simul. Symp. (EMSS)*, 2016, pp. 249–255.

[19] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002, doi: 10.1613/jair.953.

[20] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 30, 2017, pp. 3146–3154.

[21] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 30, 2017, pp. 4765–4774.

[22] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[23] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.


---

## PART 3 — Change Audit

| Original Section | Action | Reason |
|---|---|---|
| Title | **Revised** (author must confirm) | "Real-time … detection system using AI and ML" is not supported: there is no latency benchmark, and the system is an advisory risk assessment, not a detector. The new title keeps graph, UPI, pre-payment, scam link and QR. |
| Authors / affiliation | Kept (formatting corrected) | Fixed "DETECTIONAND", "Artifical", and "DR." (IEEE author lines omit titles). See Part 4 for the department question. |
| Abstract | Compressed | Every number is kept; wording is tightened; the synthetic-data and no-end-to-end caveats are retained. |
| I-A Background, I-B Problem context | Merged | Two motivation paragraphs repeated the same point about instant settlement. |
| I-C Research gap | Merged into Introduction | It restated the gap given again in II-I. The gap now appears once in each section at different depth. |
| I-D Scope, I-E Contributions | Kept (compressed) | The four contributions are kept with unchanged meaning, and the scope disclaimer (not a PSP, no NPCI integration) is kept. |
| I-F Organisation | Compressed | Two sentences. |
| II-A to II-H (eight subsections) | Merged into four thematic paragraphs | The per-paper summaries became a synthesis. Every citation is kept next to its claim. |
| II-I Gap and Table I | Kept; Table I redesigned | Eight columns became five and the table moved to a single column. Every cell is transcribed from the source table. |
| III-A to III-F (formulation) | Compressed and merged | Eight design constraints became six (C5 and C6 combined; C8 folded in as the "no payee verified" statement). Eq. (2) and the relationship status are kept. The entity-status hierarchy is removed because it is descriptive and unscored. |
| IV-A, IV-B (architecture, workflow) | Merged | The workflow figure was dropped and the read/write separation, provenance, deduplication and IST normalisation are stated in prose. The client/service stack names are removed. |
| IV-C and Table II (families) | Kept | Table II is kept. The principal-facts column was removed; the key facts are named in the text and in the contribution column. |
| IV-D and Fig. 3 (graph) | Kept (compressed) | Kept as Fig. 2 at reduced width. Eq. (10) is kept as Eq. (4), with the position-finding gates. |
| IV-E, IV-F, IV-G (model, explanation interfaces, deployment boundary) | Merged into V-A, V-D and Section VIII | This removes three repetitions of the V1-served / V2-offline statement. |
| V (mathematical formulation, 17 equations) | Compressed to Eqs. (1)–(11) | The payer-quantity equations (6)–(9) became prose, the evidence-record equation (5) is inline, and the aggregation, attribution, coverage and verdict equations are kept in full. |
| VI-A (synthetic corpus) and Table III | Kept (compressed); merged with Table IV | All generator facts are kept. |
| VI-B (V1 features, missing data) | Compressed | The bullet list became one sentence. The missing-data policy, observation mask and availability-bias statement are kept. |
| VI-C, VI-D (graph features, V1/V2 representations) | Merged | Every V2 group size, the message-pool composition, the seed-offset check and the simulated-channel caveat are kept. |
| VI-E (temporal boundary, V1 limitation) | Moved | The encoding limitation is in IV-B; the replay limitation is in Section VIII. |
| VII (fraud-risk models) | Compressed into V-A | Hyper-parameters, the five imbalance aspects, the threshold rule and the V1 re-derivation check are kept. The Brier formula is now inline. |
| VIII (aggregation and explainability) | Merged into V-C, V-D | Fact-aware corroboration, risk versus confidence and the advisory meaning of the verdicts are kept. |
| IX (methodology) and Fig. 5 | Compressed; figure removed | Split protocol, result categories, metrics, both ablation protocols, attribution method, software validation scope and environment are kept. The metric equations were removed because they are standard. |
| X-A to X-D and Tables V, VI | Merged into VII-A and Table IV | The V1 confusion counts were added to Table IV; they come from the source text. |
| X-E, X-F and Figs. 6–7 | Merged into VII-B; Fig. 7 removed | The PR/ROC figure is kept. |
| X-G and Table VII (scenario recall) | Kept | The V1 ablation result was moved here next to the scenario it concerns. |
| X-H and Fig. 8 (calibration) | Kept | All bin statistics are kept. |
| X-I and Fig. 9 (importance and SHAP) | Kept; figure redrawn compactly | Only the top ten features per panel are shown, with values identical to the source figure. |
| X-J and Table VIII (ablation) | Kept in full | All twelve rows are kept, together with the comparability caveats. |
| X-K (missing history), X-L and Table IX (software) | Merged into VII-F; Table IX became prose | Every module count (21/18/15/18/19 = 91) and property is kept. |
| XI-A to XI-H (discussion) | Compressed to two paragraphs | No new results. Relation to prior work is covered in Section II. |
| XII-A to XII-J (threats) | Compressed to a seven-item list | Every threat is retained. The security list and repository state are condensed. |
| XIII (conclusion and future work) | Compressed | The future-work list is reduced to the two highest-priority steps named in the source. |
| References | Kept unchanged | 23 entries with identical numbering; no additions, removals or edits. |

---

## PART 4 — Scientific Integrity Check

**Numerical traceability.**
- An automated check matched every number in the compressed manuscript against the source text. All values match; the only numbers not found in the source are the figure-width settings used in the build file.
- V1 confusion counts (TP 175, FN 180, FP 280, TN 29,365) were taken from Section X-B of the source and added to Table IV. They are consistent with the V1 precision (175/455 = 0.3846) and FPR (280/29,645 = 0.00945).

**Numerical inconsistency to resolve before submission.**
- The source reports the V2 validation recall as 0.746 at FPR 0.00995. The repository's later verification of the frozen artefact shows two different values:
  - 0.746 (159 of 213 validation frauds) was computed at the unrounded threshold 0.11227.
  - At the stored threshold 0.1123, which is the one applied to the test split, validation recall is 0.7418 (158 of 213).
- The test result is unaffected either way.
- The compressed paper therefore reports only the validation PR-AUC (0.6995). If you reinstate validation recall, use 0.7418 at θ = 0.1123, or state that 0.746 refers to the unrounded threshold.

**V1/V2 ambiguity.** None found.
- V1 is described throughout as the 12-feature served model.
- V2 is described as the 47-feature research representation, frozen and evaluated offline. It is never described as deployed.
- Table VI states explicitly that its uncalibrated rows (0.7234, 0.4996) are not the V2 (0.7326) or V1 (0.4860) results.

**Unsupported claims (fixed).**
- The source title's "Real-Time … Detection System Using AI and ML" was replaced, because there is no latency benchmark and the system is advisory.
- No other overclaiming wording ("state-of-the-art", "production-ready", "best", "guarantees") appears.
- One comparative phrase is kept because it is supported by Table VI and qualified there: gradient boosting "exceeds" the random forest, extremely randomised trees and logistic regression, in single fits without significance tests.

**Citation issues.**
- All 23 references are retained, unchanged, with their original numbering (first-citation order was preserved).
- Every claim remains cited next to its source.
- Reference [10] (Weber et al.) is an arXiv preprint. If a journal requires peer-reviewed sources, the author should confirm that it is acceptable.
- References [16], [18], [21] and [22] have no DOI in the source; none was invented.

**Items that need author verification.**
1. **Title.** Confirm the revised title, or revert to your own wording without "real-time".
2. **Affiliation.** The uploaded PDF says "Department of Artifical intelligence and data science", but the earlier manuscript said "Department of Information Technology". The spelling has been corrected; confirm which department is right.
3. **Author line.** "DR.K. Kishore Raju" is rendered as "K. Kishore Raju", following IEEE style. Confirm whether the guide should be listed as corresponding author.
4. **Newer experimental evidence exists but is not in the source.** The project's later V2 results package contains:
   - a ten-seed stability study (V2 PR-AUC 0.7196 ± 0.0228; seed 42 ranks 2nd of 10);
   - a 13-configuration, ten-seed family ablation;
   - a chronological three-stream replay of the complete ten-family system with V2 as the classifier family;
   - a full 820-test suite;
   - an 18-item leakage audit.
   
   Under the source-of-truth rule these were not added. Two points follow:
   - The statement that the complete verdict system "was not benchmarked end to end" is accurate for this manuscript but is now superseded by that replay evaluation.
   - The replay also gives important cautionary results: the V2 threshold recalls only about 11% of replay fraud, and about 31% of legitimate payments receive a non-APPROVE verdict.
   
   Incorporating them would strengthen credibility but is a content decision for you and your guide. It would require V2-only results for the replay and would add roughly 0.5–1 page.
5. **Repository state.** The source discloses 81 uncommitted changes, an untracked `clock.py`, and a hash mismatch for `evaluate_model_v2.py`. The compressed paper keeps a one-sentence requirement to commit and tag the code. Resolve this before submission.
6. **Security issues.** The four endpoint issues are summarised by two examples. The full list is in the original Section XII-H and should be fixed, or disclosed in full, if the code is released.

**Limitations that must remain** (all retained):
- synthetic data only;
- simulator-controlled V2 channels and template overlap;
- the graph not exercised predictively;
- no end-to-end verdict evaluation;
- no time-indexed replay;
- distribution shift, calibration on about 170 frauds per fold, and a single seed and split;
- the V1 hours-from-usual encoding;
- not deployed, no provider integration or verification, and no latency benchmark;
- unresolved access-control issues.

**Originality.**
- The prose was rewritten structurally from the authors' own manuscript, not paraphrased word by word.
- The related-work statements describe cited works in new synthesis sentences; no cited paper's wording was reproduced.
- No detector-evasion techniques were used: no synonym spinning, no invisible characters, no Unicode substitution.

---

## PART 5 — Final Page-Length Check

- **Measured length:** 10 pages, rendered with the identical template: A4, two columns, 10-pt Times New Roman, the same margins, a 0.5 cm column gap, and 8-pt references. The same pipeline rendered the original at 19 pages, matching the uploaded PDF.
  - Pages 1–9 hold the body.
  - Page 10 holds references [5]–[23] and is about 60% full.
  - The effective length is about 9.6 pages.
- **Microsoft Word:** re-flowing in Word may shift the length by about ±0.3 page, because line breaking and equation rendering differ slightly. It should remain within 9.5–10 pages.
- **If the venue enforces a strict 9-page limit,** compress in this order. No formatting changes are needed.
  1. Remove Fig. 2, the illustrative graph example (saves about 0.25 page). Eq. (4) and the prose already define the overlap rule.
  2. Shorten Table VI to rows A–F plus the two leave-one-out rows (saves about 0.15 page). The model-family comparison can stay in prose.
  3. Merge the two Discussion paragraphs of Section VIII into one (saves about 0.15 page).
