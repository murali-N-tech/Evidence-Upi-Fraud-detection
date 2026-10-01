# 3. Problem Formulation and Design Objectives

## 3.1 The Pre-Payment Assessment Problem

We consider a payer who is about to authorise a UPI-style push payment. At the moment of assessment the payment has not been submitted, so the only information available is the payment request, the state of the payer's device and account, and whatever has been observed before that moment. We write a payment request as

$$x = \{u, p, a, t, q, m, l\},$$

where $u$ is the payer, $p$ is the payee identifier as presented to the payer (a VPA, a mobile number or an address extracted from a QR payload), $a$ is the proposed amount, $t$ is the time of assessment, $q$ is the raw QR payload if one was scanned, $m$ is the message that prompted the payment if the payer supplies one, and $l$ is the set of links carried by that message or payload. Any component other than $p$ and $t$ may be absent; we denote an absent component by $\varnothing$. In particular the amount is frequently unknown when a QR code without an embedded amount is scanned, and the message and links exist only when the payer chooses to provide them.

The task is to compute, before submission, three quantities:

- a risk score $R(x) \in \{0, \dots, 99\}$ expressing how strongly the available evidence indicates that the payment should not proceed as requested;
- a coverage score $C(x) \in \{0, \dots, 100\}$ expressing how much of the evidence the system is designed to use was actually available for this request; and
- an advisory verdict $V(x) \in \{\text{APPROVE}, \text{WARN}, \text{STEP\_UP}, \text{BLOCK}\}$.

The verdict is advisory. The system does not submit, hold or reverse payments, and it has no access to the payment switch or to any provider's internal account data. A verdict of APPROVE states only that the computed risk lies below the configured intervention threshold; it is not a statement that the payment is safe.

## 3.2 Information Available Before Payment

Let $\mathcal{H}_{<t}$ denote the set of payment observations recorded strictly before $t$. It contains the payer's own completed payments, payments imported from the payer's bank or application statements, and payments confirmed through the system by other users. From $\mathcal{H}_{<t}$ we derive three views: the payer's behavioural profile $\mathcal{H}_{<t}(u)$, the payer's history with the specific payee $\mathcal{H}_{<t}(u, p)$, and the payee's history across all observed payers $\mathcal{H}_{<t}(\cdot, p)$. We also maintain a payer–payee graph $\mathcal{G}_{<t}$ built from the same observations (Section 4.4).

The pre-payment setting excludes several kinds of information that post-hoc fraud studies commonly use: the payment's settlement outcome, any subsequent complaint or chargeback, the behaviour of the payee after receiving the funds, and the fraud label itself. These are unavailable by construction at time $t$, and the system is designed so that none of them can enter an assessment.

## 3.3 Design Constraints

The architecture is required to satisfy eight constraints. They follow from the pre-payment setting and from the fact that the evidence available for a given request varies widely.

**C1. No post-decision information.** Every quantity used to assess $x$ must be computable from $x$, $\mathcal{H}_{<t}$ and $\mathcal{G}_{<t}$. The payment under assessment is not written to the history or the graph until the payer confirms that it was made, which happens after the assessment has been returned.

**C2. Missing evidence is not evidence of safety.** When a source cannot be read — for example, the payer has too little history for a behavioural baseline, or the payee has never been observed — the assessment must not become more favourable because that source contributed nothing.

**C3. Missing evidence is not evidence of fraud.** Conversely, absence must not raise risk. A first payment to a new payee is common and ordinary, and a payer with no imported statement is not suspicious.

**C4. No double counting.** Several sources read overlapping inputs. A single underlying observation — for instance, an unusual hour of payment noticed by both the behavioural rules and the classifier — must not be credited as independent corroboration.

**C5. Interpretable findings.** Every non-zero contribution to risk must be traceable to a named finding that can be stated to the payer in plain language.

**C6. Risk is separate from availability.** How much evidence exists and what that evidence indicates are different questions, and they must be reported as different quantities.

**C7. Operation before payment.** The assessment must be computed within the interaction in which the payer is deciding, without a batch recomputation of history or graph state per request.

**C8. No claimed provider verification.** The system must not present a payee as verified by a bank, a payment service provider or the national switch unless such a verification has in fact been obtained from that party.

## 3.4 Entity and Relationship Status

Two categorical descriptors summarise what is known about the counterparty. The first, the entity status $\sigma(p) \in \{\text{UNKNOWN}, \text{OBSERVED}, \text{REGISTERED}, \text{VERIFIED}\}$, describes the payee address itself. It is resolved in fixed precedence: VERIFIED if an external verification record exists for the address, REGISTERED if the address belongs to a user registered with the system, OBSERVED if payments to it appear in $\mathcal{H}_{<t}$, and UNKNOWN otherwise. None of these values is a verdict. In particular, VERIFIED can arise only from a verification actually performed by an external party; the prototype described in this paper has no integration with any bank or payment service provider and does not itself produce such records, so in the experiments no payee is VERIFIED. The entity status is defined and resolvable in the implementation, but it is a descriptive label and is not an input to the risk score (constraint C8).

The second, the relationship status $\kappa(u, p) \in \{\text{ESTABLISHED}, \text{NEW}, \text{UNKNOWN}\}$, describes the pair. It is ESTABLISHED when $\mathcal{H}_{<t}(u, p)$ contains at least one payment, NEW when the payer's history is readable and contains none, and UNKNOWN when the payer's history cannot be read at all, for example because no authenticated payer is present. This distinction is the operational form of constraints C2 and C3: NEW is a measured absence, whereas UNKNOWN is the absence of a measurement.

Imported statements complicate the relationship status because bank statements do not always carry a complete payee address. Some statement formats print a truncated fragment of the VPA or only a counterparty name. Such records are classified by identifier type, and a history built from a fragment or a name alone is reported as partial rather than being promoted to an established relationship with the exact address being paid.

## 3.5 Measured Zero Versus Unavailable

The distinction between a zero measurement and a missing one is central to the formulation. Each evidence source $i$ reports a pair $(a_i, z_i)$, where $a_i \in \{0, 1\}$ indicates whether the source was able to observe anything for this request and $z_i$ is its score. If $a_i = 1$, then $z_i \in \{0, \dots, 100\}$, and $z_i = 0$ means that the source looked and found nothing of concern. If $a_i = 0$, then $z_i$ is undefined; it is not zero. This representation lets the aggregation in Section 5 treat "nothing found" and "nothing to look at" differently, which a single numeric score cannot do.

## 3.6 Evidence Levels

Three evidence sources carry most of the information about a specific payment: the payer's behavioural profile, the payee's observed history, and the relationship between the two. We therefore summarise availability with an evidence level $L(x) \in \{\text{FULL}, \text{PARTIAL}, \text{MINIMAL}\}$. The level is FULL when all three core sources are available, PARTIAL when at least the payer-behaviour or payee-history source is available, and MINIMAL otherwise. A MINIMAL assessment rests only on the address, the amount and whatever message or link the payer supplied, and it is presented as such.

## 3.7 Design Objectives

Within these constraints the design pursues four objectives. The first is to detect payer-authorised deception, in which the payer's own behaviour may be unremarkable, by giving payee-side and relational evidence a direct route to the risk score rather than only indirect influence through a classifier. The second is to keep each contribution to risk attributable to a named observation, so that the payer can be told why an intervention was raised. The third is to degrade predictably as evidence disappears: fewer sources should lower the coverage score and the evidence level, not silently change the meaning of the risk score. The fourth is to keep per-request cost bounded, so that the relational evidence can be read during the payer's interaction without recomputation over the full history.

# 4. Proposed System Architecture

## 4.1 Overview

The system is organised as a client application, an assessment service and a persistent store (Figure 1). The client captures the payment request, either by decoding a QR code or by accepting a typed address, and optionally collects the amount, a stated purpose and the message that prompted the payment. The assessment service evaluates the request against ten evidence families, aggregates their outputs and returns the risk score, coverage score, evidence level, verdict and findings. The store holds the payment-observation ledger, the per-payee reputation aggregates, the incremental payer–payee graph and the payer profiles. The prototype is implemented with a TypeScript client, a Python service built on FastAPI, a scikit-learn classifier and an SQLite store.

Two paths touch the store. The assessment path reads it and never writes to it. The observation path writes to it, and it runs only when a payment is recorded: when the payer confirms that a checked payment was made, or when a statement is imported. This separation is what makes constraint C1 enforceable in the running system — the request being assessed cannot influence the state it is assessed against.

## 4.2 Payment Identity and the Observation Ledger

All payment addresses pass through a single normalisation function before they are stored or looked up, so that case differences, surrounding whitespace and URI encodings do not split one payee into several. Every recorded payment is written to an observation ledger with a provenance label: SYSTEM_CONFIRMED for payments confirmed through the assessment flow, STATEMENT_OBSERVED for payments parsed from an imported statement, and OTHER_OBSERVED for other sources. Each observation carries a deduplication key so that the same payment reaching the ledger through two routes — confirmed in the application and later present in a statement — is recorded once.

Statement ingestion converts heterogeneous bank and application statements into this common form. Because statements differ in how much of the counterparty identity they print, each counterparty field is classified as a complete VPA, a partial VPA fragment, a non-VPA identifier, a name only, or unknown. Only complete VPAs are joined to a payee address directly; fragments and names produce the partial-history states described in Section 3.4. All timestamps are normalised to Indian Standard Time before storage so that hour-of-day features and the temporal boundary are computed on one clock.

## 4.3 Evidence Families

The assessment service evaluates ten evidence families. Each returns an evidence record containing its availability $a_i$, its score $z_i$, a severity, a list of findings, and a set of facts. A fact is a label drawn from a closed vocabulary of fourteen underlying observations, grouped by what they describe:

- payer-side facts: AMOUNT_DEVIATION, HOUR_ANOMALY, VELOCITY_DAILY, VELOCITY_HOURLY and PAYER_RELIABILITY;
- payee-side facts: PAYEE_SHAPE, PAYEE_REPORTS, PAYEE_FAMILIARITY and GRAPH_POSITION; and
- context facts: ADDRESS_INTEGRITY, AMOUNT_ABSOLUTE, LINK_REPUTATION, STATED_PURPOSE and MESSAGE_COERCION.

Facts identify what a family observed, as distinct from which family observed it; the aggregation uses them to recognise when two families are reporting the same thing (Section 5.8). The record constructor enforces the evidence semantics of Section 3.5: an unavailable family cannot carry a score, findings or facts. Table 2 lists the families; they are described below in the order of the information they draw on.

**Address and QR (`address_and_qr`).** This family examines the payment address and, where present, the QR payload: whether the address is well formed, whether the handle is plausible, whether the address contains visually confusable characters, and whether the payload is a payment request at all. It is always available. A malformed address, or a payload that is not a payment request, yields a hard-block finding (Section 5.10).

**Payee history (`payee_history`).** This family reads the reputation aggregates maintained for the payee across all observed payers: how long the address has been observed, how many distinct payers have paid it, how many of them returned, whether amounts are unusually uniform, and whether the address has been reported or blocked. Its findings include an unseen payee (a low-weight warning), reported and blocked payees, a mule-like pattern of many one-time payers, a new address receiving heavy traffic, and an established payee (informational). It is available whenever the address can be looked up.

**Payer behaviour (`payer_behaviour`).** This family compares the request with the payer's own profile: amount relative to the payer's typical payment, hour relative to the payer's usual hours, the day's payment count relative to the payer's daily average, novelty of the payee to this payer, and the payer's own record of failed payments. It requires a payer profile. Rule outputs below a notability threshold are mapped to a score of zero, so that an ordinary payment does not place a permanent floor under every assessment.

**Relationship (`relationship`).** This family reports the relationship status $\kappa(u, p)$ together with the history state of the pair — paid once, paid often, new to this payer but observed elsewhere, or partial identity. It is deliberately informational: its score is always zero. Its role is to contribute coverage and to carry the PAYEE_FAMILIARITY fact, not to raise or lower risk directly. It is unavailable when no authenticated payer is present.

**Amount context (`amount_context`).** This family evaluates the amount in absolute terms: against the per-transaction limit of the payment rail and against the size of transfer that is plausible to a counterparty with no established relationship. An amount above the rail limit is a hard-block finding. Comparison with the payer's own norm is excluded here because it belongs to the behavioural families. The family is unavailable when the amount is not known.

**Stated intent (`stated_intent`).** When the payer states what the payment is for, this family checks the stated purpose against known pretexts — for example, "refund" or "the other party asked me to pay" — and against the payee. It is unavailable when no purpose is stated.

**Message pressure (`message_pressure`).** When the payer supplies the message that prompted the payment, this family estimates coercive pressure from cues such as claimed authority, urgency, fear, requests for credentials, secrecy and offers of remote assistance. It is unavailable when no message is supplied.

**Link safety (`link_safety`).** This family inspects URLs carried in the message or payload for indicators such as brand–domain conflict. It is always evaluated, with a score of zero when no links are present.

**Fraud-probability model (`ml_classifier`).** This family obtains a calibrated fraud probability from a gradient-boosting classifier over payer-behaviour and payee features (Section 4.5) and maps it to a bounded score (Section 5.7). It requires a known amount, a payer profile and at least five prior payments by the payer.

**Graph (`graph`).** This family reads the incremental payer–payee graph (Section 4.4). It reports shape findings about the payee's neighbourhood and position findings about the payee's relation to other addresses, but only position findings contribute to its score.

## 4.4 Incremental Payer–Payee Graph

The graph is maintained incrementally on the observation path. Each confirmed or imported payment from payer $u$ to payee $p$ updates a directed edge $(u, p)$ carrying a payment count and first and last timestamps, updates a per-payee aggregate record, and merges the connected components of $u$ and $p$ in a union–find structure. These updates execute within the same database transaction that records the payment, so the graph cannot drift from the ledger. The per-payee record holds the number of distinct payers, the number of payers who have paid more than once, the total number of payments, and the first and last arrival and activity times; from these the graph view derives the repeat ratio, the arrival span, the lifespan and the payment rate. Components are only ever merged, which matches an append-only ledger.

The assessment path reads the graph through a read-only view that never rebuilds it. Shape features are a single indexed lookup of the payee's record; the overlap feature is a bounded two-hop query that finds the peer payee sharing the most payers with $p$.

The graph family distinguishes two kinds of finding. Shape findings describe the payee's own neighbourhood: a large number of one-time payers, all payers arriving within a short window, and a high rate of payments from new payers. These describe the same observation that the payee-history family already scores, so in the graph family they are informational, carry zero weight, and are tagged with the shared fact PAYEE_SHAPE rather than with a graph-specific fact. Position findings describe where the payee sits relative to other addresses. The implemented position finding fires when a substantial share of the payee's payers also pay a single other address and almost none of them pays either address twice — a pattern consistent with collection accounts sharing a victim pool, and not with independent merchants, whose customers return. Only a position finding earns the GRAPH_POSITION fact and a non-zero graph score. Component size is recorded but not used as a finding, because in a connected payment network most addresses belong to one large component.

This design differs from the two graph-based systems discussed in Section 2 in scope and mechanism. Spade+ [5] maintains dense-subgraph detection incrementally under edge insertions, using a peeling procedure to track suspicious dense blocks in a large transaction graph. PGLTSE [7] learns temporal node representations with a graph neural network over transaction sequences. The present system does neither: it computes a small set of explicit, interpretable features about one payee's neighbourhood at request time, and no learned graph embedding is involved. The trade-off is deliberate. Explicit features can be named in a finding shown to a payer, can be computed within a single request, and remain meaningful on the small graphs that a single deployment observes; the cost is that they capture only patterns that have been specified in advance.

## 4.5 Fraud-Probability Model

The classifier is a histogram-based gradient-boosting model wrapped in isotonic probability calibration with five-fold cross-fitting, trained with balanced class weights. Its operating threshold is selected on a validation split and then applied unchanged to a held-out test split. The model is treated as one evidence source among ten rather than as the decision-maker: it contributes a bounded score, and on its own it cannot reach the BLOCK band (Section 5.7).

Two feature representations are reported and must be kept distinct (Table 4). The first, V1, contains twelve features describing the payer's amount relative to their own median, time of day and deviation from usual hours, velocity, and payee age, payer count, repeat ratio and familiarity to this payer, together with an indicator of whether payee history was available. V1 is the representation used by the running prototype. The second, V2, expands the representation to 47 features in nine groups — payer timing, amount, velocity, payee, relationship, graph, message, link and stated intent — each group carrying an explicit availability indicator where its source can be absent. The V2 model was trained, frozen with hashed artefacts and evaluated offline; it is not served by the running prototype. Results for the two representations are reported separately and are not combined.

## 4.6 Aggregation and Verdict

The aggregation stage receives the ten evidence records and produces the outputs defined in Section 3.1. Risk is formed by a damped maximum over available family scores, to which a corroboration bonus is added only when distinct facts are reported by more than one family. Coverage is the weighted share of families that were available. The verdict is obtained from the risk score through fixed bands, with a small set of findings — a blocked payee, a malformed address, a non-payment payload, a missing payee and an amount above the rail limit — overriding the bands and producing BLOCK directly. Section 5 gives the exact rules.

## 4.7 Explanation

The system produces explanations at two levels, which serve different readers. At the evidence level, every assessment returns its findings, each with a severity and a plain-language message, together with the family that raised it and the facts that corroborated it. This is what the payer sees, and it is complete in the sense of constraint C5: every non-zero contribution to risk comes from a named finding. The classifier's contribution is made interpretable in the same vocabulary by attributing facts to it counterfactually — a fact is attributed to the model only if blinding the inputs associated with that fact materially lowers the predicted probability (Section 5.7).

At the model level, feature attributions are computed offline to analyse what the classifier has learned: permutation importance on the held-out test split, scored by average precision, and SHAP values [20] from a tree explainer applied to the uncalibrated booster of the first calibration fold on a sample of test rows. The SHAP values therefore describe the boosted model's score before isotonic calibration; calibration is monotone, so it preserves the ordering of predictions, but SHAP magnitudes are on the uncalibrated scale. These attributions support the analysis in the experimental sections; they are not part of the verdict engine and are not shown to the payer.

## 4.8 Deployment Boundary

The prototype runs as a mobile or web client in front of the assessment service. It has no connection to NPCI, to any payment service provider or to any bank's internal systems, and it does not initiate or intercept payments. The payment history it uses consists of payments confirmed through the application and statements the user chooses to import. Its payee-side evidence is therefore limited to what has been observed by that deployment, which is a principal limitation discussed later in the paper.

# 5. Mathematical Formulation

This section states the quantities computed by the system and the rules that combine them. The rules are those of the implementation; constants are the implemented values, and where a quantity is produced by a family-specific rule set rather than a closed-form expression, we say so rather than inventing one. Notation is summarised in Table 1.

## 5.1 Payment Request and Evidence Families

A payment request is $x = \{u, p, a, t, q, m, l\}$ as defined in Section 3.1, with $\varnothing$ marking an absent component. Let $\mathcal{F} = \{1, \dots, 10\}$ index the evidence families of Section 4.3. Throughout, $h(t) \in \{0, \dots, 23\}$ is the hour of $t$ in Indian Standard Time, and we use a clamped linear map

$$g(v;\, v_0, v_1, y_0, y_1) = y_0 + (y_1 - y_0)\,\min\!\Big(1, \max\!\Big(0, \frac{v - v_0}{v_1 - v_0}\Big)\Big),$$

which carries a measurement $v$ from the interval between $v_0$ and $v_1$ onto the interval between $y_0$ and $y_1$. The implementation uses this single map wherever a measurement is converted into a score, so that no family defines its own response curve.

## 5.2 Pre-Payment Temporal Boundary

Let $\mathcal{H}$ be the observation ledger and let $\tau(o)$ be the time of observation $o$. The history available to an assessment at time $t$ is

$$\mathcal{H}_{<t} = \{\, o \in \mathcal{H} : \tau(o) < t \,\},$$

with the derived views $\mathcal{H}_{<t}(u)$, $\mathcal{H}_{<t}(u, p)$ and $\mathcal{H}_{<t}(\cdot, p)$ restricting $\mathcal{H}_{<t}$ to the payer, the pair and the payee respectively. The payer's history with the payee is computed by a single function that applies the strict inequality explicitly. Every feature and finding for $x$ is a function of $x$, $\mathcal{H}_{<t}$ and $\mathcal{G}_{<t}$ only.

In the running system this boundary is enforced structurally rather than by filtering: the observation for $x$ itself is written only after the payer confirms the payment, which follows the assessment, so at assessment time the stored state contains no observation at or after $t$ for that request. The reputation aggregates and the graph are read in their current state, which coincides with the state before $t$ for a live request. It does not coincide for a retrospective replay of past payments against a store that already contains later ones; replay would require time-indexed aggregates, which the prototype does not maintain. The boundary is therefore a property of the live assessment path, and we do not claim more than that.

## 5.3 Evidence Records and Availability

Family $i$ returns an evidence record

$$e_i(x) = (a_i,\, z_i,\, \Gamma_i,\, \Phi_i),$$

where $a_i \in \{0, 1\}$ is availability, $z_i$ is the family score, $\Gamma_i$ is the list of findings and $\Phi_i \subseteq \mathcal{V}$ is the set of facts, drawn from the fourteen-element vocabulary $\mathcal{V}$ of Section 4.3. The record satisfies

$$a_i = 1 \;\Rightarrow\; z_i \in \{0, \dots, 100\}, \qquad a_i = 0 \;\Rightarrow\; z_i = \bot,\; \Gamma_i = \emptyset,\; \Phi_i = \emptyset,$$

where $\bot$ denotes "undefined". The two cases are not interchangeable: $a_i = 1, z_i = 0$ records that family $i$ observed the request and found nothing of concern, whereas $a_i = 0$ records that it observed nothing. The implementation rejects an unavailable record that carries a score or facts. A family is said to be *active* if $a_i = 1$ and $z_i > 0$.

Where a family raises several findings, its score is produced by that family's own bounded rule. For the amount-context and graph families the rule is the sum of finding weights capped at 100; the payer-behaviour and classifier families use the mappings given in Sections 5.4 and 5.7; the relationship family is fixed at $z = 0$; and the address, payee-history, link and message families use their own rule sets, which are not reproduced here as closed-form expressions. The message family reports zero when only weak cues are present.

## 5.4 Payer Behavioural Quantities

Let $n_u$ be the number of outgoing payments in $\mathcal{H}_{<t}(u)$ and let $b_u$ be the median amount of those payments (the mean is used only for profiles that predate the median). The amount baseline is treated as measurable only if

$$n_u \ge n_{\min} = 5 \quad \text{and} \quad b_u > 0 .$$

When it is measurable, the relative amount is $a / b_u$. The time-of-day quantities are a night indicator and a circular distance from the payer's most active hour $\bar h_u$:

$$\nu(t) = \mathbb{1}\big[h(t) < 6 \;\lor\; h(t) \ge 22\big], \qquad d_u(t) = \min\big(|h(t) - \bar h_u|,\; 24 - |h(t) - \bar h_u|\big).$$

The velocity quantities are the time since the payer's previous payment, capped at one day, $\Delta_u(t) = \min(t - t_u^{\text{last}},\, 86{,}400\ \text{s})$; the number of payments in the preceding hour, $k_u^{(1\text{h})}(t)$; and today's payment count divided by the payer's average daily count, with a denominator of 1 when the average is unknown.

These quantities feed both the payer-behaviour rules and the classifier. The rules produce a raw score $r_u \in [0, 100]$, which is converted into the family score by

$$z_{\text{pb}} = \begin{cases} 0, & r_u < 25,\\[2pt] \operatorname{round}\big(g(r_u;\, 25, 85, 12, 52)\big), & r_u \ge 25. \end{cases}$$

The lower branch prevents the non-zero raw score of an entirely ordinary payment from acting as a permanent floor on risk.

In V1, an unknown usual hour is encoded as $d_u(t) = 0$, which coincides with the value for a payment at exactly the usual hour. This is a known conflation of "unmeasured" with "measured zero" in the V1 representation; V2 addresses the analogous problem for other groups through explicit availability indicators.

## 5.5 Payee and Relationship Quantities

For payee $p$, let $N_p$ be the number of distinct payers in $\mathcal{H}_{<t}(\cdot, p)$, $N^{\text{rep}}_p$ the number of those who paid more than once, and $A_p$ the number of days since $p$ was first observed. The repeat ratio is

$$r_p = \frac{N^{\text{rep}}_p}{\max(N_p, 1)},$$

and it is treated as measurable only when $N_p \ge 5$; below that, a ratio computed from two or three payers says little about the address, so the feature is set to missing rather than to a value. The counts $N_p$ and $A_p$ are exact whenever the payee has been observed and are used at any size.

The relationship status $\kappa(u, p)$ of Section 3.4 is computed from $|\mathcal{H}_{<t}(u, p)|$: ESTABLISHED if positive, NEW if zero and the payer's history is readable, UNKNOWN otherwise. The classifier's payer-relative novelty indicator is

$$\eta(u, p) = \begin{cases} 0, & \kappa(u, p) = \text{ESTABLISHED},\\ 1, & \kappa(u, p) = \text{NEW},\end{cases}$$

and, only when no payer is available, falls back to whether $p$ has been observed at all. In the implementation the fallback is stated as a weaker question rather than hidden.

## 5.6 Incremental Graph Quantities

Let $\mathcal{G}_{<t} = (V_{<t}, E_{<t})$, where $V_{<t}$ contains payers and payees and $E_{<t}$ contains a directed edge $(u, p)$ for each pair with at least one observation, annotated with a payment count and first and last times. Let $P(p) = \{u : (u, p) \in E_{<t}\}$ be the payer set of $p$, so that $N_p = |P(p)|$. On each new observation of $(u, p)$, the edge annotation and the per-payee record are updated in constant time, and the components of $u$ and $p$ are merged by union–find, which has near-constant amortised cost. The graph is never rebuilt on the assessment path.

From the per-payee record we obtain the arrival span $S_p$ (days between the first and last arrival of a new payer), the lifespan $T_p$ (days between first and last activity), the payment rate $\lambda_p$ (payments per day over the lifespan) and the component size $|K(p)|$.

The overlap quantity looks one step beyond the payee. For a candidate peer $p'$ sharing payers with $p$, define

$$\omega(p, p') = \frac{|P(p) \cap P(p')|}{\min\big(|P(p)|,\, |P(p')|\big)},$$

and let $p^\star$ be the peer that maximises $|P(p) \cap P(p')|$, found by a bounded two-hop query from $p$ through its payers. The overlap is discounted — treated as not informative — if either address has fewer than $N_{\min} = 10$ payers, or if the peer's own repeat ratio is at least $\epsilon = 0.1$, which is the signature of a merchant with returning customers rather than of a collection account.

The graph family raises a *position* finding when

$$|P(p) \cap P(p^\star)| \ge 4, \qquad \omega(p, p^\star) \ge 0.6, \qquad r_p < 0.1,$$

in addition to the discounting conditions above. It raises informational *shape* findings when $N_p \ge 10$ and $r_p < 0.1$; when $N_p \ge 10$ and all payers arrived within 30 days; and when $N_p \ge 10$ and $\lambda_p \ge 5$. The graph family's score is

$$z_{\text{gr}} = \min\Big(100,\; \sum_{\gamma \in \Gamma^{\text{pos}}} w(\gamma)\Big),$$

where $\Gamma^{\text{pos}}$ is the set of position findings; shape findings contribute zero. Correspondingly, $\text{GRAPH\_POSITION} \in \Phi_{\text{gr}}$ if and only if $\Gamma^{\text{pos}} \ne \emptyset$, and shape findings are tagged with the shared fact PAYEE_SHAPE, which the payee-history family already reports. The graph family therefore adds risk only for what no other family can observe.

## 5.7 Fraud-Probability Model as an Evidence Source

Let $\phi(x) \in \mathbb{R}^d$ be the feature vector, with $d = 12$ for V1 and $d = 47$ for V2, in which unmeasured entries are represented as missing values that the gradient-boosting learner routes natively rather than being imputed. Let $f$ be the gradient-boosting classifier and $\pi(x) = \hat{P}(y = 1 \mid \phi(x))$ its isotonically calibrated output, and let $\theta$ be the operating threshold selected on the validation split.

The classifier family is available only if the amount is known, a payer profile exists, and $n_u \ge 5$. When it is available, its score is

$$z_{\text{ml}} = \begin{cases} \operatorname{round}\big(g(\pi;\, 0, \theta, 0, 18)\big), & \pi < \theta,\\[2pt] \operatorname{round}\big(g(\pi;\, \theta, 0.80, 30, 66)\big), & \pi \ge \theta. \end{cases}$$

The gap between 18 and 30 separates below-threshold and above-threshold readings, and the ceiling of 66 lies below the BLOCK boundary of 70: the classifier alone can raise a STEP_UP but cannot produce BLOCK without corroboration from another family. This ceiling reflects that the classifier is the family most exposed to inputs outside its training distribution, not that its probabilities are unreliable.

The facts attributed to the classifier are determined counterfactually. The model reads inputs related to six facts — PAYEE_SHAPE, PAYEE_FAMILIARITY, AMOUNT_DEVIATION, HOUR_ANOMALY, VELOCITY_DAILY and VELOCITY_HOURLY — but declaring all six for every prediction would let one probability supply six corroborating observations. For each such fact $v$, let $\phi^{(-v)}(x)$ be a probe vector in which the inputs underlying $v$ are neutralised: for PAYEE_SHAPE the payee-history features are set to missing, a state the model was trained on; for the other facts the inputs are replaced by the value corresponding to the payer's ordinary behaviour (for example, an amount ratio of 1 and zero distance from the usual hour). The probes are used only to attribute facts and never alter $z_{\text{ml}}$. Then

$$v \in \Phi_{\text{ml}} \iff \pi(x) - \pi\big(\phi^{(-v)}(x)\big) \ge \max\big(0.01,\; 0.15\,\pi(x)\big).$$

The relative term makes attribution proportionate — a drop of 0.03 is most of the signal at $\pi = 0.08$ but negligible at $\pi = 0.75$ — and the absolute floor prevents attributing facts to vanishingly small changes. Because these facts are drawn from the same vocabulary as those of the rule-based families, the corroboration rule of Section 5.9 sees a classifier flag on, say, HOUR_ANOMALY as the same observation that the payer-behaviour rules report, not as an independent one.

## 5.8 Base Risk by Damped Maximum

Let $z_{(1)} \ge z_{(2)} \ge \dots \ge z_{(n)}$ be the scores of the active families (Section 5.3), sorted in descending order. The base risk is

$$R_0(x) = \min\Big(99,\; \operatorname{round}\Big(z_{(1)} + \beta \sum_{j=2}^{n} z_{(j)}\Big)\Big), \qquad \beta = 0.4,$$

with $R_0 = 0$ when no family is active. The strongest family sets the level and each further family contributes 40% of its score. Unavailable families do not appear in the sum, so they neither lower nor raise the base risk (constraints C2 and C3); available families with zero score contribute nothing either, but, unlike unavailable ones, they count toward coverage (Section 5.10). Because of the cap, the system never reports a risk of 100.

## 5.9 Fact-Aware Corroboration

Additional risk is granted only when distinct observations agree. Define the set of participating families and the facts they report as

$$\mathcal{A}(x) = \{\, i \in \mathcal{F} : a_i = 1,\; z_i \ge \zeta \,\}, \qquad \Phi^\star(x) = \bigcup_{i \in \mathcal{A}(x)} \Phi_i, \qquad \zeta = 20,$$

and let $\mathcal{A}^\star(x) = \{ i \in \mathcal{A}(x) : \Phi_i \ne \emptyset \}$ be the participating families that report at least one fact. The corroboration strength is

$$k(x) = \min\big(|\Phi^\star(x)|,\; |\mathcal{A}^\star(x)|\big),$$

and the corroboration bonus is

$$B(k) = \begin{cases} 0, & k < 2,\\ 6, & k = 2,\\ 14, & k = 3,\\ 22, & k = 4,\\ 28, & k = 5,\\ 32, & k \ge 6. \end{cases}$$

Both terms of the minimum are needed. The family count prevents a single family that reports several facts — the payer-behaviour rules noticing both an unusual amount and an unusual hour — from corroborating itself. The fact count prevents several families that read the same underlying observation from appearing to agree; if the classifier and the payee-history family both report PAYEE_SHAPE, the observation is counted once. The final risk is

$$R(x) = \min\big(99,\; R_0(x) + B(k(x))\big).$$

Because the relationship family has $z = 0$, it never participates in corroboration, and because shape findings in the graph family carry zero weight, the graph family participates only when a position finding has fired.

## 5.10 Coverage and Evidence Level

Coverage is the weighted share of families that were available:

$$C(x) = \operatorname{round}\!\left(100 \cdot \frac{\sum_{i \in \mathcal{F}} a_i\, c_i}{\sum_{i \in \mathcal{F}} c_i}\right),$$

with the implemented weights $c_i$: payee history 24, relationship 18, payer behaviour 16, classifier 12, address and QR 11, graph 8, amount context 5, message pressure 3, link safety 2 and stated intent 1. The weights sum to 100 and express the evidential value of each family's availability for a specific payment; they are not risk weights and they play no part in $R(x)$. Coverage depends only on the vector $(a_i)$ and not on any score, so a request with high risk may have low coverage and vice versa (constraint C6). Coverage is also deliberately not the classifier's probability; it states how much of the designed evidence was present, not how likely fraud is.

With $\mathcal{C} = \{\text{payee history}, \text{payer behaviour}, \text{relationship}\}$ the core families, the evidence level is

$$L(x) = \begin{cases} \text{FULL}, & a_i = 1 \ \ \forall i \in \mathcal{C},\\ \text{PARTIAL}, & \text{otherwise, if } a_{\text{pb}} = 1 \text{ or } a_{\text{ph}} = 1,\\ \text{MINIMAL}, & \text{otherwise.} \end{cases}$$

## 5.11 Verdict Mapping

Let $\Gamma(x) = \bigcup_i \Gamma_i$ be all findings raised for $x$, and let $\mathcal{B}$ be the hard-block set: a blocked payee, a malformed address, a payload that is not a payment request, a missing payee, an empty request, and an amount above the rail limit. These are conditions under which a request cannot be a valid payment to a legitimate recipient, whatever the arithmetic says. The verdict is

$$V(x) = \begin{cases} \text{BLOCK}, & \Gamma(x) \cap \mathcal{B} \ne \emptyset,\\ \text{BLOCK}, & R(x) \ge 70,\\ \text{STEP\_UP}, & 45 \le R(x) < 70,\\ \text{WARN}, & 22 \le R(x) < 45,\\ \text{APPROVE}, & R(x) < 22. \end{cases}$$

The bands are defined once in the implementation and all verdicts are derived from them. APPROVE denotes that the computed risk is below the configured intervention threshold of 22; it is not a certification that the payment is legitimate, and it is always returned together with $C(x)$ and $L(x)$ so that an APPROVE reached on MINIMAL evidence is distinguishable from one reached on FULL evidence. The verdict is advisory throughout: STEP_UP and BLOCK are recommendations to the payer to verify or not proceed, not actions taken on the payment.

---

# Figure and Table Specifications

## Figure 1 — Proposed system architecture

**Type.** Block diagram, left-to-right data flow, single column width or full page width. Vector (SVG/PDF), greyscale-safe.

**Blocks.**

1. *Client application* (left): QR scanner / address entry; optional inputs — amount, stated purpose, pasted message. Output arrow labelled "payment request $x = \{u,p,a,t,q,m,l\}$".
2. *Assessment service* (centre, largest box), containing:
   - "Normalise address" (single normaliser) → fan-out to ten evidence-family boxes arranged in three labelled bands: **Payer** (payer behaviour; fraud-probability model), **Payee and relation** (payee history; relationship; graph), **Context** (address and QR; amount context; stated intent; message pressure; link safety).
   - Each family box emits a small tuple "$(a_i, z_i, \Gamma_i, \Phi_i)$".
   - Fan-in to "Aggregation": three sub-boxes — "Damped max $R_0$", "Fact-aware corroboration $B(k)$", "Coverage $C$ / level $L$".
   - "Verdict mapping" with a side input "hard-block findings $\mathcal{B}$".
3. *Output* (right): risk $R$, coverage $C$, evidence level $L$, verdict $V$, findings. Return arrow to the client.
4. *Store* (bottom): four cylinders — observation ledger (with provenance labels), reputation aggregates, incremental graph (edges + per-payee record + union–find components), payer profiles.

**Arrows and styling.**
- Solid arrows from store to families = **read-only** assessment path.
- Dashed arrows from client to store labelled "confirm payment / import statement" = **observation path (write)**, entering the store *below* the assessment service, not through it.
- A vertical dotted line between the assessment and observation paths labelled "time $t$: assessment precedes observation".
- The ML box marked "V1 (12 features) — prototype"; a separate, greyed box outside the service labelled "V2 (47 features) — frozen, offline evaluation only", with no arrow into the live path.

**Caption (draft).** *Architecture of the pre-payment assessment system. The assessment path (solid) reads the store and never writes to it; payments enter the ledger, reputation aggregates and graph only through the observation path (dashed) after the payer confirms them or imports a statement. The V2 model is evaluated offline and is not part of the live path.*

**Must not appear.** NPCI, PSP or bank-switch boxes; any arrow suggesting the system submits or stops a payment; the word "safe".

## Figure 2 — Incremental payer–payee graph (conceptual)

**Type.** Conceptual bipartite graph, two panels (a) and (b). Not drawn from real or synthetic data; caption must say "illustrative".

**Panel (a) — graph state $\mathcal{G}_{<t}$.**
- Left column: payer nodes Payer A, Payer B, Payer C, Payer D (circles).
- Right column: payee nodes Payee X, Payee Y (squares).
- Edges: A→X, B→X, C→X, C→Y, D→Y. Edge labels show payment count (e.g., "1") and first time.
- Beside each payee a small record card: $N_p$, $N^{\text{rep}}_p$, $r_p$, $S_p$, $T_p$. Example: X — $N=3$, $N^{\text{rep}}=0$; Y — $N=2$, $N^{\text{rep}}=0$.
- Shaded hull around {A, B, C, D, X, Y} labelled "one component $K$ (union–find)", showing that C links X and Y.
- Highlight $P(X) \cap P(Y) = \{C\}$ and annotate $\omega(X, Y) = 1/\min(3, 2) = 0.5$.

**Panel (b) — incremental update.**
- A new observation "Payer D → Payee X" (dashed, then solid). Show:
  - edge D→X added; X's card updates to $N = 4$;
  - $P(X) \cap P(Y) = \{C, D\}$; $\omega(X,Y) = 2/\min(4,2) = 1.0$;
  - note: "no component merge needed (already same component)".
- Callout: "Assessment for a request at time $t$ reads only $\mathcal{G}_{<t}$; the request's own edge is added only after confirmation."

**Caption note.** State explicitly that the example is far below the implemented thresholds (position findings require at least 10 payers on each address, at least 4 shared payers, $\omega \ge 0.6$ and repeat ratio below 0.1), so no finding would fire; the figure illustrates the data structure and update, not a detection.

## Table 1 — Notation

Columns: **Symbol | Meaning | Domain / value**. Rows, in order of first use:

| Symbol | Meaning | Domain / value |
|---|---|---|
| $x = \{u,p,a,t,q,m,l\}$ | payment request: payer, payee identifier, amount, time, QR payload, message, links | — |
| $\varnothing$ | absent request component | — |
| $R(x)$, $R_0(x)$ | final risk; base risk before corroboration | $\{0,\dots,99\}$ |
| $C(x)$ | coverage score | $\{0,\dots,100\}$ |
| $V(x)$ | advisory verdict | APPROVE, WARN, STEP_UP, BLOCK |
| $L(x)$ | evidence level | FULL, PARTIAL, MINIMAL |
| $\mathcal{H}$, $\mathcal{H}_{<t}$ | observation ledger; observations strictly before $t$ | — |
| $\mathcal{G}_{<t} = (V_{<t}, E_{<t})$ | payer–payee graph from $\mathcal{H}_{<t}$ | — |
| $\sigma(p)$ | entity status (descriptive, not scored) | UNKNOWN, OBSERVED, REGISTERED, VERIFIED |
| $\kappa(u,p)$ | relationship status | ESTABLISHED, NEW, UNKNOWN |
| $\mathcal{F}$, $i$ | evidence families and index | 10 families |
| $e_i = (a_i, z_i, \Gamma_i, \Phi_i)$ | evidence record | — |
| $a_i$ | availability of family $i$ | $\{0,1\}$ |
| $z_i$ | family score | $\{0,\dots,100\}$ or $\bot$ |
| $\Gamma_i$, $\Phi_i$ | findings; facts | $\Phi_i \subseteq \mathcal{V}$ |
| $\mathcal{V}$ | fact vocabulary | 14 facts |
| $g(\cdot)$ | clamped linear map | — |
| $n_u$, $b_u$ | payer's payment count; median amount | $n_{\min} = 5$ |
| $\nu(t)$, $d_u(t)$, $\bar h_u$ | night indicator; circular hour distance; usual hour | — |
| $r_u$ | raw payer-behaviour rule score | $[0,100]$ |
| $N_p$, $N^{\text{rep}}_p$, $A_p$ | distinct payers; repeat payers; days observed | — |
| $r_p$ | payee repeat ratio | measurable if $N_p \ge 5$ |
| $\eta(u,p)$ | payer-relative novelty | $\{0,1\}$ |
| $P(p)$ | payer set of $p$ | — |
| $S_p$, $T_p$, $\lambda_p$, $K(p)$ | arrival span; lifespan; payments/day; component | — |
| $\omega(p,p')$, $p^\star$ | payer overlap; peer of maximal overlap | $[0,1]$ |
| $\phi(x)$, $d$ | feature vector; dimension | 12 (V1), 47 (V2) |
| $\pi(x)$, $\theta$ | calibrated probability; operating threshold | — |
| $\beta$ | damping factor | 0.4 |
| $\zeta$ | participation threshold | 20 |
| $\mathcal{A}$, $\Phi^\star$, $k$, $B(k)$ | participating families; their facts; strength; bonus | $B \le 32$ |
| $c_i$ | coverage weight | sum 100 |
| $\mathcal{B}$ | hard-block findings | 6 codes |

Footnote: $a$ (amount) and $a_i$ (availability) are distinguished by the subscript.

## Table 2 — Evidence families

Columns: **Family (implementation name) | Evidence source | Availability condition | Contribution to $R$ | Facts reported | Coverage weight $c_i$**.

| Family | Source | Available when | Contribution to risk | Facts | $c_i$ |
|---|---|---|---|---|---|
| Payee history (`payee_history`) | reputation aggregates over all payers | address can be looked up | scored; blocked payee is hard-block | PAYEE_REPORTS (reported, blocked); PAYEE_SHAPE (all other findings) | 24 |
| Relationship (`relationship`) | $\mathcal{H}_{<t}(u,p)$ | authenticated payer present | informational, $z=0$ | PAYEE_FAMILIARITY | 18 |
| Payer behaviour (`payer_behaviour`) | payer profile | profile exists | $0$ if $r_u<25$, else 12–52 | AMOUNT_DEVIATION, HOUR_ANOMALY, VELOCITY_DAILY, PAYEE_FAMILIARITY, PAYER_RELIABILITY | 16 |
| Fraud-probability model (`ml_classifier`) | calibrated V1 classifier | amount known, profile, $n_u \ge 5$ | 0–18 below $\theta$; 30–66 above | subset of PAYEE_SHAPE, PAYEE_FAMILIARITY, AMOUNT_DEVIATION, HOUR_ANOMALY, VELOCITY_DAILY, VELOCITY_HOURLY, by counterfactual probe | 12 |
| Address and QR (`address_and_qr`) | address, QR payload | always | scored; hard-block codes | ADDRESS_INTEGRITY | 11 |
| Graph (`graph`) | $\mathcal{G}_{<t}$ | payee in graph | position findings only | GRAPH_POSITION (position); PAYEE_SHAPE (shape, weight 0) | 8 |
| Amount context (`amount_context`) | amount vs rail limit and unfamiliar payee | amount known | scored; over-limit hard-block | AMOUNT_ABSOLUTE | 5 |
| Message pressure (`message_pressure`) | payer-supplied message | message supplied | scored; 0 if weak cues only | MESSAGE_COERCION | 3 |
| Link safety (`link_safety`) | URLs in message/payload | always (0 if no link) | scored | LINK_REPUTATION | 2 |
| Stated intent (`stated_intent`) | payer-stated purpose | purpose stated | scored | STATED_PURPOSE | 1 |

Rows ordered by coverage weight. Per-finding weights are not listed; the table reports only implemented family-level constants. Table note: PAYER_RELIABILITY refers to the payer's own failed-payment history; VELOCITY_HOURLY is reached only through the classifier.

## Table 3 — Feature groups and availability (V2)

Columns: **Group | Features (count) | Source | Availability indicator | Value when unavailable**.

| Group | Features | Source | Indicator | When unavailable |
|---|---|---|---|---|
| Payer timing (2) | is_night, hours_from_usual | payer profile, $h(t)$ | none | hours_from_usual = 0 when usual hour unknown (inherited from V1) |
| Amount (2) | amount, amount_over_user_avg | request, $b_u$ | none | model not invoked if amount unknown |
| Velocity (3) | seconds_since_last_txn, txns_last_hour, txns_today_over_avg | $\mathcal{H}_{<t}(u)$ | none | defaults of a quiet payer (no recent payments) |
| Payee (5) | payee_age_days, payee_distinct_payers, payee_repeat_ratio, payee_history_available, payee_is_phone_address | reputation aggregates | payee_history_available | missing (NaN); repeat ratio missing if $N_p<5$ |
| Relationship (4) | payee_is_new, prior_payments_to_payee, days_since_first_payment_to_payee, relationship_available | $\mathcal{H}_{<t}(u,p)$ | relationship_available (always 1: the payer's ledger is readable) | zero prior payments is a measurement; relationship age missing when no prior payment |
| Graph (7) | arrival_span_days, lifespan_days, payments_per_day, component_size, shared_payer_peak, overlap_ratio, graph_available | $\mathcal{G}_{<t}$ | graph_available | whole group missing if payee not in graph; overlap_ratio missing below the payer-count gate |
| Message (18) | 10 cue indicators, coercion probability and score, weak-cues flag, length, word count, digit and upper-case ratios, message_available | payer-supplied message | message_available | whole group missing |
| Link (3) | link_present, link_suspicious_score, link_brand_conflict | URLs in the message | message_available | missing when no message; zeros when a message has no link (a measurement) |
| Intent (3) | intent_stated, intent_is_refund_or_asked, intent_payee_mismatch | stated purpose | intent_stated | intent_stated = 0; the other two missing |
| **Total** | **47** | | | |

"Missing" means NaN routed natively by the learner, never an imputed value. Table note: the relationship group is always available in V2 because the payer's own ledger can always be read in the offline corpus; in the live path the relationship family is unavailable without an authenticated payer.

## Table 4 — V1 versus V2 feature architecture

Columns: **Aspect | V1 | V2**.

| Aspect | V1 | V2 |
|---|---|---|
| Features | 12 | 47 in 9 groups |
| Groups covered | payer timing, amount, velocity, payee (incl. payer-relative novelty) | V1 groups + relationship, graph, message, link, stated intent |
| Missing-value handling | NaN for unmeasured payee counts and repeat ratio; one availability flag | explicit availability indicator per absent-capable group |
| Learner | HistGradientBoosting, isotonic calibration (5-fold), balanced class weight | same configuration |
| Threshold | selected on validation, applied once to test | same protocol |
| Status | used by the running prototype | trained, frozen (hashed artefacts), evaluated offline only |
| Role in live verdict | yes, via ML family (capped at 66) | none |
| Reported results | Section on results, labelled V1 | Section on results, labelled V2 |

Table note: *V1 and V2 results are never pooled or compared as if the prototype used V2.*
