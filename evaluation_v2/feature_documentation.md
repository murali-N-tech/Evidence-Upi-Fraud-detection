# V2 feature documentation

Every feature added in V2, with the checks the research brief asks for.
All are computed from information that exists before the payment is made;
none reads the label, a later transaction, or anything the payment itself creates.

### `prior_payments_to_payee`  (relationship)

- **Definition**: Count of payments this payer has already made to this payee, before T.
- **Source**: Payer's own ledger (payee_payers edge in production)
- **Available at pre-payment time**: Yes
- **Potential leakage**: None: counts completed past payments only; the payment under decision is excluded.
- **Why it represents fraud behaviour**: A first payment to a stranger is the setting for almost every scam; an established relationship is the strongest exculpatory signal a payer-side system has.
- **Train/serve parity**: dataset_v2._relationship_row / features_v2.relationship_features - one function each side

### `days_since_first_payment_to_payee`  (relationship)

- **Definition**: Days since this payer first paid this payee. NaN when there is no prior payment.
- **Source**: Payer's own ledger
- **Available at pre-payment time**: Yes
- **Potential leakage**: None. NaN carries 'no relationship' rather than 0, which would read as 'started today'.
- **Why it represents fraud behaviour**: A relationship that has existed for months is hard to fake; a one-day-old one is what a mule account has.
- **Train/serve parity**: same functions both sides

### `relationship_available`  (relationship)

- **Definition**: 1.0 always: the payer's own ledger is always readable.
- **Source**: Payer's own ledger
- **Available at pre-payment time**: Yes
- **Potential leakage**: None.
- **Why it represents fraud behaviour**: Distinguishes 'no prior payments' (a measurement) from 'we could not look'.
- **Train/serve parity**: same

### `arrival_span_days`  (graph)

- **Definition**: Days between the first and most recent NEW payer arriving at this payee.
- **Source**: graph_cache.payee_graph_cache (arrival_first/arrival_last)
- **Available at pre-payment time**: Yes - maintained incrementally as payments are confirmed
- **Potential leakage**: None: only edges observed before T. NaN when the payee was never observed.
- **Why it represents fraud behaviour**: A collection account gathers its payers in a burst; a shop gathers them over months.
- **Train/serve parity**: graph_cache.graph_view -> features_v2.graph_features; generator mirrors the same fields

### `lifespan_days`  (graph)

- **Definition**: Days between the payee's first and last observed activity.
- **Source**: graph_cache cache row
- **Available at pre-payment time**: Yes
- **Potential leakage**: None; NaN when unobserved.
- **Why it represents fraud behaviour**: Mule accounts are short-lived by design.
- **Train/serve parity**: same

### `payments_per_day`  (graph)

- **Definition**: Observed payments divided by observed lifespan.
- **Source**: graph_cache cache row
- **Available at pre-payment time**: Yes
- **Potential leakage**: None; NaN when unobserved.
- **Why it represents fraud behaviour**: Rate, not raw volume: a legitimate busy merchant has a high rate AND a long lifespan, which is why the two are separate features.
- **Train/serve parity**: same

### `component_size`  (graph)

- **Definition**: Size of the connected component this payee sits in (persisted union-find).
- **Source**: graph_components table
- **Available at pre-payment time**: Yes
- **Potential leakage**: None: components only grow as edges are observed; no future edge is read.
- **Why it represents fraud behaviour**: Rings sit in small, tight components; popular merchants sit in large ones. Size alone is NOT a fraud signal, which is why it is a feature and not a rule.
- **Train/serve parity**: graph_cache.component_size -> features_v2

### `shared_payer_peak`  (graph)

- **Definition**: Largest number of payers this payee shares with any single other payee.
- **Source**: Bounded two-hop query over payee_payers
- **Available at pre-payment time**: Yes
- **Potential leakage**: None.
- **Why it represents fraud behaviour**: Ring members share their payer pool; unrelated merchants mostly do not.
- **Train/serve parity**: same

### `overlap_ratio`  (graph)

- **Definition**: shared_payer_peak divided by the smaller of the two payer counts. NaN unless BOTH payees have >= 10 payers.
- **Source**: Bounded two-hop query, gated by MIN_PAYERS_FOR_OVERLAP
- **Available at pre-payment time**: Yes
- **Potential leakage**: None. The gate is the fix for the household false positive: five people paying two new shops produce ratio 1.0 over a tiny pool.
- **Why it represents fraud behaviour**: Normalised overlap, not raw degree - the form that survived the false-positive audit.
- **Train/serve parity**: gate mirrored in generator and in features_v2.graph_features

### `graph_available`  (graph)

- **Definition**: 1.0 when a graph row exists for this payee, else 0.0.
- **Source**: graph_cache
- **Available at pre-payment time**: Yes
- **Potential leakage**: None.
- **Why it represents fraud behaviour**: Keeps 'no graph position' distinct from 'a quiet graph position'.
- **Train/serve parity**: same

### `msg_<10 pattern flags>`  (message)

- **Definition**: Binary: did this coercion pattern fire in the message (authority, urgency, fear, pay_request, credential_request, secrecy, remote_control, verify_payment, lure, contact_offline).
- **Source**: coercion.analyse_message - the production extractor
- **Available at pre-payment time**: Yes: the payer pastes the message before paying
- **Potential leakage**: None: the text is an input to the decision, not a product of it.
- **Why it represents fraud behaviour**: Scam messages apply pressure; the hard negatives in the corpus contain the same vocabulary, so a flag alone is weak evidence.
- **Train/serve parity**: identical function called from generator and from features_v2

### `msg_coercion_probability`  (message)

- **Definition**: Fitted logistic probability over the 10 patterns.
- **Source**: coercion.analyse_message (models/coercion_weights.json)
- **Available at pre-payment time**: Yes
- **Potential leakage**: None. Fitted on data/coercion_corpus.jsonl, which is independent of the transaction simulator.
- **Why it represents fraud behaviour**: Summarises pressure in one number; carries most of the message signal in the importance table.
- **Train/serve parity**: same

### `msg_coercion_score / msg_weak_cues_only`  (message)

- **Definition**: The production 0-100 score and the flag for 'only weak cues fired'.
- **Source**: coercion.analyse_message
- **Available at pre-payment time**: Yes
- **Potential leakage**: None.
- **Why it represents fraud behaviour**: weak_only is the hard-negative discriminator: bank notices trip weak cues and nothing else.
- **Train/serve parity**: same

### `msg_length_chars / msg_word_count / msg_digit_ratio / msg_uppercase_ratio`  (message)

- **Definition**: Structural properties of the text.
- **Source**: Computed from the message
- **Available at pre-payment time**: Yes
- **Potential leakage**: None.
- **Why it represents fraud behaviour**: Scam text is often long, numeric (amounts, reference numbers) and shouty; so are some legitimate bank alerts, which is the point.
- **Train/serve parity**: same

### `message_available`  (message)

- **Definition**: 1.0 when a message was supplied.
- **Source**: Request
- **Available at pre-payment time**: Yes
- **Potential leakage**: None. Deliberately near-uninformative: 55% of legitimate rows and 62% of social-engineering rows carry one (AUC 0.40-0.60, asserted by test).
- **Why it represents fraud behaviour**: Distinguishes 'no text to read' from 'text with no pressure'.
- **Train/serve parity**: same

### `link_present / link_suspicious_score / link_brand_conflict`  (link)

- **Definition**: Whether the message carried a URL, the production analyser's score for the worst one, and whether the brand the URL claims differs from the payee.
- **Source**: scam_link.analyse_links / analyse_url
- **Available at pre-payment time**: Yes
- **Potential leakage**: None. NaN when there is no message; 0.0 when there is a message with no link - a measurement.
- **Why it represents fraud behaviour**: Look-alike domains impersonating a bank are a scam staple; legitimate messages carry real biller and bank domains at a similar rate.
- **Train/serve parity**: identical function both sides

### `intent_stated / intent_is_refund_or_asked / intent_payee_mismatch`  (intent)

- **Definition**: Whether the payer stated a purpose, whether it is one of the scam-adjacent purposes, and whether a bill/investment is going to a personal phone address.
- **Source**: The payer's own selection from intent.INTENTS
- **Available at pre-payment time**: Yes
- **Potential leakage**: None: it is what the payer says before paying.
- **Why it represents fraud behaviour**: Victims often describe the scam truthfully ('a refund', 'I was asked to'); the mismatch case is a purpose that does not fit the recipient.
- **Train/serve parity**: dataset_v2.intent_features / features_v2 - one function

### `payee_is_phone_address`  (payee)

- **Definition**: 1.0 when the payee address is a personal phone number rather than a merchant handle.
- **Source**: Address parsing (upi_qr)
- **Available at pre-payment time**: Yes
- **Potential leakage**: None.
- **Why it represents fraud behaviour**: Collection accounts are more often phone-number addresses; so are many honest small sellers, so it is weak on its own.
- **Train/serve parity**: parsed identically both sides
