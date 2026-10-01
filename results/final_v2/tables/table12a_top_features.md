**Table 12a — Top 15 V2 features by permutation importance (test split)**

|   Rank | Feature                           | Evidence family   |   Permutation importance (AP drop) |     sd |   Mean |SHAP| (booster) |
|-------:|:----------------------------------|:------------------|-----------------------------------:|-------:|------------------------:|
|      1 | payee_is_new                      | Relationship      |                             0.3656 | 0.0158 |                  0.8397 |
|      2 | arrival_span_days                 | Graph             |                             0.2455 | 0.0152 |                  0.2944 |
|      3 | amount_over_user_avg              | Payer behaviour   |                             0.2059 | 0.0145 |                  0.158  |
|      4 | msg_coercion_probability          | Message pressure  |                             0.1379 | 0.0105 |                  0.0755 |
|      5 | component_size                    | Graph             |                             0.1209 | 0.0037 |                  0.1488 |
|      6 | txns_last_hour                    | Payer behaviour   |                             0.1018 | 0.0056 |                  0.0472 |
|      7 | days_since_first_payment_to_payee | Relationship      |                             0.0765 | 0.0124 |                  0.0936 |
|      8 | seconds_since_last_txn            | Payer behaviour   |                             0.0732 | 0.0076 |                  0.081  |
|      9 | lifespan_days                     | Graph             |                             0.0587 | 0.0101 |                  0.072  |
|     10 | txns_today_over_avg               | Payer behaviour   |                             0.0534 | 0.0049 |                  0.0442 |
|     11 | payee_distinct_payers             | Payee history     |                             0.0383 | 0.0022 |                  0.0956 |
|     12 | payee_repeat_ratio                | Payee history     |                             0.0319 | 0.0049 |                  0.039  |
|     13 | intent_is_refund_or_asked         | Stated intent     |                             0.031  | 0.0052 |                  0.0387 |
|     14 | payee_is_phone_address            | Address and QR    |                             0.03   | 0.0044 |                  0.0844 |
|     15 | payee_age_days                    | Payee history     |                             0.0181 | 0.0056 |                  0.0523 |

Importance = mean drop in average precision when the column is shuffled (5 repeats). Correlated features share importance; importance is predictive usefulness under this protocol, not causal influence, and is distinct from the ten-family risk aggregation.
