**Table 11 — Chronological graph-replay validation (three streams)**

| Test                                          |   Seed 42 checked |   Seed 43 checked |   Seed 44 checked |   Violations (all seeds) | Result   |
|:----------------------------------------------|------------------:|------------------:|------------------:|-------------------------:|:---------|
| 1_first_pair_is_new                           |             60000 |             60000 |             60000 |                        0 | PASS     |
| 2_prior_pair_count_matches_earlier_rows       |             60000 |             60000 |             60000 |                        0 | PASS     |
| 3_distinct_payers_only_prior                  |             60000 |             60000 |             60000 |                        0 | PASS     |
| 4_arrival_window_only_prior                   |             47980 |             46950 |             47704 |                        0 | PASS     |
| 5_component_is_pre_payment                    |             47980 |             46950 |             47704 |                        0 | PASS     |
| 6_position_finding_only_when_gates_hold       |             47980 |             46950 |             47704 |                        0 | PASS     |
| 7_unobserved_payee_graph_unavailable_not_zero |             12020 |             13050 |             12296 |                        0 | PASS     |
| 8_independent_rebuild_prior_only_identical    |                51 |                54 |                60 |                        0 | PASS     |
| 9a_earlier_observations_not_rewritten         |             30000 |             30000 |             30000 |                        0 | PASS     |
| 10_store_contains_only_earlier_payments       |             60000 |             60000 |             60000 |                        0 | PASS     |

Tests 1–7 and 10 are checked online at every assessment against an independent shadow model of earlier payments; 8 by an independent rebuild of the store from earlier payments only; 9a by comparing observation rows written before the stream midpoint with the final store.
