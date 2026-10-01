**Table 9 — Ten-family end-to-end evaluation (chronological replay, V2 as the Classifier family)**

| Slice                             | Quantity                                                                  | Value (mean ± sd over 3 streams)   |
|:----------------------------------|:--------------------------------------------------------------------------|:-----------------------------------|
| Last 25% of each stream (primary) | Payments / fraud per stream                                               | 15,000 / 211 ± 36                  |
| Last 25% of each stream (primary) | Fraud prevalence (%)                                                      | 1.40 ± 0.24                        |
| Last 25% of each stream (primary) | Ten-family risk R — PR-AUC                                                | 0.120 ± 0.017                      |
| Last 25% of each stream (primary) | Ten-family risk R — ROC-AUC                                               | 0.850 ± 0.016                      |
| Last 25% of each stream (primary) | Frozen V2 probability alone — PR-AUC (rows where the family is available) | 0.246 ± 0.013                      |
| Last 25% of each stream (primary) | Frozen V2 probability alone — ROC-AUC (same rows)                         | 0.901 ± 0.019                      |
| Last 25% of each stream (primary) | Frozen V2 alone at θ = 0.1123 — recall / FPR (%)                          | 10.8 ± 3.0 / 0.25 ± 0.04           |
| Last 25% of each stream (primary) | Verdict WARN or above — recall / FPR (%)                                  | 83.0 ± 2.6 / 31.1 ± 0.6            |
| Last 25% of each stream (primary) | Verdict STEP_UP or above — recall / FPR (%)                               | 67.8 ± 3.7 / 12.6 ± 0.9            |
| Last 25% of each stream (primary) | Verdict BLOCK — recall / FPR (%)                                          | 51.0 ± 6.4 / 6.78 ± 0.58           |
| Last 25% of each stream (primary) | Intervention rate, all payments (non-APPROVE, %)                          | 31.9 ± 0.7                         |
| Full stream                       | Payments / fraud per stream                                               | 60,000 / 720 ± 0                   |
| Full stream                       | Fraud prevalence (%)                                                      | 1.20 ± 0.00                        |
| Full stream                       | Ten-family risk R — PR-AUC                                                | 0.047 ± 0.004                      |
| Full stream                       | Ten-family risk R — ROC-AUC                                               | 0.774 ± 0.010                      |
| Full stream                       | Frozen V2 probability alone — PR-AUC (rows where the family is available) | 0.119 ± 0.012                      |
| Full stream                       | Frozen V2 probability alone — ROC-AUC (same rows)                         | 0.875 ± 0.020                      |
| Full stream                       | Frozen V2 alone at θ = 0.1123 — recall / FPR (%)                          | 8.7 ± 0.9 / 0.50 ± 0.02            |
| Full stream                       | Verdict WARN or above — recall / FPR (%)                                  | 84.2 ± 1.2 / 52.3 ± 1.0            |
| Full stream                       | Verdict STEP_UP or above — recall / FPR (%)                               | 69.4 ± 1.9 / 23.8 ± 0.6            |
| Full stream                       | Verdict BLOCK — recall / FPR (%)                                          | 50.9 ± 1.4 / 12.99 ± 0.45          |
| Full stream                       | Intervention rate, all payments (non-APPROVE, %)                          | 52.7 ± 1.0                         |

Verdicts are advisory states of the risk score, not ground-truth classes; recall/FPR at a verdict level are operating points of R. Three replay streams (seeds 42–44), 60,000 payments each; the last 25% (days ≈112–150) follows a warm-up of the shared store.
