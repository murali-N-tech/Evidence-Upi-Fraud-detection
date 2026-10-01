**Table 4 — Overall V2 classification performance (frozen model, seed 42)**

| Metric                     |   Validation (n=18,000) |   Test (n=30,000) |
|:---------------------------|------------------------:|------------------:|
| ROC-AUC                    |                  0.9868 |            0.99   |
| PR-AUC (average precision) |                  0.6995 |            0.7326 |
| Accuracy                   |                  0.9871 |            0.9876 |
| Precision                  |                  0.4716 |            0.4843 |
| Recall (TPR)               |                  0.7418 |            0.7803 |
| F1                         |                  0.5766 |            0.5976 |
| Specificity (TNR)          |                  0.99   |            0.99   |
| FPR                        |                  0.01   |            0.01   |
| FNR                        |                  0.2582 |            0.2197 |
| NPV                        |                  0.9969 |            0.9973 |
| Balanced accuracy          |                  0.8659 |            0.8852 |
| MCC                        |                  0.5856 |            0.6091 |
| Brier score                |                  0.0059 |            0.0056 |
| Brier skill score          |                  0.4936 |            0.5251 |
| Threshold                  |                  0.1123 |            0.1123 |
| Fraud prevalence           |                  0.0118 |            0.0118 |
| Alert rate                 |                  0.0186 |            0.0191 |

Threshold 0.1123 selected on the validation split (highest validation recall with validation FPR ≤ 1%). Validation values are the data the threshold was chosen on; test values are the single held-out measurement.
