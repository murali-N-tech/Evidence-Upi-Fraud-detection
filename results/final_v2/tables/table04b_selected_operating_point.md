**Table 4b — Selected operating point**

| Split                   |   Threshold |   Recall |     FPR |   Precision |     F1 |   Specificity |   Predicted fraud rate |
|:------------------------|------------:|---------:|--------:|------------:|-------:|--------------:|-----------------------:|
| Validation (selection)  |      0.1123 |   0.7418 | 0.00995 |      0.4716 | 0.5766 |          0.99 |                 0.0186 |
| Test (frozen threshold) |      0.1123 |   0.7803 | 0.00995 |      0.4843 | 0.5976 |          0.99 |                 0.0191 |

Selection rule: highest validation recall with validation FPR ≤ 1% (train_model_v2.recall_at_fpr). Re-derived threshold 0.112265 → rounded 0.1123 (stored 0.1123).
