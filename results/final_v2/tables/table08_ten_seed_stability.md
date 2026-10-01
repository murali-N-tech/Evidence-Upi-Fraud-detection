**Table 8 — V2 ten-seed stability (seeds 42–51; test split of 30,000 per seed)**

| Metric                          |     Mean |     SD |      Min |       Max |   Seed 42 (original run) |
|:--------------------------------|---------:|-------:|---------:|----------:|-------------------------:|
| PR-AUC                          |  0.7196  | 0.0228 |  0.6829  |   0.7693  |                  0.7326  |
| ROC-AUC                         |  0.9882  | 0.0021 |  0.9833  |   0.9911  |                  0.99    |
| Recall at ~1% FPR (%)           | 75.66    | 3.16   | 70.75    |  80.99    |                 78.03    |
| Realised test FPR (%)           |  0.991   | 0.078  |  0.901   |   1.174   |                  0.995   |
| Precision (%)                   | 48.16    | 2.44   | 44.41    |  52.41    |                 48.43    |
| F1                              |  0.5883  | 0.0243 |  0.5564  |   0.6364  |                  0.5976  |
| Brier score                     |  0.00573 | 0.0003 |  0.00516 |   0.00614 |                  0.00555 |
| Brier skill score               |  0.5166  | 0.023  |  0.4858  |   0.5687  |                  0.5251  |
| Validation threshold            |  0.1326  | 0.0143 |  0.1123  |   0.1544  |                  0.1123  |
| Recall — social engineering (%) | 62.36    | 4.48   | 55.98    |  67.82    |                 65.93    |
| Recall — mule collection (%)    | 82.98    | 3.73   | 76.47    |  89.29    |                 85       |
| Recall — account takeover (%)   | 93.64    | 3.12   | 88.89    |  98.67    |                 94.52    |
| Recall — card testing (%)       | 97.26    | 3.08   | 92.59    | 100       |                100       |

Each seed regenerates the corpus, re-splits (stratified, random_state = seed), refits the calibrated V2 model, selects the threshold on its own validation split (FPR ≤ 1%) and measures the test split once. Statistics are over the ten seed-level values (sd with ddof = 1). Seed 42 reproduces the frozen V2 artefact exactly.
