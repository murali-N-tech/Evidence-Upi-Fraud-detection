**Table 6 — Scenario-level fraud detection (V2, test split, threshold 0.1123)**

| Scenario (synthetic)      |   Test fraud | Detected (TP)   | Missed (FN)   | Recall               |   Median probability |   Mean probability |
|:--------------------------|-------------:|:----------------|:--------------|:---------------------|---------------------:|-------------------:|
| Social engineering        |          182 | 120             | 62            | 0.6593               |              0.2838  |             0.3705 |
| Mule collection           |           80 | 68              | 12            | 0.8500               |              0.4309  |             0.438  |
| Account takeover          |           73 | 69              | 4             | 0.9452               |              0.8892  |             0.7859 |
| Card testing              |           20 | 20              | 0             | 1.0000               |              0.9876  |             0.8679 |
| Legitimate (false alarms) |        29645 | —               | —             | FPR 0.00995 (295 FP) |              4.1e-05 |             0.0059 |

Per-scenario precision and F1 are not reported: false positives are legitimate payments and belong to no scenario, so a scenario-specific precision is not defined. Scenario categories and proportions are simulator settings.
