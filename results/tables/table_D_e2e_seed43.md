| Slice          |     n |   fraud |   Risk PR-AUC |   Risk ROC-AUC |   Classifier-only PR-AUC |   WARN+ recall % |   WARN+ FPR % |   STEP_UP+ recall % |   STEP_UP+ FPR % |   BLOCK recall % |   BLOCK FPR % |   Intervention % |
|:---------------|------:|--------:|--------------:|---------------:|-------------------------:|-----------------:|--------------:|--------------------:|-----------------:|-----------------:|--------------:|-----------------:|
| full_stream    | 60000 |     720 |        0.0417 |         0.7625 |                   0.039  |            83.61 |         55.76 |               67.08 |            24.22 |            49.17 |        13.429 |            56.09 |
| test_last25pct | 15000 |     242 |        0.0988 |         0.8361 |                   0.0889 |            81.82 |         35.44 |               63.64 |            13.53 |            45.04 |         7.203 |            36.19 |

| group              | slice          |     n |   APPROVE |   WARN |   STEP_UP |   BLOCK |
|:-------------------|:---------------|------:|----------:|-------:|----------:|--------:|
| account_takeover   | full_stream    |   144 |     34.72 |  27.08 |     18.75 |   19.44 |
| card_testing       | full_stream    |    58 |     79.31 |  13.79 |      1.72 |    5.17 |
| legitimate         | full_stream    | 59280 |     44.24 |  31.54 |     10.79 |   13.43 |
| mule_collection    | full_stream    |   158 |      3.16 |  17.09 |     21.52 |   58.23 |
| social_engineering | full_stream    |   360 |      4.72 |  12.5  |     18.61 |   64.17 |
| account_takeover   | test_last25pct |    50 |     36    |  28    |     18    |   18    |
| card_testing       | test_last25pct |    19 |     78.95 |  10.53 |      5.26 |    5.26 |
| legitimate         | test_last25pct | 14758 |     64.56 |  21.91 |      6.33 |    7.2  |
| mule_collection    | test_last25pct |    49 |      6.12 |  16.33 |     20.41 |   57.14 |
| social_engineering | test_last25pct |   124 |      6.45 |  16.13 |     20.16 |   57.26 |