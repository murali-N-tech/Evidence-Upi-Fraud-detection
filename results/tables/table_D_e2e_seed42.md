| Slice          |     n |   fraud |   Risk PR-AUC |   Risk ROC-AUC |   Classifier-only PR-AUC |   WARN+ recall % |   WARN+ FPR % |   STEP_UP+ recall % |   STEP_UP+ FPR % |   BLOCK recall % |   BLOCK FPR % |   Intervention % |
|:---------------|------:|--------:|--------------:|---------------:|-------------------------:|-----------------:|--------------:|--------------------:|-----------------:|-----------------:|--------------:|-----------------:|
| full_stream    | 60000 |     720 |        0.0494 |         0.7818 |                   0.0423 |            85.28 |         53.69 |               70.56 |            23.12 |            52.22 |        12.564 |            54.07 |
| test_last25pct | 15000 |     172 |        0.1124 |         0.8639 |                   0.0803 |            84.88 |         33.98 |               71.51 |            12.4  |            56.4  |         6.98  |            34.56 |

| group              | slice          |     n |   APPROVE |   WARN |   STEP_UP |   BLOCK |
|:-------------------|:---------------|------:|----------:|-------:|----------:|--------:|
| account_takeover   | full_stream    |   144 |     25    |  19.44 |     28.47 |   27.08 |
| card_testing       | full_stream    |    58 |     86.21 |  13.79 |      0    |    0    |
| legitimate         | full_stream    | 59280 |     46.31 |  30.57 |     10.55 |   12.56 |
| mule_collection    | full_stream    |   158 |      5.7  |  21.52 |     18.99 |   53.8  |
| social_engineering | full_stream    |   360 |      3.06 |  10    |     16.94 |   70    |
| account_takeover   | test_last25pct |    41 |     34.15 |  21.95 |     21.95 |   21.95 |
| card_testing       | test_last25pct |    11 |     90.91 |   9.09 |      0    |    0    |
| legitimate         | test_last25pct | 14828 |     66.02 |  21.58 |      5.42 |    6.98 |
| mule_collection    | test_last25pct |    50 |      4    |  20    |     22    |   54    |
| social_engineering | test_last25pct |    70 |      0    |   4.29 |      8.57 |   87.14 |