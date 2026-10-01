**Table 12b — Evidence-family importance (V2, test split)**

| Family           |   Number of features |   Grouped permutation importance (AP drop) |     sd |   Sum of single-feature importances |
|:-----------------|---------------------:|-------------------------------------------:|-------:|------------------------------------:|
| Relationship     |                    4 |                                     0.4208 | 0.01   |                              0.4426 |
| Payer behaviour  |                    6 |                                     0.405  | 0.0152 |                              0.4476 |
| Graph            |                    7 |                                     0.3883 | 0.0145 |                              0.4345 |
| Message pressure |                   18 |                                     0.1702 | 0.0109 |                              0.1377 |
| Payee history    |                    4 |                                     0.0774 | 0.0055 |                              0.0883 |
| Stated intent    |                    3 |                                     0.0349 | 0.003  |                              0.0308 |
| Address and QR   |                    1 |                                     0.0292 | 0.0051 |                              0.03   |
| Amount context   |                    1 |                                     0.0113 | 0.0023 |                              0.0096 |
| Link safety      |                    3 |                                     0      | 0.001  |                             -0.0004 |

Grouped permutation importance shuffles all columns of a family together (5 repeats), which avoids splitting importance between correlated columns. The Classifier family has no input column of its own.
