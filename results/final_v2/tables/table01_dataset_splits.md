**Table 1 — Dataset and split statistics (V2 corpus, seed 42)**

| Split      |   Total |   Legitimate |   Fraud |   Fraud % |
|:-----------|--------:|-------------:|--------:|----------:|
| All        |  120000 |       118581 |    1419 |    1.1825 |
| Train      |   72000 |        71149 |     851 |    1.1819 |
| Validation |   18000 |        17787 |     213 |    1.1833 |
| Test       |   30000 |        29645 |     355 |    1.1833 |

Simulated payers: 2,000 distinct payers realised (generator setting 2,000; counted by instrumenting the generator, whose output was checked to be identical to the cached V2 frame — the V2 frame stores no payer identifier). Synthetic data.
