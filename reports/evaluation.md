# Model evaluation

Training run `20261015T175504`. Train < 2026-07-01 <= validation < 2026-08-01 <= test.
Rows: train 30166, validation 10454, test 9380.

## Incoming Call Last 24h: with vs. without

Both variants are trained on every run. The table compares the best model of each.

| Variant | Model | Valid ROC-AUC | Valid PR-AUC | Test ROC-AUC | Test PR-AUC | Test Brier | Test precision @ 5%, 10%, 20%, 30% |
|---|---|---|---|---|---|---|---|
| without_call | logreg | 0.7127 | 0.1978 | 0.6974 | 0.1485 | 0.0665 | 0.217 / 0.178 / 0.145 / 0.133 |
| with_call | logreg | 0.7142 | 0.2048 | 0.6987 | 0.1549 | 0.0665 | 0.215 / 0.185 / 0.147 / 0.131 |

- PR-AUC gain from the feature on validation: **+0.0070** (ROC-AUC +0.0015). On test: +0.0064.
- Timing confirmed: **False**. Minimum gain to bother: 0.005.
- **Decision: exclude the feature. Production model is `20261015T175504-logreg-without_call`.**
- Reason: The feature adds +0.0070 PR-AUC on validation, but we cannot confirm the call happens before scoring. A call recorded after abandonment could be part of the outcome itself, so the gain may not hold in production. Excluded until the timing is confirmed.

## All candidates

| Candidate | Valid PR-AUC | Test ROC-AUC | Test PR-AUC | Test Brier | Test log loss |
|---|---|---|---|---|---|
| logreg_without_call | 0.1978 | 0.6974 | 0.1485 | 0.0665 | 0.2486 |
| xgboost_without_call | 0.1959 | 0.6928 | 0.1461 | 0.0667 | 0.2493 |
| logreg_with_call | 0.2048 | 0.6987 | 0.1549 | 0.0665 | 0.2482 |
| xgboost_with_call | 0.1992 | 0.6979 | 0.1550 | 0.0665 | 0.2482 |

## Chosen model on test (`logreg_without_call`)

Base rate 0.074, mean predicted probability 0.093.

| Top k called | Leads | Precision | Recall | Lift |
|---|---|---|---|---|
| 5% | 469 | 0.217 | 0.147 | 2.95 |
| 10% | 938 | 0.178 | 0.241 | 2.41 |
| 20% | 1876 | 0.145 | 0.393 | 1.97 |
| 30% | 2814 | 0.133 | 0.542 | 1.81 |

### By segment

| segment      | value      |    n |   conversion |   mean_score |   roc_auc |   pr_auc |
|:-------------|:-----------|-----:|-------------:|-------------:|----------:|---------:|
| product_type | carbody    | 2575 |       0.0641 |       0.075  |    0.6934 |   0.1451 |
| product_type | thirdparty | 6805 |       0.0774 |       0.0999 |    0.697  |   0.1513 |
| channel      | CRM        | 1129 |       0.1027 |       0.1026 |    0.6922 |   0.1935 |
| channel      | Paid       | 3044 |       0.0631 |       0.086  |    0.6843 |   0.1222 |
| channel      | Referral   | 1662 |       0.0632 |       0.0868 |    0.7215 |   0.1521 |
| channel      | SEO        | 3545 |       0.0787 |       0.099  |    0.6908 |   0.1547 |

### Permutation importance (PR-AUC drop on test)

| feature                   |   pr_auc_drop |
|:--------------------------|--------------:|
| visited_offer_page        |        0.0221 |
| has_previous_purchase     |        0.0205 |
| days_to_policy_expiry     |        0.0181 |
| offer_views_last_7d       |        0.0067 |
| channel                   |        0.0046 |
| minutes_since_abandonment |        0.0039 |
| discount_percent          |        0.0033 |
| payment_type              |        0.0027 |
| sessions_last_7d          |        0.0014 |
| price_comparisons_last_7d |        0.0008 |
| device                    |        0.0006 |
| city                      |        0.0005 |
| insurance_company         |        0.0002 |
| days_since_last_visit     |        0      |
| product_type              |       -0.0001 |
| partner                   |       -0.0003 |
| price                     |       -0.0009 |

### Drift by month (PSI vs. training window, features above 0.1 only)

No feature above 0.1 in any month.

### Predicted vs. actual conversion by month

Train months are in-sample, July is the calibration month, August is test.

| month   |   leads |   actual |   predicted |
|:--------|--------:|---------:|------------:|
| 2026-04 |   10081 |   0.0976 |      0.0932 |
| 2026-05 |   10245 |   0.0972 |      0.0932 |
| 2026-06 |    9840 |   0.0949 |      0.0936 |
| 2026-07 |   10454 |   0.0936 |      0.0935 |
| 2026-08 |    9380 |   0.0738 |      0.0931 |
