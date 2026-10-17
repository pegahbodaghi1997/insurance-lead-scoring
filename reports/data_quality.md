# Data quality

- Rows loaded: 50180, after cleaning: 50000
- Duplicate lead ids dropped: 180 (copies differ only in created_at, first event kept)
- Positive rate: 0.0917
- Out of range values set to null: none
- Rows with offer views but visited_offer_page = 0: 7554
- Missing values: {'city': 932, 'price': 928, 'discount_percent': 541}
- Share of leads with an already expired policy: 0.1427
- expected_margin / price: median 0.026, min 0.024, max 0.042

## Volume and conversion by month

| created_at   |   leads |   conversion |
|:-------------|--------:|-------------:|
| 2026-04      |   10081 |       0.0976 |
| 2026-05      |   10245 |       0.0972 |
| 2026-06      |    9840 |       0.0949 |
| 2026-07      |   10454 |       0.0936 |
| 2026-08      |    9380 |       0.0738 |

## Single-feature ROC-AUC (leakage screen)

| feature                   |    auc |
|:--------------------------|-------:|
| offer_views_last_7d       | 0.6186 |
| visited_offer_page        | 0.6086 |
| has_previous_purchase     | 0.5919 |
| days_to_policy_expiry     | 0.5794 |
| sessions_last_7d          | 0.5749 |
| minutes_since_abandonment | 0.5647 |
| channel                   | 0.5342 |
| price                     | 0.5316 |
| product_type              | 0.5314 |
| discount_percent          | 0.5307 |
| expected_margin           | 0.5306 |
| insurance_company         | 0.5294 |
| incoming_call_last_24h    | 0.5237 |
| payment_type              | 0.522  |
| days_since_last_visit     | 0.5147 |
| city                      | 0.5131 |
| device                    | 0.5112 |
| partner                   | 0.5074 |
| price_comparisons_last_7d | 0.5024 |

Features above 0.75: none
