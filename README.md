# Lead scoring for abandoned insurance purchases

Every day a lot of users start buying insurance, see a few offers and leave. Sales can only call some of them back.
This project estimates, for every lead, the probability that the purchase is completed and writes a ranked call list
into Postgres so the sales team can take leads from the top.

All of it runs in Docker: loading the CSV into Postgres, cleaning, training, evaluation, scoring and a basic
monitoring step.

## Running it

You only need Docker (with compose v2). Nothing has to be installed on the host.

```bash
make all        # build, start Postgres, run the whole pipeline
make test       # unit tests
```

If `make` isn't available, these are the same thing:

```bash
docker compose build
docker compose up -d db
docker compose run --rm app all
docker compose run --rm --no-deps --entrypoint pytest app -q tests
```

A full run takes about a minute. Each step can also be run on its own:

| Command | What it does |
| --- | --- |
| `docker compose run --rm app ingest` | creates the schemas and loads `data/leads.csv` into `raw.leads` (truncate + reload, so it can be re-run) |
| `docker compose run --rm app eda` | data quality report (`reports/data_quality.md`) and EDA charts |
| `docker compose run --rm app train` | trains both feature-set variants, makes the incoming-call decision, registers models in `ml.model_versions` |
| `docker compose run --rm app evaluate` | test-month evaluation of all candidates, charts, `reports/evaluation.md` |
| `docker compose run --rm app predict` | scores leads with the active model into `ml.lead_scores`, then runs the monitoring checks |
| `docker compose run --rm app monitor` | re-runs drift / data quality checks for the latest scoring run |

`all` runs ingest → eda → train → evaluate → predict. The pipeline only reads the CSV in `ingest`. Every later step
reads from Postgres.

Postgres is published on `localhost:5433` (user / password / db: `leads`), so you can connect with any client.
`make psql` opens a shell inside the container. Settings live in `config/config.yaml`. The DB credentials can be
overridden with a `.env` file (see `.env.example`).

To start from zero: `make clean` drops the database volume and the trained models.

## Layout

```text
config/config.yaml      split dates, feature lists, search space, priority tiers, monitoring thresholds
sql/schema.sql          raw + ml schemas, tables and the views sales would query
data/                   leads.csv and the data dictionary
src/leadscore/
  ingest.py             CSV -> raw.leads
  data.py               read from Postgres, cleaning, time split
  features.py           LeadFeatureBuilder (sklearn transformer) + encoders
  model.py              logistic regression / XGBoost fitting, calibration, model wrapper
  train.py              trains both variants, incoming-call decision, model registry
  evaluate.py           test metrics, segment metrics, permutation importance, charts, report
  predict.py            scoring, ranking, tiers -> ml.lead_scores
  monitoring.py         PSI drift + null-rate / unseen-category checks
  eda.py                data quality report and EDA charts
  cli.py                `python -m leadscore <command>`
tests/                  unit tests (no DB needed)
charts/                 every chart, regenerated on each run
reports/                data_quality.md / evaluation.md (+ json versions)
artifacts/models/       pickled models per training run (git-ignored)
```

## What the data looks like and what I did about it

50,180 rows, 22 columns, April to August 2026. Target rate is 9.2%. The full report is in `reports/data_quality.md`.
The points that changed decisions:

- **Duplicates.** 180 lead ids appear twice. The two copies are identical except `created_at`, which differs by a
  few minutes. That looks like a resubmitted form, not two leads. I keep the first event, which leaves 50,000 rows.
  The raw table keeps both copies. Cleaning happens in the pipeline, so the raw data stays as delivered.
- **Missing values.** `city` (932), `price` (928), `discount_percent` (541), all under 2%. City becomes an `Unknown`
  category. Price is imputed with the median of its product type, because carbody and third-party prices are very
  different. Discount is imputed with the median. Both numeric columns also get a missing flag. The missing rows
  convert at about the same rate as the rest, so there is no strong pattern behind the missingness.
- **Ranges / outliers.** Nothing impossible. `days_to_policy_expiry` is negative for 14% of leads. That is not an
  error: it means the current policy has already expired, and that group converts best (13%). Price is
  right-skewed (1.5M to 40M) but plausible, so I use log(price) instead of clipping. The cleaning step still
  checks ranges and nulls anything outside them, so bad values in future batches don't reach the model.
- **An inconsistency.** 7,581 leads have `visited_offer_page = 0` but `offer_views_last_7d > 0`. These two fields
  probably cover different windows (current session vs. 7 days). Both carry signal, so both stay, but this is
  something to ask the data owner about.
- **Conversion drops in August.** It sits around 9.5% from April to July, then falls to 7.4% in August. The drop
  is flat across all four August weeks, so it is not the usual "recent leads haven't had time to convert" pattern
  (that would hit the last week hardest). None of the input distributions move either: monthly PSI is below 0.01
  for every feature. So the outcome shifted and the inputs didn't. This is why the split is by time, and it
  shows up again in calibration (below).
- **Leakage screen.** I computed the ROC-AUC of each column on its own (`charts/eda_univariate_auc.png`). The
  best single column is around 0.62. Nothing looks like the target leaking in directly.

## Which columns go into the model

| Column | Used? | Why |
| --- | --- | --- |
| `lead_id` | no | identifier |
| `created_at` | only hour and weekday | also drives the split |
| `expected_margin` | no | it is ~2.6% of price (ratio 0.024–0.042), so it adds nothing beyond price. It is used for expected value in the output instead |
| `incoming_call_last_24h` | **no** (see next section) | the timing is unclear, so it is a leakage risk |
| `completed_purchase` | target | |
| everything else | yes | |

Derived features: log price, log net price (after discount), expired-policy flag, offer views per session,
comparisons per session, missing flags, hour, weekday. All of it lives in `LeadFeatureBuilder`. It is fitted
on the training window only and pickled with the model, so scoring uses the same imputation values. The
builder takes a whitelist of raw columns, and a unit test checks that changing anything outside it (target,
margin, the call flag) doesn't change the features.

## Incoming Call Last 24h: trained both ways, excluded

The data dictionary describes this column as "an incoming call related to the user in the last 24 hours", but
doesn't say what the 24 hours are measured from. If it was captured when the user abandoned the flow, it is a
fair intent signal. If it was captured later, close to when the outcome was recorded, the call may be the user
phoning in to finish the purchase. In that case it is part of the outcome, and the model would look better
offline than it will be in production.

Instead of guessing, every training run builds both variants with the same procedure and compares them on the
validation month. The decision rule is in `train.decide_incoming_call` and is configured in
`config.yaml → leakage_review`:

- if the timing is **not confirmed** (current state), the feature is excluded. The reason text depends on whether
  the gain was negligible or material.
- if the timing is confirmed **and** the validation PR-AUC gain is at least 0.005, it is included.

Results from the last run (best model per variant, both are logistic regression):

| Variant | Valid ROC-AUC | Valid PR-AUC | Test ROC-AUC | Test PR-AUC | Test precision@10% |
| --- | --- | --- | --- | --- | --- |
| without call | 0.7127 | 0.1978 | 0.6974 | 0.1485 | 0.178 |
| with call | 0.7142 | 0.2048 | 0.6987 | 0.1549 | 0.185 |

The feature adds +0.007 PR-AUC on validation and +0.006 on test. Precision at the top 10% goes up by less than
one percentage point. That's above the 0.005 threshold, so it isn't nothing. But the gain is small, and the
downside if it is a leak (a model that quietly gets worse in production) is bigger. Only 8% of leads have a call
at all. There is also a business angle: someone who called us in the last 24 hours is probably already talking
to sales, so pushing them up the call list may not be the best use of an agent's time anyway.

**Decision: the production model is the one without the feature.** Both models are registered in
`ml.model_versions`. The rejected one has `is_active = false` and the reason in `decision_note`. Once someone
confirms the timing, switching is a config change (`timing_confirmed: true`) and a retrain. The comparison is in
`reports/evaluation.md` and `charts/eval_incoming_call_comparison.png`.

## Modeling

**Split.** Out-of-time: train April–June (30,166), validation July (10,454), test August (9,380). The model will
always score leads that come after its training data, and a random split would hide the August shift. Validation
is used for early stopping, model selection, calibration and the incoming-call decision. Test is touched only by
`evaluate`.

**Models.** I trained two, on purpose:

- logistic regression (one-hot + standardized numerics, C from a small grid) as a baseline that is easy to
  explain
- XGBoost with a 15-trial random search and early stopping on validation PR-AUC. No heavy tuning.

Logistic regression won on validation in both variants (0.1978 vs 0.1959 without the call feature). XGBoost
didn't find much to add: its best trials were depth-2 trees, which says the signal is mostly additive. With the
scores this close, the simpler model wins: it is easier to explain to sales, more stable, and cheaper to run. The
active model is therefore logistic regression.

**Class imbalance (9%).** No resampling or class weights. SMOTE or reweighting would distort the probabilities,
and the output is supposed to be a probability that the sales team can read as one. Imbalance is handled on the
evaluation side instead: PR-AUC and top-k metrics rather than accuracy.

**Calibration.** Platt scaling fitted on the validation month. I chose it over isotonic because it is monotonic
and smooth, so the ranking is exactly the same as the raw model. Isotonic would create ties at the top of the
list, which is where ranking matters most.

**Metrics.** Selection uses PR-AUC on validation (more informative than ROC-AUC with a 9% base rate). The
evaluation also reports ROC-AUC, Brier score, log loss, and precision / recall / lift when calling the top
5/10/20/30% of leads. The last group is what sales actually experiences, given their limited capacity. It also
reports calibration by decile, metrics per product type and channel, and permutation importance.

## Results (test month, August)

| Top k called | Leads | Precision | Recall | Lift |
| --- | --- | --- | --- | --- |
| 5% | 469 | 0.217 | 0.147 | 2.95 |
| 10% | 938 | 0.178 | 0.241 | 2.41 |
| 20% | 1,876 | 0.145 | 0.393 | 1.97 |
| 30% | 2,814 | 0.133 | 0.542 | 1.81 |

ROC-AUC 0.697, PR-AUC 0.149 (base rate 7.4%). If sales calls the top 20% of leads, they reach about 39% of all
purchases, and those calls convert at about twice the average rate. The signal in this data is moderate. No
single column is strong, and the important ones are `visited_offer_page`, `has_previous_purchase` and
`days_to_policy_expiry`, then offer views and channel. Performance is similar across product types and channels
(ROC-AUC 0.68–0.72 in every segment).

**Calibration on test is off, and that is expected.** The model predicts about 9.3% for August, but the real
rate is 7.4% (`reports/evaluation.md`, "Predicted vs. actual"). The ranking holds (ROC-AUC 0.71 on validation,
0.70 on test), but the absolute probabilities are too high for August. That follows from the shift described
above: the inputs look the same, so a model can't see the change. The practical fix is to recalibrate on the most
recent labelled month before each scoring cycle. I left that out to keep the test month clean.

## Output in Postgres

```text
raw.leads                  CSV as delivered
ml.model_versions          every registered model: features, params, train window, valid + test metrics,
                           decision note, artifact path, is_active (at most one active, enforced by an index)
ml.scoring_runs            one row per scoring batch: model version, time, rank_by, row count, mean score
ml.lead_scores             one row per (run, lead): probability, expected_value, priority_rank, priority_tier,
                           model_version, scored_at
ml.monitoring_metrics      PSI / null-rate / unseen-category checks per run with a status
ml.latest_lead_scores      view: latest finished run, ordered by priority_rank
ml.latest_lead_scores_by_value   same batch ranked by expected value
```

`priority_rank` is 1 for the first lead to call. Tiers are cut by share of the batch: P1 = top 10%, P2 = next 20%,
P3 = the rest (configurable). In the last run, P1 leads average 24% predicted probability, P2 14% and P3 6%.

Ranking is by probability. It maximises purchases per call and matches what the task asks for.
`expected_value = probability × expected_margin` is stored too, and the `_by_value` view sorts by it. Margin
varies about 14x between the 1st and 99th percentile, while probability varies much less, so ranking by expected
value mostly means ranking by price. Whether sales should chase the most purchases or the most margin is a business decision, so
both are available. `rank_by` in the config switches the main ranking.

```sql
-- today's call list for the team
SELECT lead_id, priority_rank, priority_tier, probability
FROM ml.latest_lead_scores
WHERE priority_tier = 'P1';

-- which model produced it and how it did
SELECT model_version, feature_set, metrics->'test'->>'pr_auc' AS test_pr_auc, decision_note
FROM ml.model_versions WHERE is_active;
```

In this project every lead in the table gets scored, because the dataset has no notion of "still open".
`scoring.created_from` in the config limits the batch to recent leads. In production the batch would be the
open, not-yet-contacted leads of the day.

## Monitoring

After each scoring run, `predict` compares the batch with the training window and writes the results to
`ml.monitoring_metrics`:

- PSI per input feature and for the score (decile bins for numerics, one bin per value for binary columns,
  shares per category). The status is `warn` at 0.1 and `alert` at 0.25.
- null rate per column vs. training: warn if it doubles and is above 1%
- share of categories never seen in training

Warnings are also written to the log. Charts: `monitoring_psi_latest_run.png` and `monitoring_psi_by_month.png`.

This catches broken inputs and population shifts. As the August data shows, it does not catch everything: the
inputs were stable while conversion dropped 2 points. The monitoring that matters most needs labels, which arrive
with a delay. That means tracking actual conversion per tier and per score decile, and precision@k against the
values above. That needs an outcomes feed this dataset doesn't have, so it is in the next-steps list.

## Assumptions

- Each row is a snapshot taken when the user abandoned the flow, and all behaviour columns (`*_last_7d`,
  `days_since_last_visit`, `minutes_since_abandonment`) are known at that point. The incoming call flag is the one
  exception I treat as uncertain.
- `expected_margin` is known before the purchase (it comes from the quote). That's why it can be used for
  expected value at scoring time.
- Duplicated lead ids are resubmissions of the same lead.
- All leads have the same calling cost, so ranking by probability (or expected value) is the right objective.

## Limitations

- The model predicts "will buy", not "will buy *because we called*". Some top-ranked leads would buy anyway. The
  right target for a call list is uplift, but the data has no record of who was actually called, so it can't be
  estimated here.
- The August shift isn't explained. Calibration on new data is off until it is refreshed with recent labels.
- The validation month is used for early stopping, selection and calibration together. With 10k rows and a
  2-parameter calibrator the risk is small, but the test month is the only fully clean estimate.
- Tiers are relative to the batch. A weak day still gets 10% P1 leads. Absolute probability thresholds would need
  reliable calibration first.
- The scoring batch is the whole table, not a real daily queue.

## With more time

1. Confirm the timing of `incoming_call_last_24h` (and the 7-day windows) with whoever owns the event data. If it
   is safe, flip `timing_confirmed`.
2. Recalibrate on the latest labelled weeks before each scoring cycle, and retrain monthly on a rolling window.
3. Collect call outcomes (who was called, when, result) and move to an uplift model, or at least an A/B holdout
   group, to measure what calling actually adds.
4. Label-based monitoring: actual conversion per tier and per decile once outcomes arrive, with alerts.
5. Move scheduling to an orchestrator (Airflow / cron), keep model artifacts in object storage or MLflow instead
   of a local folder, and use Alembic migrations instead of a single schema file.
6. Rolling-origin cross-validation (several train/validation cutoffs) to get a variance estimate on the metrics
   instead of a single test month.
