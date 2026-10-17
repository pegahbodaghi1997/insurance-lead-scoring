CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS ml;

-- CSV as delivered. No primary key on purpose: the file has repeated lead ids
-- and cleaning is the pipeline's job, not the loader's.
CREATE TABLE IF NOT EXISTS raw.leads (
    lead_id                     TEXT NOT NULL,
    created_at                  TIMESTAMP NOT NULL,
    product_type                TEXT,
    channel                     TEXT,
    device                      TEXT,
    partner                     TEXT,
    city                        TEXT,
    insurance_company           TEXT,
    payment_type                TEXT,
    minutes_since_abandonment   INTEGER,
    days_to_policy_expiry       INTEGER,
    price                       NUMERIC(14, 2),
    discount_percent            NUMERIC(5, 2),
    has_previous_purchase       SMALLINT,
    visited_offer_page          SMALLINT,
    incoming_call_last_24h      SMALLINT,
    sessions_last_7d            INTEGER,
    offer_views_last_7d         INTEGER,
    price_comparisons_last_7d   INTEGER,
    days_since_last_visit       NUMERIC(6, 2),
    expected_margin             NUMERIC(14, 2),
    completed_purchase          SMALLINT,
    loaded_at                   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_raw_leads_lead_id ON raw.leads (lead_id);
CREATE INDEX IF NOT EXISTS ix_raw_leads_created_at ON raw.leads (created_at);

-- One row per trained model. Both incoming-call variants are registered,
-- only one of them is active.
CREATE TABLE IF NOT EXISTS ml.model_versions (
    model_version   TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    algorithm       TEXT NOT NULL,
    feature_set     TEXT NOT NULL,
    features        JSONB NOT NULL,
    params          JSONB,
    train_from      TIMESTAMP,
    train_to        TIMESTAMP,
    metrics         JSONB,
    decision_note   TEXT,
    artifact_path   TEXT NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT FALSE
);
-- at most one active model
CREATE UNIQUE INDEX IF NOT EXISTS ux_model_versions_active
    ON ml.model_versions (is_active) WHERE is_active;

CREATE TABLE IF NOT EXISTS ml.scoring_runs (
    run_id          UUID PRIMARY KEY,
    model_version   TEXT NOT NULL REFERENCES ml.model_versions (model_version),
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ,
    rank_by         TEXT NOT NULL,
    n_leads         INTEGER,
    mean_score      DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS ml.lead_scores (
    run_id          UUID NOT NULL REFERENCES ml.scoring_runs (run_id),
    lead_id         TEXT NOT NULL,
    model_version   TEXT NOT NULL REFERENCES ml.model_versions (model_version),
    probability     DOUBLE PRECISION NOT NULL CHECK (probability BETWEEN 0 AND 1),
    expected_value  NUMERIC(14, 2),
    priority_rank   INTEGER NOT NULL,
    priority_tier   TEXT NOT NULL CHECK (priority_tier IN ('P1', 'P2', 'P3')),
    scored_at       TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (run_id, lead_id)
);
CREATE INDEX IF NOT EXISTS ix_lead_scores_rank ON ml.lead_scores (run_id, priority_rank);
CREATE INDEX IF NOT EXISTS ix_lead_scores_lead ON ml.lead_scores (lead_id);

-- Drift and data quality numbers per scoring run (long format).
CREATE TABLE IF NOT EXISTS ml.monitoring_metrics (
    run_id      UUID NOT NULL REFERENCES ml.scoring_runs (run_id),
    metric      TEXT NOT NULL,
    feature     TEXT NOT NULL,
    value       DOUBLE PRECISION,
    status      TEXT NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, metric, feature)
);

-- What the sales team should query: the latest run, best leads first.
CREATE OR REPLACE VIEW ml.latest_lead_scores AS
SELECT s.lead_id,
       s.priority_rank,
       s.priority_tier,
       s.probability,
       s.expected_value,
       s.model_version,
       s.scored_at,
       s.run_id
FROM ml.lead_scores s
WHERE s.run_id = (SELECT run_id FROM ml.scoring_runs
                  WHERE finished_at IS NOT NULL
                  ORDER BY finished_at DESC LIMIT 1)
ORDER BY s.priority_rank;

-- Same batch ordered by expected margin instead of probability.
CREATE OR REPLACE VIEW ml.latest_lead_scores_by_value AS
SELECT l.*,
       RANK() OVER (ORDER BY l.expected_value DESC NULLS LAST) AS value_rank
FROM ml.latest_lead_scores l
ORDER BY value_rank;
