"""Score leads with the active model and write the ranked list to Postgres."""
import logging
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import clean_leads, load_raw_leads
from .db import execute, read_sql
from .model import LeadScoringModel
from .monitoring import monitor_latest_run

log = logging.getLogger(__name__)


def active_model(engine):
    row = read_sql(engine, """SELECT model_version, artifact_path FROM ml.model_versions
                              WHERE is_active""")
    if row.empty:
        raise RuntimeError("no active model in ml.model_versions, run `train` first")
    model = LeadScoringModel.load(row.artifact_path[0])
    assert model.model_version == row.model_version[0], "artifact does not match registry"
    return model


def assign_priority(scores, rank_by, tiers):
    """Rank within the batch (1 = call first) and cut into P1/P2/P3 by share of the batch."""
    scores = scores.copy()
    key = scores[rank_by].fillna(-1)
    # ties broken by probability, then lead_id, so the order is reproducible
    order = np.lexsort((scores["lead_id"].to_numpy(), -scores["probability"].to_numpy(), -key.to_numpy()))
    rank = np.empty(len(scores), dtype=int)
    rank[order] = np.arange(1, len(scores) + 1)
    scores["priority_rank"] = rank

    share = scores["priority_rank"] / len(scores)
    scores["priority_tier"] = np.where(share <= tiers["P1"], "P1",
                                       np.where(share <= tiers["P2"], "P2", "P3"))
    return scores


def run_scoring(engine, cfg):
    sc = cfg["scoring"]
    model = active_model(engine)
    df, report = clean_leads(load_raw_leads(engine, cfg["data"]["raw_table"], sc["created_from"]))
    if df.empty:
        log.warning("nothing to score")
        return None

    run_id = uuid.uuid4()
    started = datetime.now(timezone.utc)
    execute(engine, """INSERT INTO ml.scoring_runs (run_id, model_version, started_at, rank_by)
                       VALUES (:r, :mv, :s, :rb)""",
            r=str(run_id), mv=model.model_version, s=started, rb=sc["rank_by"])

    p = model.predict_proba(df)
    scores = pd.DataFrame({
        "lead_id": df["lead_id"].to_numpy(),
        "probability": p,
        # value if converted x probability; margin is known at scoring time
        "expected_value": (p * df["expected_margin"].to_numpy()).round(2),
    })
    scores = assign_priority(scores, sc["rank_by"], sc["tiers"])
    scores["run_id"] = str(run_id)
    scores["model_version"] = model.model_version
    scores["scored_at"] = started

    with engine.begin() as conn:
        scores.to_sql("lead_scores", conn, schema="ml", if_exists="append", index=False,
                      method="multi", chunksize=5000)
    execute(engine, """UPDATE ml.scoring_runs SET finished_at = :f, n_leads = :n, mean_score = :m
                       WHERE run_id = :r""",
            f=datetime.now(timezone.utc), n=len(scores), m=float(p.mean()), r=str(run_id))

    tiers = scores.groupby("priority_tier")["probability"].agg(["size", "mean"])
    log.info("scored %d leads with %s (run %s)", len(scores), model.model_version, run_id)
    log.info("tiers:\n%s", tiers.round(4).to_string())

    monitor_latest_run(engine, cfg, scored_df=df.assign(probability=p), model=model, run_id=run_id)
    return run_id
