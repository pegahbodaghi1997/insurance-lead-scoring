"""Drift (PSI) and basic data quality checks for scoring batches.

The training window is the reference. For numeric columns we store decile edges
and the share of rows in each bin, for categoricals the share per category.
At scoring time the same bins are applied to the new batch and compared.
"""
import logging

import numpy as np
import pandas as pd

from .data import CATEGORICAL
from .db import read_sql
from .model import LeadScoringModel
from .plots import BLUE, GRAY, ORANGE, plt, save

log = logging.getLogger(__name__)

EPS = 1e-4


def _numeric_bins(values, n_bins):
    uniq = np.unique(values[~np.isnan(values)])
    if len(uniq) <= n_bins:
        # binary / low-cardinality columns: one bin per value, quantiles would collapse
        edges = np.concatenate([[-np.inf], (uniq[:-1] + uniq[1:]) / 2, [np.inf]])
        return edges
    edges = np.unique(np.nanquantile(values, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _shares_numeric(values, edges):
    values = pd.Series(values, dtype=float)
    counts = pd.cut(values, edges).value_counts(sort=False).to_numpy()
    shares = list(counts / max(len(values), 1))
    shares.append(float(values.isna().mean()))  # missing gets its own bucket
    return shares


def _shares_categorical(values, categories):
    values = pd.Series(values).astype(object).where(pd.Series(values).notna(), "__missing__")
    freq = values.value_counts(normalize=True)
    shares = [float(freq.get(c, 0.0)) for c in categories]
    shares.append(float(1 - freq.reindex(categories).fillna(0).sum()))  # unseen
    return shares


def build_reference(df, columns, scores, n_bins):
    ref = {"columns": {}, "null_rate": {}}
    for col in columns:
        if col in CATEGORICAL:
            cats = sorted(df[col].astype(object).where(df[col].notna(), "__missing__").unique())
            ref["columns"][col] = {"type": "cat", "categories": cats,
                                   "shares": _shares_categorical(df[col], cats)}
        else:
            edges = _numeric_bins(df[col].to_numpy(dtype=float), n_bins)
            ref["columns"][col] = {"type": "num", "edges": edges.tolist(),
                                   "shares": _shares_numeric(df[col], edges)}
        ref["null_rate"][col] = float(df[col].isna().mean())
    edges = _numeric_bins(scores, n_bins)
    ref["score"] = {"type": "num", "edges": edges.tolist(), "shares": _shares_numeric(scores, edges)}
    return ref


def psi(expected, actual):
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def batch_shares(spec, values):
    if spec["type"] == "cat":
        return _shares_categorical(values, spec["categories"])
    return _shares_numeric(values, np.asarray(spec["edges"]))


def psi_status(value, warn, alert):
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warn"
    return "ok"


def drift_table(reference, df, scores, cfg):
    warn, alert = cfg["monitoring"]["psi_warn"], cfg["monitoring"]["psi_alert"]
    rows = []
    for col, spec in reference["columns"].items():
        v = psi(spec["shares"], batch_shares(spec, df[col]))
        rows.append({"metric": "psi", "feature": col, "value": v, "status": psi_status(v, warn, alert)})
    v = psi(reference["score"]["shares"], batch_shares(reference["score"], scores))
    rows.append({"metric": "psi", "feature": "score", "value": v, "status": psi_status(v, warn, alert)})

    # null rate jump vs. training: flag if it doubles and is above 1%
    for col, ref_rate in reference["null_rate"].items():
        rate = float(df[col].isna().mean())
        bad = rate > 0.01 and rate > 2 * ref_rate
        rows.append({"metric": "null_rate", "feature": col, "value": rate,
                     "status": "warn" if bad else "ok"})

    # categories never seen during training
    for col, spec in reference["columns"].items():
        if spec["type"] == "cat":
            unseen = ~df[col].astype(object).where(df[col].notna(), "__missing__").isin(spec["categories"])
            rows.append({"metric": "unseen_category_rate", "feature": col,
                         "value": float(unseen.mean()),
                         "status": "warn" if unseen.mean() > 0.01 else "ok"})
    return pd.DataFrame(rows)


def write_monitoring(engine, run_id, table):
    out = table.assign(run_id=str(run_id))
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM ml.monitoring_metrics WHERE run_id = %s", (str(run_id),))
        out[["run_id", "metric", "feature", "value", "status"]].to_sql(
            "monitoring_metrics", conn, schema="ml", if_exists="append", index=False)


def monitor_latest_run(engine, cfg, scored_df=None, model=None, run_id=None):
    """Drift report for a scoring run. Called right after scoring, or on its own."""
    if run_id is None:
        latest = read_sql(engine, """
            SELECT r.run_id, r.model_version, m.artifact_path
            FROM ml.scoring_runs r JOIN ml.model_versions m USING (model_version)
            WHERE r.finished_at IS NOT NULL ORDER BY r.finished_at DESC LIMIT 1""")
        if latest.empty:
            raise RuntimeError("no finished scoring run found, run `predict` first")
        run_id = latest.run_id[0]
        model = LeadScoringModel.load(latest.artifact_path[0])
    if scored_df is None:
        from .data import clean_leads, load_raw_leads
        scored_df, _ = clean_leads(load_raw_leads(engine))
        scores = read_sql(engine, "SELECT lead_id, probability FROM ml.lead_scores WHERE run_id = :r",
                          r=str(run_id))
        scored_df = scored_df.merge(scores, on="lead_id")
    table = drift_table(model.reference, scored_df, scored_df["probability"].to_numpy(), cfg)
    write_monitoring(engine, run_id, table)

    psi_rows = table[table.metric == "psi"].sort_values("value")
    _plot_psi(psi_rows, cfg, cfg["paths"]["charts"] / "monitoring_psi_latest_run.png",
              "PSI vs training window, latest scoring run")

    flagged = table[table.status != "ok"]
    for _, r in flagged.iterrows():
        log.warning("monitoring %s: %s %s=%.3f", r.status, r.feature, r.metric, r.value)
    log.info("monitoring: %d checks, %d flagged", len(table), len(flagged))
    return table


def monthly_drift(df, model, cfg):
    """PSI of each month vs. the training reference. Shows how the data moves over time."""
    rows = []
    for month, part in df.groupby(df["created_at"].dt.to_period("M")):
        scores = model.predict_proba(part)
        t = drift_table(model.reference, part, scores, cfg)
        t = t[t.metric == "psi"].assign(month=str(month))
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def _plot_psi(rows, cfg, path, title):
    warn, alert = cfg["monitoring"]["psi_warn"], cfg["monitoring"]["psi_alert"]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(rows) + 1.2))
    colors = [ORANGE if v >= warn else BLUE for v in rows["value"]]
    ax.barh(rows["feature"], rows["value"], color=colors, height=0.6)
    ax.axvline(warn, color=GRAY, linestyle="--", linewidth=1)
    ax.text(warn, len(rows) - 0.4, f" warn {warn}", color=GRAY, fontsize=8)
    if rows["value"].max() > alert * 0.6:
        ax.axvline(alert, color=GRAY, linestyle=":", linewidth=1)
    ax.set_xlabel("PSI")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    save(fig, path)
