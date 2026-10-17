"""Data quality report and a handful of EDA charts.

Not meant to be exhaustive - the goal is to surface the things that change
decisions later (duplicates, missing values, label drift over time, features
that look too good).
"""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import BINARY, CATEGORICAL, clean_leads, load_raw_leads
from .plots import BLUE, GRAY, ORANGE, TEXT_2, pct_axis, plt, save

log = logging.getLogger(__name__)

NUMERIC = ["minutes_since_abandonment", "days_to_policy_expiry", "price", "discount_percent",
           "sessions_last_7d", "offer_views_last_7d", "price_comparisons_last_7d",
           "days_since_last_visit", "expected_margin"]


def _with_missing(s):
    return s.astype(object).where(s.notna(), "missing")


def univariate_auc(df, target):
    """ROC-AUC of each column on its own. Anything close to 1 is suspicious."""
    y = df[target]
    rows = []
    for col in CATEGORICAL + BINARY + NUMERIC:
        x = df[col]
        if col in CATEGORICAL:
            x = _with_missing(x)
            x = x.map(y.groupby(x).mean())
        else:
            x = x.fillna(x.median())
        auc = roc_auc_score(y, x)
        rows.append({"feature": col, "auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def run_eda(engine, cfg):
    target = cfg["target"]
    charts, reports = cfg["paths"]["charts"], cfg["paths"]["reports"]
    raw = load_raw_leads(engine, cfg["data"]["raw_table"])
    df, report = clean_leads(raw)

    monthly = (df.groupby(df["created_at"].dt.to_period("M"))[target]
               .agg(leads="size", conversion="mean").reset_index())
    monthly["created_at"] = monthly["created_at"].astype(str)
    report["monthly"] = monthly.round(4).to_dict("records")

    report["expected_margin_to_price"] = (df["expected_margin"] / df["price"]).describe().round(4).to_dict()
    report["expired_policy_share"] = round(float((df["days_to_policy_expiry"] < 0).mean()), 4)

    auc = univariate_auc(df, target)
    report["univariate_auc"] = auc.round(4).to_dict("records")
    suspicious = auc[auc.auc > 0.75]
    report["suspicious_features"] = suspicious.feature.tolist()

    rates = {}
    for col in ["incoming_call_last_24h", "visited_offer_page", "has_previous_purchase"]:
        rates[col] = df.groupby(col)[target].agg(["size", "mean"]).round(4).to_dict("index")
    report["binary_conversion"] = {k: {str(int(i)): v for i, v in d.items()} for k, d in rates.items()}

    with open(reports / "data_quality.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    (reports / "data_quality.md").write_text(_markdown(report, monthly, auc))

    _plot_monthly(monthly, charts)
    _plot_missing(raw, charts)
    _plot_categorical(df, target, charts)
    _plot_numeric(df, target, charts)
    _plot_univariate_auc(auc, charts)
    log.info("eda done, %d charts in %s", 5, charts)
    return report


def _markdown(report, monthly, auc):
    lines = ["# Data quality", "",
             f"- Rows loaded: {report['rows_in']}, after cleaning: {report['rows_out']}",
             f"- Duplicate lead ids dropped: {report['duplicate_lead_ids_dropped']} "
             "(copies differ only in created_at, first event kept)",
             f"- Positive rate: {report['positive_rate']}",
             f"- Out of range values set to null: {report['out_of_range_set_to_null'] or 'none'}",
             f"- Rows with offer views but visited_offer_page = 0: {report['offer_views_without_offer_page']}",
             f"- Missing values: {report['missing_values']}",
             f"- Share of leads with an already expired policy: {report['expired_policy_share']}",
             f"- expected_margin / price: median {report['expected_margin_to_price']['50%']}, "
             f"min {report['expected_margin_to_price']['min']}, max {report['expected_margin_to_price']['max']}",
             "", "## Volume and conversion by month", "",
             monthly.round(4).to_markdown(index=False), "",
             "## Single-feature ROC-AUC (leakage screen)", "",
             auc.round(4).to_markdown(index=False), "",
             f"Features above 0.75: {report['suspicious_features'] or 'none'}", ""]
    return "\n".join(lines)


def _plot_monthly(monthly, charts):
    # two panels instead of a dual axis
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.bar(monthly["created_at"], monthly["leads"], color=BLUE, width=0.6)
    a1.set(title="Leads per month", ylabel="Leads")
    a1.grid(axis="x", visible=False)
    a2.plot(monthly["created_at"], monthly["conversion"], color=ORANGE, marker="o", markersize=6)
    for x, v in zip(monthly["created_at"], monthly["conversion"]):
        a2.annotate(f"{v:.1%}", (x, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=TEXT_2)
    a2.set(title="Conversion rate per month", ylabel="Completed purchase")
    a2.set_ylim(0, monthly["conversion"].max() * 1.3)
    pct_axis(a2)
    save(fig, charts / "eda_monthly_volume_conversion.png")


def _plot_missing(raw, charts):
    miss = raw.isna().mean().sort_values()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(6, 0.4 * max(len(miss), 1) + 1.2))
    ax.barh(miss.index, miss.values, color=BLUE, height=0.5)
    for i, v in enumerate(miss.values):
        ax.text(v, i, f" {v:.2%}", va="center", fontsize=8, color=TEXT_2)
    ax.set(title="Share of missing values (raw table)", xlabel="Missing")
    pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    save(fig, charts / "eda_missing_values.png")


def _plot_categorical(df, target, charts):
    cols = CATEGORICAL + BINARY
    base = df[target].mean()
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    for ax, col in zip(axes.flat, cols):
        key = _with_missing(df[col])
        if col in BINARY:
            key = key.map({0.0: "no", 1.0: "yes"}).fillna("missing")
        g = df.groupby(key)[target].agg(["mean", "size"])
        g = g.sort_values("mean")
        ax.barh(g.index.astype(str), g["mean"], color=BLUE, height=0.6)
        ax.axvline(base, color=GRAY, linestyle="--", linewidth=1)
        ax.set_title(col)
        pct_axis(ax, "x")
        ax.grid(axis="y", visible=False)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.suptitle("Conversion rate by category (dashed line = overall rate)", fontsize=11)
    fig.tight_layout()
    save(fig, charts / "eda_conversion_by_category.png")


def _plot_numeric(df, target, charts):
    ncols = 3
    nrows = int(np.ceil(len(NUMERIC) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    base = df[target].mean()
    for ax, col in zip(axes.flat, NUMERIC):
        x = df[col]
        bins = pd.qcut(x, 10, duplicates="drop")
        g = df.groupby(bins, observed=True)[target].mean()
        mids = [iv.mid for iv in g.index]
        ax.plot(mids, g.values, color=BLUE, marker="o", markersize=4)
        ax.axhline(base, color=GRAY, linestyle="--", linewidth=1)
        ax.set_title(col)
        pct_axis(ax)
    for ax in list(axes.flat)[len(NUMERIC):]:
        ax.axis("off")
    fig.suptitle("Conversion rate by decile of each numeric column", fontsize=11)
    fig.tight_layout()
    save(fig, charts / "eda_conversion_by_numeric_decile.png")


def _plot_univariate_auc(auc, charts):
    auc = auc.sort_values("auc")
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(auc) + 1.2))
    colors = [ORANGE if a > 0.75 else BLUE for a in auc["auc"]]
    ax.barh(auc["feature"], auc["auc"], color=colors, height=0.6)
    ax.axvline(0.5, color=GRAY, linestyle="--", linewidth=1)
    ax.set_xlim(0.45, max(0.8, auc["auc"].max() + 0.02))
    ax.set(title="Single-feature ROC-AUC (leakage screen)", xlabel="ROC-AUC")
    ax.grid(axis="y", visible=False)
    save(fig, charts / "eda_univariate_auc.png")
