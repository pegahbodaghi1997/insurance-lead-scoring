"""Read leads from Postgres, clean them and split by time."""
import logging

import numpy as np
import pandas as pd

from .db import read_sql

log = logging.getLogger(__name__)

CATEGORICAL = ["product_type", "channel", "device", "partner", "city",
               "insurance_company", "payment_type"]
BINARY = ["has_previous_purchase", "visited_offer_page", "incoming_call_last_24h"]

# Valid ranges for numeric columns. Values outside are set to NaN and counted.
# days_to_policy_expiry can be negative: that means the current policy has
# already expired, which is a real (and high-converting) situation.
VALID_RANGES = {
    "minutes_since_abandonment": (0, None),
    "days_to_policy_expiry": (-365, 365),
    "price": (1, None),
    "discount_percent": (0, 100),
    "sessions_last_7d": (0, None),
    "offer_views_last_7d": (0, None),
    "price_comparisons_last_7d": (0, None),
    "days_since_last_visit": (0, None),
    "expected_margin": (0, None),
}


def load_raw_leads(engine, table="raw.leads", created_from=None):
    query = f"SELECT * FROM {table}"
    params = {}
    if created_from:
        query += " WHERE created_at >= :created_from"
        params["created_from"] = created_from
    df = read_sql(engine, query, **params)
    return df.drop(columns=["loaded_at"], errors="ignore")


def clean_leads(df):
    """Returns (clean_df, report). The report is a dict of what was changed."""
    df = df.copy()
    report = {"rows_in": int(len(df))}

    # 180 lead ids appear twice; the copies differ only in created_at by a few
    # minutes, which looks like a resubmitted form. Keep the first event.
    df = df.sort_values(["lead_id", "created_at"])
    dupes = df.duplicated("lead_id", keep="first")
    report["duplicate_lead_ids_dropped"] = int(dupes.sum())
    df = df[~dupes]

    for col in CATEGORICAL:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    numeric_cols = list(VALID_RANGES) + BINARY
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    out_of_range = {}
    for col, (lo, hi) in VALID_RANGES.items():
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= df[col] < lo
        if hi is not None:
            bad |= df[col] > hi
        out_of_range[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    for col in BINARY:
        bad = df[col].notna() & ~df[col].isin([0, 1])
        out_of_range[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    report["out_of_range_set_to_null"] = {k: v for k, v in out_of_range.items() if v}

    # visited_offer_page=0 with offer views > 0 would be contradictory
    report["offer_views_without_offer_page"] = int(
        ((df["visited_offer_page"] == 0) & (df["offer_views_last_7d"] > 0)).sum()
    )

    report["missing_values"] = {c: int(n) for c, n in df.isna().sum().items() if n}
    report["rows_out"] = int(len(df))
    if "completed_purchase" in df:
        report["positive_rate"] = round(float(df["completed_purchase"].mean()), 4)

    log.info("cleaning: %d -> %d rows", report["rows_in"], report["rows_out"])
    return df.sort_values("created_at").reset_index(drop=True), report


def time_split(df, train_end, valid_end):
    train_end, valid_end = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    train = df[df["created_at"] < train_end]
    valid = df[(df["created_at"] >= train_end) & (df["created_at"] < valid_end)]
    test = df[df["created_at"] >= valid_end]
    log.info("split sizes train=%d valid=%d test=%d", len(train), len(valid), len(test))
    return train, valid, test
