import numpy as np
import pandas as pd

from leadscore.data import clean_leads, time_split


def test_duplicate_lead_keeps_first_event(leads):
    dup = leads.iloc[[0]].copy()
    dup["created_at"] = dup["created_at"] - pd.Timedelta(minutes=5)
    df = pd.concat([leads, dup], ignore_index=True)

    clean, report = clean_leads(df)

    assert report["duplicate_lead_ids_dropped"] == 1
    assert clean["lead_id"].is_unique
    kept = clean.loc[clean.lead_id == leads.lead_id[0], "created_at"].iloc[0]
    assert kept == dup["created_at"].iloc[0]


def test_out_of_range_becomes_null_but_expired_policy_is_kept(leads):
    leads.loc[0, "discount_percent"] = 150
    leads.loc[1, "days_to_policy_expiry"] = -10

    clean, report = clean_leads(leads)

    row0 = clean[clean.lead_id == leads.lead_id[0]].iloc[0]
    row1 = clean[clean.lead_id == leads.lead_id[1]].iloc[0]
    assert np.isnan(row0["discount_percent"])
    assert row1["days_to_policy_expiry"] == -10
    assert report["out_of_range_set_to_null"] == {"discount_percent": 1}


def test_time_split_has_no_overlap(leads):
    clean, _ = clean_leads(leads)
    train, valid, test = time_split(clean, "2026-06-01", "2026-07-15")

    assert train.created_at.max() < valid.created_at.min()
    assert valid.created_at.max() < test.created_at.min()
    assert len(train) + len(valid) + len(test) == len(clean)
