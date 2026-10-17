import numpy as np
import pandas as pd

from leadscore.data import clean_leads
from leadscore.features import UNKNOWN, LeadFeatureBuilder

COLUMNS = ["product_type", "channel", "device", "partner", "city", "insurance_company",
           "payment_type", "minutes_since_abandonment", "days_to_policy_expiry", "price",
           "discount_percent", "has_previous_purchase", "visited_offer_page", "sessions_last_7d",
           "offer_views_last_7d", "price_comparisons_last_7d", "days_since_last_visit"]


def test_output_ignores_columns_outside_the_whitelist(leads):
    df, _ = clean_leads(leads)
    builder = LeadFeatureBuilder(COLUMNS).fit(df)
    before = builder.transform(df)

    # target, margin and the call flag must not influence the features
    changed = df.copy()
    changed["completed_purchase"] = 1 - changed["completed_purchase"]
    changed["expected_margin"] = 0
    changed["incoming_call_last_24h"] = 1
    after = builder.transform(changed)

    pd.testing.assert_frame_equal(before, after)
    assert "incoming_call_last_24h" not in before.columns
    assert list(before.columns) == builder.output_columns()


def test_price_is_imputed_per_product(leads):
    df, _ = clean_leads(leads)
    df.loc[df.product_type == "carbody", "price"] = 30e6
    df.loc[df.product_type == "thirdparty", "price"] = 5e6
    builder = LeadFeatureBuilder(COLUMNS).fit(df)

    new = df.head(2).copy()
    new["product_type"] = ["carbody", "thirdparty"]
    new["price"] = np.nan
    out = builder.transform(new)

    assert np.allclose(np.exp(out["log_price"]), [30e6, 5e6])
    assert out["price_missing"].tolist() == [1, 1]


def test_missing_and_unseen_categories(leads):
    df, _ = clean_leads(leads)
    builder = LeadFeatureBuilder(COLUMNS).fit(df)

    new = df.head(2).copy()
    new["city"] = [pd.NA, "Yazd"]
    out = builder.transform(new)

    assert out["city"].tolist() == [UNKNOWN, "Yazd"]
