import numpy as np
import pandas as pd
import pytest


def make_leads(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "lead_id": [f"L{i:05d}" for i in range(n)],
        "created_at": pd.Timestamp("2026-04-01") + pd.to_timedelta(rng.integers(0, 150 * 24, n), unit="h"),
        "product_type": rng.choice(["thirdparty", "carbody"], n),
        "channel": rng.choice(["SEO", "Paid", "CRM", "Referral"], n),
        "device": rng.choice(["mobile", "desktop"], n),
        "partner": rng.choice(["direct", "shoraka"], n),
        "city": rng.choice(["Tehran", "Shiraz", None], n),
        "insurance_company": rng.choice(["Iran", "Asia"], n),
        "payment_type": rng.choice(["cash", "bnpl"], n),
        "minutes_since_abandonment": rng.integers(1, 300, n),
        "days_to_policy_expiry": rng.integers(-20, 60, n),
        "price": rng.uniform(2e6, 2e7, n),
        "discount_percent": rng.uniform(0, 15, n),
        "has_previous_purchase": rng.integers(0, 2, n),
        "visited_offer_page": rng.integers(0, 2, n),
        "incoming_call_last_24h": rng.integers(0, 2, n),
        "sessions_last_7d": rng.integers(1, 10, n),
        "offer_views_last_7d": rng.integers(0, 8, n),
        "price_comparisons_last_7d": rng.integers(0, 5, n),
        "days_since_last_visit": rng.uniform(0, 30, n),
    })
    df["expected_margin"] = df["price"] * 0.026
    df["completed_purchase"] = rng.binomial(1, 0.1, n)
    return df


@pytest.fixture
def leads():
    return make_leads()
