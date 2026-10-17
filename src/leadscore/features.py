"""Feature engineering shared by training and scoring.

LeadFeatureBuilder is a sklearn transformer so the exact same imputation values
learned on the training window are reused at scoring time (they're pickled with
the model).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL

UNKNOWN = "Unknown"


class LeadFeatureBuilder(BaseEstimator, TransformerMixin):
    """Picks the allowed raw columns, imputes and adds derived features.

    `columns` is the whitelist of raw inputs. Anything not in it (target,
    expected_margin, the incoming call flag for the variant without it) is
    never looked at.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        X = X[self._raw_cols()]
        self.cat_cols_ = [c for c in self.columns if c in CATEGORICAL]
        self.num_cols_ = [c for c in self.columns if c not in CATEGORICAL]

        # price depends a lot on the product, so impute it per product
        self.price_by_product_ = X.groupby("product_type")["price"].median().to_dict()
        self.medians_ = X[self.num_cols_].median().to_dict()
        return self

    def _raw_cols(self):
        return list(self.columns) + ["created_at"]

    def transform(self, X):
        X = X[self._raw_cols()].copy()
        out = pd.DataFrame(index=X.index)

        for c in self.cat_cols_:
            out[c] = X[c].astype(object).where(X[c].notna(), UNKNOWN)

        out["price_missing"] = X["price"].isna().astype(int)
        out["discount_missing"] = X["discount_percent"].isna().astype(int)
        product_median = X["product_type"].map(self.price_by_product_)
        X["price"] = X["price"].fillna(product_median)
        for c in self.num_cols_:
            X[c] = X[c].astype(float).fillna(self.medians_[c])

        for c in self.num_cols_:
            if c != "price":
                out[c] = X[c]

        # price is right-skewed (1.5M to 40M), log works better for the linear model
        out["log_price"] = np.log(X["price"])
        out["log_net_price"] = np.log(X["price"] * (1 - X["discount_percent"] / 100))
        out["policy_expired"] = (X["days_to_policy_expiry"] < 0).astype(int)
        sessions = X["sessions_last_7d"].clip(lower=1)
        out["offer_views_per_session"] = X["offer_views_last_7d"] / sessions
        out["comparisons_per_session"] = X["price_comparisons_last_7d"] / sessions

        created = pd.to_datetime(X["created_at"])
        out["hour"] = created.dt.hour
        out["day_of_week"] = created.dt.dayofweek
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns())

    def output_columns(self):
        num = [c for c in self.num_cols_ if c != "price"]
        return (self.cat_cols_ + ["price_missing", "discount_missing"] + num
                + ["log_price", "log_net_price", "policy_expired",
                   "offer_views_per_session", "comparisons_per_session",
                   "hour", "day_of_week"])


def make_encoder(scale):
    """One-hot for categoricals; numerics scaled for logreg, passed through for trees."""
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             make_column_selector(dtype_include=object)),
            ("num", num_step, make_column_selector(dtype_exclude=object)),
        ],
        verbose_feature_names_out=False,
    )
