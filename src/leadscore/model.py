"""Model wrapper and the two estimators we compare (logistic regression, XGBoost)."""
import logging

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .features import LeadFeatureBuilder, make_encoder

log = logging.getLogger(__name__)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class LeadScoringModel:
    """Fitted pipeline + Platt calibration + the metadata needed to use it later.

    Platt scaling (a 1-d logistic regression on the raw logit) is used instead
    of isotonic because it is monotonic and smooth: the ranking of leads stays
    exactly the same as the raw model, isotonic would create ties at the top.
    """

    def __init__(self, pipeline, algorithm, feature_set, columns, params):
        self.pipeline = pipeline
        self.algorithm = algorithm
        self.feature_set = feature_set
        self.columns = columns
        self.params = params
        self.calibrator = None
        self.model_version = None
        self.reference = None  # filled by training, used by monitoring

    def predict_raw(self, df):
        return self.pipeline.predict_proba(df)[:, 1]

    def calibrate(self, df, y):
        self.calibrator = LogisticRegression(C=1e6)
        self.calibrator.fit(_logit(self.predict_raw(df)), y)
        return self

    def predict_proba(self, df):
        raw = self.predict_raw(df)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict_proba(_logit(raw))[:, 1]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def fit_logreg(train, y_train, valid, y_valid, columns, c_grid):
    best, best_ap, trials = None, -1, []
    for c in c_grid:
        pipe = Pipeline([
            ("features", LeadFeatureBuilder(columns)),
            ("encode", make_encoder(scale=True)),
            ("clf", LogisticRegression(C=c, max_iter=2000)),
        ])
        pipe.fit(train, y_train)
        ap = average_precision_score(y_valid, pipe.predict_proba(valid)[:, 1])
        trials.append({"C": c, "valid_pr_auc": round(ap, 4)})
        if ap > best_ap:
            best, best_ap, best_c = pipe, ap, c
    log.info("logreg best C=%s valid PR-AUC=%.4f", best_c, best_ap)
    return best, {"C": best_c}, trials


def sample_params(space, rng):
    return {k: v[rng.integers(len(v))] for k, v in space.items()}


def fit_xgboost(train, y_train, valid, y_valid, columns, xgb_cfg, seed):
    # features are deterministic given the train window, so build them once
    builder = LeadFeatureBuilder(columns).fit(train)
    encoder = make_encoder(scale=False)
    X_train = encoder.fit_transform(builder.transform(train))
    X_valid = encoder.transform(builder.transform(valid))

    rng = np.random.default_rng(seed)
    best, best_ap, trials = None, -1, []
    for i in range(xgb_cfg["n_trials"]):
        params = sample_params(xgb_cfg["search_space"], rng)
        clf = XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
            eval_metric="aucpr",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            **params,
        )
        clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        ap = average_precision_score(y_valid, clf.predict_proba(X_valid)[:, 1])
        trials.append({**params, "best_iteration": int(clf.best_iteration),
                       "valid_pr_auc": round(ap, 4)})
        if ap > best_ap:
            best, best_ap = clf, ap
            best_params = {**params, "n_estimators": int(clf.best_iteration) + 1}

    log.info("xgboost best valid PR-AUC=%.4f params=%s", best_ap, best_params)
    pipe = Pipeline([("features", builder), ("encode", encoder), ("clf", best)])
    return pipe, best_params, trials
