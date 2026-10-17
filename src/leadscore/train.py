"""Train both feature-set variants, pick the algorithm per variant, decide on the
incoming call feature and register the models in Postgres.

Model selection and the incoming-call decision only look at the validation
month. The test month is left for `evaluate`.
"""
import json
import logging
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import feature_columns
from .data import clean_leads, load_raw_leads, time_split
from .db import execute, init_schema
from .model import LeadScoringModel, fit_logreg, fit_xgboost
from .monitoring import build_reference

log = logging.getLogger(__name__)

VARIANTS = {"without_call": False, "with_call": True}


def decide_incoming_call(valid_without, valid_with, timing_confirmed, min_gain):
    """Should the incoming call flag go into the production model?

    Default is no. It goes in only when somebody has confirmed the call is
    known before the lead is scored AND it actually helps.
    """
    pr_gain = valid_with["pr_auc"] - valid_without["pr_auc"]
    roc_gain = valid_with["roc_auc"] - valid_without["roc_auc"]
    material = pr_gain >= min_gain

    if timing_confirmed and material:
        include = True
        reason = (f"Timing is confirmed and the feature adds {pr_gain:+.4f} PR-AUC on "
                  "validation, so it is used.")
    elif timing_confirmed:
        include = False
        reason = (f"Timing is confirmed but the gain ({pr_gain:+.4f} PR-AUC) is below the "
                  f"{min_gain} threshold, so it is not worth the extra dependency.")
    elif not material:
        include = False
        reason = (f"The gain is negligible ({pr_gain:+.4f} PR-AUC, below {min_gain}). "
                  "Not worth the risk of using a feature that may be recorded after "
                  "the moment we score the lead.")
    else:
        include = False
        reason = (f"The feature adds {pr_gain:+.4f} PR-AUC on validation, but we cannot "
                  "confirm the call happens before scoring. A call recorded after "
                  "abandonment could be part of the outcome itself, so the gain may not "
                  "hold in production. Excluded until the timing is confirmed.")

    return {
        "include": include,
        "chosen_variant": "with_call" if include else "without_call",
        "pr_auc_gain": round(pr_gain, 4),
        "roc_auc_gain": round(roc_gain, 4),
        "timing_confirmed": timing_confirmed,
        "min_pr_auc_gain": min_gain,
        "reason": reason,
    }


def _valid_metrics(y, p):
    return {"roc_auc": round(roc_auc_score(y, p), 4),
            "pr_auc": round(average_precision_score(y, p), 4)}


def run_training(engine, cfg):
    init_schema(engine)
    target = cfg["target"]
    df, _ = clean_leads(load_raw_leads(engine, cfg["data"]["raw_table"]))
    train, valid, _ = time_split(df, cfg["split"]["train_end"], cfg["split"]["valid_end"])
    y_train, y_valid = train[target].astype(int), valid[target].astype(int)

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = cfg["paths"]["models"] / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {"run_tag": run_tag, "split": cfg["split"],
               "sizes": {"train": len(train), "valid": len(valid)},
               "candidates": {}, "finalists": {}}

    for variant, include_call in VARIANTS.items():
        cols = feature_columns(cfg, include_call)
        log.info("training variant %s (%d raw columns)", variant, len(cols))

        lr_pipe, lr_params, lr_trials = fit_logreg(
            train, y_train, valid, y_valid, cols, cfg["model"]["logreg"]["C"])
        xgb_pipe, xgb_params, xgb_trials = fit_xgboost(
            train, y_train, valid, y_valid, cols, cfg["model"]["xgboost"], cfg["seed"])

        best = None
        for algo, pipe, params, trials in [("logreg", lr_pipe, lr_params, lr_trials),
                                           ("xgboost", xgb_pipe, xgb_params, xgb_trials)]:
            model = LeadScoringModel(pipe, algo, variant, cols, params)
            model.calibrate(valid, y_valid)
            model.model_version = f"{run_tag}-{algo}-{variant}"
            metrics = _valid_metrics(y_valid, model.predict_proba(valid))

            path = run_dir / f"{algo}_{variant}.joblib"
            summary["candidates"][f"{algo}_{variant}"] = {
                "model_version": model.model_version, "algorithm": algo,
                "variant": variant, "params": params, "valid": metrics,
                "search": trials, "artifact": str(path),
            }
            if best is None or metrics["pr_auc"] > best[1]["pr_auc"]:
                best = (model, metrics, path)
            # save every candidate so evaluate can report all four on test
            model.reference = build_reference(
                train, cols, model.predict_proba(valid), cfg["monitoring"]["psi_bins"])
            model.save(path)

        model, metrics, path = best
        summary["finalists"][variant] = f"{model.algorithm}_{variant}"
        log.info("variant %s -> %s (valid PR-AUC %.4f)", variant, model.algorithm, metrics["pr_auc"])

    fin = {v: summary["candidates"][k] for v, k in summary["finalists"].items()}
    lr_cfg = cfg["leakage_review"]
    decision = decide_incoming_call(fin["without_call"]["valid"], fin["with_call"]["valid"],
                                    lr_cfg["timing_confirmed"], lr_cfg["min_pr_auc_gain"])
    summary["incoming_call_decision"] = decision
    log.info("incoming call decision: include=%s. %s", decision["include"], decision["reason"])

    _register(engine, summary, fin, decision, train)

    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    with open(cfg["paths"]["models"] / "LATEST", "w") as f:
        f.write(run_tag)
    return summary


def _register(engine, summary, finalists, decision, train):
    """Both finalists go into ml.model_versions, only the chosen one is active."""
    chosen = decision["chosen_variant"]
    execute(engine, "UPDATE ml.model_versions SET is_active = FALSE WHERE is_active")
    for variant, cand in finalists.items():
        active = variant == chosen
        note = ("ACTIVE. " if active else "Not active. ") + decision["reason"]
        execute(
            engine,
            """INSERT INTO ml.model_versions
               (model_version, algorithm, feature_set, features, params, train_from, train_to,
                metrics, decision_note, artifact_path, is_active)
               VALUES (:mv, :algo, :fs, CAST(:features AS JSONB), CAST(:params AS JSONB),
                       :tf, :tt, CAST(:metrics AS JSONB), :note, :path, :active)""",
            mv=cand["model_version"], algo=cand["algorithm"], fs=variant,
            features=json.dumps(LeadScoringModel.load(cand["artifact"]).columns),
            params=json.dumps(cand["params"], default=_json_default),
            tf=train["created_at"].min().to_pydatetime(),
            tt=train["created_at"].max().to_pydatetime(),
            metrics=json.dumps({"valid": cand["valid"]}), note=note,
            path=cand["artifact"], active=active,
        )


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)
