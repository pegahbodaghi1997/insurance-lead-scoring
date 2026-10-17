"""Evaluate the latest training run on the held-out test month.

Writes charts to charts/, a markdown + json report to reports/, and stores the
test metrics next to the validation ones in ml.model_versions.
"""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

from .data import clean_leads, load_raw_leads, time_split
from .db import execute
from .model import LeadScoringModel
from .monitoring import monthly_drift
from .plots import BLUE, GRAY, ORANGE, TEXT_2, pct_axis, plt, save

log = logging.getLogger(__name__)


def top_k_metrics(y, p, k):
    """Precision / recall / lift if sales only calls the top k share of leads."""
    y = np.asarray(y)
    n = max(int(round(len(y) * k)), 1)
    top = np.argsort(-p, kind="stable")[:n]
    precision = y[top].mean()
    return {"k": k, "n_called": n, "precision": round(float(precision), 4),
            "recall": round(float(y[top].sum() / max(y.sum(), 1)), 4),
            "lift": round(float(precision / y.mean()), 2)}


def classification_metrics(y, p, top_k):
    y = np.asarray(y)
    return {
        "n": int(len(y)),
        "base_rate": round(float(y.mean()), 4),
        "mean_score": round(float(p.mean()), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "log_loss": round(log_loss(y, p), 4),
        "top_k": [top_k_metrics(y, p, k) for k in top_k],
    }


def segment_metrics(df, y, p, segments):
    rows = []
    for seg in segments:
        for value, idx in df.groupby(seg, dropna=False).groups.items():
            ys, ps = y.loc[idx], pd.Series(p, index=df.index).loc[idx]
            if ys.nunique() < 2:
                continue
            rows.append({"segment": seg, "value": str(value), "n": len(idx),
                         "conversion": round(ys.mean(), 4), "mean_score": round(ps.mean(), 4),
                         "roc_auc": round(roc_auc_score(ys, ps), 4),
                         "pr_auc": round(average_precision_score(ys, ps), 4)})
    return pd.DataFrame(rows)


def permutation_importance(model, df, y, columns, seed, repeats=3):
    """Drop in PR-AUC when one raw input column is shuffled."""
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.predict_proba(df))
    rows = []
    for col in columns:
        drops = []
        for _ in range(repeats):
            shuffled = df.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(base - average_precision_score(y, model.predict_proba(shuffled)))
        rows.append({"feature": col, "pr_auc_drop": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False)


def latest_summary(cfg):
    models_dir = cfg["paths"]["models"]
    tag = (models_dir / "LATEST").read_text().strip()
    with open(models_dir / tag / "training_summary.json") as f:
        return json.load(f)


def run_evaluation(engine, cfg):
    target = cfg["target"]
    charts, reports = cfg["paths"]["charts"], cfg["paths"]["reports"]
    summary = latest_summary(cfg)
    decision = summary["incoming_call_decision"]

    df, _ = clean_leads(load_raw_leads(engine, cfg["data"]["raw_table"]))
    _, valid, test = time_split(df, cfg["split"]["train_end"], cfg["split"]["valid_end"])
    y_test = test[target].astype(int)

    results, preds, models = {}, {}, {}
    for name, cand in summary["candidates"].items():
        model = LeadScoringModel.load(cand["artifact"])
        p = model.predict_proba(test)
        models[name], preds[name] = model, p
        results[name] = {"model_version": cand["model_version"], "algorithm": cand["algorithm"],
                         "variant": cand["variant"], "valid": cand["valid"],
                         "test": classification_metrics(y_test, p, cfg["evaluation"]["top_k"])}
        log.info("%s test ROC-AUC=%.4f PR-AUC=%.4f", name,
                 results[name]["test"]["roc_auc"], results[name]["test"]["pr_auc"])

    finalists = summary["finalists"]
    chosen = finalists[decision["chosen_variant"]]
    chosen_model, p_chosen = models[chosen], preds[chosen]

    seg = segment_metrics(test, y_test, p_chosen, cfg["evaluation"]["segments"])
    imp = permutation_importance(chosen_model, test, y_test, chosen_model.columns, cfg["seed"])
    drift = monthly_drift(df, chosen_model, cfg)
    # inputs can be stable while the outcome moves, only labels show that
    by_month = (df.assign(pred=chosen_model.predict_proba(df),
                          month=df["created_at"].dt.to_period("M").astype(str))
                .groupby("month").agg(leads=(target, "size"), actual=(target, "mean"),
                                      predicted=("pred", "mean")).reset_index())

    # charts
    _plot_roc_pr(y_test, {n: preds[n] for n in finalists.values()}, charts)
    _plot_calibration(y_test, p_chosen, chosen, charts)
    _plot_gains(y_test, p_chosen, chosen, charts)
    _plot_score_distribution(y_test, p_chosen, charts)
    _plot_importance(imp, charts)
    _plot_call_comparison(results, finalists, decision, cfg, charts)
    _plot_monthly_drift(drift, cfg, charts)

    # write test metrics back to the registry
    for variant, name in finalists.items():
        execute(engine,
                """UPDATE ml.model_versions
                   SET metrics = metrics || CAST(:m AS JSONB) WHERE model_version = :mv""",
                m=json.dumps({"test": results[name]["test"]}),
                mv=results[name]["model_version"])

    report = {"training_run": summary["run_tag"], "split": summary["split"],
              "sizes": {**summary["sizes"], "test": len(test)},
              "finalists": finalists, "chosen_model": chosen,
              "incoming_call_decision": decision, "results": results,
              "segments": seg.to_dict("records"),
              "predicted_vs_actual_by_month": by_month.round(4).to_dict("records"),
              "permutation_importance": imp.to_dict("records")}
    with open(reports / "evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    (reports / "evaluation.md").write_text(_markdown(report, chosen, seg, imp, drift, by_month))
    log.info("evaluation written to %s", reports)
    return report


# ---------------------------------------------------------------- report

def _fmt_topk(m):
    return " / ".join(f"{t['precision']:.3f}" for t in m["top_k"])


def _markdown(r, chosen, seg, imp, drift, by_month):
    d = r["incoming_call_decision"]
    res = r["results"]
    ks = ", ".join(f"{int(t['k'] * 100)}%" for t in res[chosen]["test"]["top_k"])
    lines = [
        "# Model evaluation",
        "",
        f"Training run `{r['training_run']}`. Train < {r['split']['train_end']} <= validation "
        f"< {r['split']['valid_end']} <= test.",
        f"Rows: train {r['sizes']['train']}, validation {r['sizes']['valid']}, test {r['sizes']['test']}.",
        "",
        "## Incoming Call Last 24h: with vs. without",
        "",
        "Both variants are trained on every run. The table compares the best model of each.",
        "",
        "| Variant | Model | Valid ROC-AUC | Valid PR-AUC | Test ROC-AUC | Test PR-AUC | Test Brier | "
        f"Test precision @ {ks} |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variant, name in r["finalists"].items():
        m = res[name]
        lines.append(f"| {variant} | {m['algorithm']} | {m['valid']['roc_auc']:.4f} | "
                     f"{m['valid']['pr_auc']:.4f} | {m['test']['roc_auc']:.4f} | "
                     f"{m['test']['pr_auc']:.4f} | {m['test']['brier']:.4f} | {_fmt_topk(m['test'])} |")
    t_gain = res[r["finalists"]["with_call"]]["test"]["pr_auc"] - res[r["finalists"]["without_call"]]["test"]["pr_auc"]
    lines += [
        "",
        f"- PR-AUC gain from the feature on validation: **{d['pr_auc_gain']:+.4f}** "
        f"(ROC-AUC {d['roc_auc_gain']:+.4f}). On test: {t_gain:+.4f}.",
        f"- Timing confirmed: **{d['timing_confirmed']}**. Minimum gain to bother: {d['min_pr_auc_gain']}.",
        f"- **Decision: {'include' if d['include'] else 'exclude'} the feature. "
        f"Production model is `{res[chosen]['model_version']}`.**",
        f"- Reason: {d['reason']}",
        "",
        "## All candidates",
        "",
        "| Candidate | Valid PR-AUC | Test ROC-AUC | Test PR-AUC | Test Brier | Test log loss |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in res.items():
        lines.append(f"| {name} | {m['valid']['pr_auc']:.4f} | {m['test']['roc_auc']:.4f} | "
                     f"{m['test']['pr_auc']:.4f} | {m['test']['brier']:.4f} | {m['test']['log_loss']:.4f} |")

    t = res[chosen]["test"]
    lines += [
        "",
        f"## Chosen model on test (`{chosen}`)",
        "",
        f"Base rate {t['base_rate']:.3f}, mean predicted probability {t['mean_score']:.3f}.",
        "",
        "| Top k called | Leads | Precision | Recall | Lift |",
        "|---|---|---|---|---|",
    ]
    for row in t["top_k"]:
        lines.append(f"| {int(row['k'] * 100)}% | {row['n_called']} | {row['precision']:.3f} | "
                     f"{row['recall']:.3f} | {row['lift']:.2f} |")
    lines += ["", "### By segment", "", seg.to_markdown(index=False) if len(seg) else "n/a",
              "", "### Permutation importance (PR-AUC drop on test)", "",
              imp.assign(pr_auc_drop=imp.pr_auc_drop.round(4)).to_markdown(index=False),
              "", "### Drift by month (PSI vs. training window, features above 0.1 only)", ""]
    flagged = drift[drift.value >= 0.1]
    lines.append(flagged[["month", "feature", "value"]].round(3).to_markdown(index=False)
                 if len(flagged) else "No feature above 0.1 in any month.")
    lines += ["", "### Predicted vs. actual conversion by month", "",
              "Train months are in-sample, July is the calibration month, August is test.", "",
              by_month.round(4).to_markdown(index=False)]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- charts

def _plot_roc_pr(y, preds, charts):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for (name, p), color in zip(preds.items(), [BLUE, ORANGE]):
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=color, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        a2.plot(rec, prec, color=color, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1)
    a2.axhline(np.mean(y), color=GRAY, linestyle="--", linewidth=1, label="base rate")
    a1.set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
    a2.set(title="Precision-recall (test)", xlabel="Recall", ylabel="Precision")
    a1.legend(loc="lower right")
    a2.legend(loc="upper right")
    save(fig, charts / "eval_roc_pr_curves.png")


def _plot_calibration(y, p, name, charts):
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    lim = max(frac.max(), mean_pred.max()) * 1.1
    ax.plot([0, lim], [0, lim], color=GRAY, linestyle="--", linewidth=1, label="perfect")
    ax.plot(mean_pred, frac, color=BLUE, marker="o", markersize=5, label=name)
    ax.set(title="Calibration on test (deciles)", xlabel="Mean predicted probability",
           ylabel="Observed conversion rate", xlim=(0, lim), ylim=(0, lim))
    pct_axis(ax, "x")
    pct_axis(ax, "y")
    ax.legend(loc="upper left")
    save(fig, charts / "eval_calibration.png")


def _plot_gains(y, p, name, charts):
    y = np.asarray(y)
    order = np.argsort(-p, kind="stable")
    called = np.arange(1, len(y) + 1) / len(y)
    captured = np.cumsum(y[order]) / y.sum()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(called, captured, color=BLUE, label=name)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="random")
    for k in (0.1, 0.2, 0.3):
        i = int(k * len(y)) - 1
        ax.plot(called[i], captured[i], "o", color=BLUE, markersize=5)
        ax.annotate(f"{captured[i]:.0%}", (called[i], captured[i]), textcoords="offset points",
                    xytext=(6, -12), color=TEXT_2, fontsize=8)
    ax.set(title="Cumulative gains (test)", xlabel="Share of leads called (ranked by score)",
           ylabel="Share of purchases captured")
    pct_axis(ax, "x")
    pct_axis(ax, "y")
    ax.legend(loc="lower right")
    save(fig, charts / "eval_cumulative_gains.png")


def _plot_score_distribution(y, p, charts):
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(0, np.quantile(p, 0.995), 40)
    ax.hist(p[y == 0], bins=bins, color=BLUE, alpha=0.6, density=True, label="no purchase")
    ax.hist(p[y == 1], bins=bins, color=ORANGE, alpha=0.6, density=True, label="purchase")
    ax.set(title="Score distribution by outcome (test)", xlabel="Predicted probability", ylabel="Density")
    pct_axis(ax, "x")
    ax.legend()
    save(fig, charts / "eval_score_distribution.png")


def _plot_importance(imp, charts):
    imp = imp.sort_values("pr_auc_drop")
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * len(imp) + 1.2))
    ax.barh(imp["feature"], imp["pr_auc_drop"], color=BLUE, height=0.6)
    ax.set(title="Permutation importance (test)", xlabel="Drop in PR-AUC when shuffled")
    ax.grid(axis="y", visible=False)
    save(fig, charts / "eval_permutation_importance.png")


def _plot_call_comparison(results, finalists, decision, cfg, charts):
    """One small panel per metric so the (small) differences are visible
    without cutting the y-axis."""
    without, with_ = results[finalists["without_call"]], results[finalists["with_call"]]
    metrics = [("Valid PR-AUC", lambda m: m["valid"]["pr_auc"]),
               ("Test PR-AUC", lambda m: m["test"]["pr_auc"]),
               ("Test ROC-AUC", lambda m: m["test"]["roc_auc"])]
    for k in cfg["evaluation"]["top_k"][:3]:
        metrics.append((f"Test precision@{int(k * 100)}%",
                        lambda m, k=k: next(t["precision"] for t in m["test"]["top_k"] if t["k"] == k)))

    fig, axes = plt.subplots(1, len(metrics), figsize=(2 * len(metrics), 3.4), sharey=False)
    for ax, (label, f) in zip(axes, metrics):
        a, b = f(without), f(with_)
        bars = ax.bar(["without", "with"], [a, b], color=[BLUE, ORANGE], width=0.6)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, color=TEXT_2, padding=2)
        ax.set_ylim(0, max(a, b) * 1.2)
        ax.set_title(f"{label}\n{b - a:+.4f}", fontsize=9)
        ax.grid(axis="x", visible=False)
    verdict = "included" if decision["include"] else "excluded"
    fig.suptitle(f"Incoming Call Last 24h, without vs. with the feature "
                 f"(best model per variant). Decision: {verdict}", fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, charts / "eval_incoming_call_comparison.png")


def _plot_monthly_drift(drift, cfg, charts):
    pivot = drift.pivot(index="feature", columns="month", values="value")
    pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(pivot) + 1.4))
    im = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto", vmin=0,
                   vmax=max(cfg["monitoring"]["psi_warn"], float(pivot.max().max())))
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.06 else TEXT_2)
    ax.grid(False)
    ax.set_title(f"PSI by month vs. training window (warn at {cfg['monitoring']['psi_warn']})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="PSI")
    save(fig, charts / "monitoring_psi_by_month.png")
