"""
src/evaluation/evaluate.py
Score the val and test rows with the saved models, add two baseline rules,
and build the results tables: overall, at the 1% alert budget, and per
laundering pattern.

Rules are in config/evaluation.json.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import average_precision, recall_at_budget, roc_auc, threshold_metrics
from src.models.data import feature_columns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_CONFIG_PATH = PROJECT_ROOT / "config" / "evaluation.json"

# what each model is allowed to look at (shown in the results table)
FEATURE_SET = {
    "always_legitimate": "none",
    "ach_rule": "payment_format only",
    "logistic_regression": "10 features",
    "naive_bayes": "10 features",
    "decision_tree": "10 features",
}


def load_evaluation_config(path=EVALUATION_CONFIG_PATH):
    """Read config/evaluation.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model(models_dir, name):
    """Open a saved model: fitted pipeline, val threshold and parameters."""
    with open(Path(models_dir) / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def score_all(df, models, model_cfg):
    """Give every row one score per model, plus the two baseline rules.

    Returns a table of scores and a dict of thresholds.
    """
    scores = df[["transaction_id", "split", "is_laundering"]].copy()
    thresholds = {}

    # Baseline 1: never flag anything (score 0 is always below threshold 1)
    scores["always_legitimate"] = 0.0
    thresholds["always_legitimate"] = 1.0

    # Baseline 2: flag every ACH payment (score 1 for ACH, threshold 1)
    scores["ach_rule"] = (df["payment_format"] == "ACH").astype(float)
    thresholds["ach_rule"] = 1.0

    # Saved models: probability of laundering, and the threshold chosen on val
    X = df[feature_columns(model_cfg)]
    for name, saved in models.items():
        scores[name] = saved["pipeline"].predict_proba(X)[:, 1]
        thresholds[name] = saved["threshold"]

    return scores, thresholds


def build_tables(scores, thresholds, views, budget):
    """Measure every model on every view. Returns (results, budget) tables."""
    results, budget_rows = [], []

    for model, threshold in thresholds.items():
        for view, splits in views.items():
            part = scores[scores["split"].isin(splits)]
            y, s = part["is_laundering"], part[model]

            results.append({
                "model": model,
                "feature_set": FEATURE_SET.get(model, ""),
                "test_view": view,
                "n_rows": len(part),
                "laundering_rows": int(y.sum()),
                **threshold_metrics(y, s, threshold),
                "pr_auc": average_precision(y, s),
                "roc_auc": roc_auc(y, s),
            })

            ranked = part[["transaction_id", "is_laundering"]].assign(score=s)
            budget_rows.append({
                "model": model,
                "test_view": view,
                **recall_at_budget(ranked, budget),
            })

    return pd.DataFrame(results), pd.DataFrame(budget_rows)


def pattern_recall(scores, thresholds, views):
    """For each model, test view and pattern: how many laundering rows were flagged.

    scores needs a pattern_type column. Laundering rows without a named
    pattern are grouped as "none". A pattern with no rows in a view gets n/a.
    """
    laundering = scores[scores["is_laundering"] == 1].copy()
    laundering["pattern_type"] = laundering["pattern_type"].fillna("none")
    all_patterns = sorted(laundering["pattern_type"].unique())

    rows = []
    for model, threshold in thresholds.items():
        for view, splits in views.items():
            if view == "val":
                continue
            part = laundering[laundering["split"].isin(splits)]
            for pattern in all_patterns:
                group = part[part["pattern_type"] == pattern]
                caught = int((group[model] >= threshold).sum())
                rows.append({
                    "model": model,
                    "test_view": view,
                    "pattern_type": pattern,
                    "laundering_rows": len(group),
                    "caught": caught,
                    "recall": caught / len(group) if len(group) else np.nan,
                })

    return pd.DataFrame(rows)
