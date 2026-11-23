"""
src/models/train.py
Build, tune and compare the three baseline models on the validation split.

Every model is a scikit-learn Pipeline (preprocessing + estimator), so all
trainable preprocessing (one-hot categories, scaler means/variances) is
fitted on the training split only and then applied unchanged to val.

Model selection:
1. For each grid configuration, fit on train and score val.
2. Keep the configuration with the highest val PR-AUC (average precision).
3. Set that model's decision threshold to the value maximising val F1.
Test splits are never used here.
"""

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models.metrics import best_f1_threshold, evaluate_scores

MODEL_NAMES = ("logistic_regression", "naive_bayes", "decision_tree")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _to_float(x):
    return np.asarray(x, dtype=float)


def build_preprocessor(cfg: dict, scale: bool, log1p_counts: bool) -> ColumnTransformer:
    """Return a ColumnTransformer for the 10 model features.

    - categorical: one-hot encoded; categories unseen in train are ignored.
    - boolean: cast to 0/1.
    - count features: optional log1p (stateless), then optional scaling.
    - other numeric (log_amount_paid): optional scaling.
    """
    groups = cfg["feature_groups"]
    counts = list(groups["log1p_counts"])
    other_numeric = [c for c in groups["numeric"] if c not in counts]

    count_steps = []
    if log1p_counts:
        count_steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    if scale:
        count_steps.append(("scale", StandardScaler()))
    count_tf = Pipeline(count_steps) if count_steps else "passthrough"
    other_tf = StandardScaler() if scale else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), groups["categorical"]),
            ("bool", FunctionTransformer(_to_float, feature_names_out="one-to-one"), groups["boolean"]),
            ("amount", other_tf, other_numeric),
            ("counts", count_tf, counts),
        ],
        verbose_feature_names_out=True,
    )


def build_estimator(name: str, params: dict, random_state: int):
    """Return an unfitted estimator for ``name`` with ``params``."""
    if name == "logistic_regression":
        return LogisticRegression(random_state=random_state, **params)
    if name == "naive_bayes":
        return GaussianNB(**params)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state, **params)
    raise ValueError(f"Unknown model name: {name!r}")


def build_pipeline(name: str, params: dict, cfg: dict) -> Pipeline:
    """Return preprocessing + estimator for one model configuration."""
    model_cfg = cfg["models"][name]
    prep = model_cfg["preprocessing"]
    all_params = {**model_cfg.get("fixed_params", {}), **params}
    return Pipeline([
        ("prep", build_preprocessor(cfg, scale=prep["scale"], log1p_counts=prep["log1p_counts"])),
        ("model", build_estimator(name, all_params, cfg["random_state"])),
    ])


def iter_grid(grid: dict):
    """Yield every combination of a {param: [values]} grid as a dict."""
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def positive_scores(pipeline: Pipeline, X) -> np.ndarray:
    """Return the predicted probability of the laundering class."""
    return pipeline.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_model(name, cfg, X_train, y_train, X_val, y_val, progress=None) -> dict:
    """Fit every grid configuration on train and pick the best by val PR-AUC.

    Returns a dict with:
      grid_results  DataFrame, one row per configuration
      best_params   winning configuration
      pipeline      fitted winning Pipeline
      threshold     val-F1-optimal threshold for the winner
      val_scores    winner's val probabilities
      metrics_tuned / metrics_default   val metrics at that threshold / at 0.5
    """
    rows = []
    best = None
    for params in iter_grid(cfg["models"][name]["grid"]):
        t0 = time.time()
        pipe = build_pipeline(name, params, cfg)
        pipe.fit(X_train, y_train)
        val_scores = positive_scores(pipe, X_val)
        train_pr_auc = average_precision_score(y_train, positive_scores(pipe, X_train))
        val_pr_auc = average_precision_score(y_val, val_scores)
        row = {
            **{k: ("None" if v is None else v) for k, v in params.items()},
            "train_pr_auc": train_pr_auc,
            "val_pr_auc": val_pr_auc,
            "fit_seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        if progress:
            progress(f"[{name}] {params} -> val PR-AUC {val_pr_auc:.4f} (train {train_pr_auc:.4f})")
        if best is None or val_pr_auc > best["val_pr_auc"]:
            best = {"params": params, "pipeline": pipe, "val_scores": val_scores, "val_pr_auc": val_pr_auc}

    threshold = best_f1_threshold(y_val, best["val_scores"])
    return {
        "name": name,
        "grid_results": pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True),
        "best_params": best["params"],
        "pipeline": best["pipeline"],
        "threshold": threshold,
        "val_scores": best["val_scores"],
        "metrics_tuned": evaluate_scores(y_val, best["val_scores"], threshold),
        "metrics_default": evaluate_scores(y_val, best["val_scores"], 0.5),
    }


def run_model_selection(cfg, X_train, y_train, X_val, y_val, models=MODEL_NAMES, progress=None) -> dict:
    """Tune each model in ``models``; return {name: tune_model result}."""
    return {
        name: tune_model(name, cfg, X_train, y_train, X_val, y_val, progress=progress)
        for name in models
    }


def comparison_table(results: dict, which: str = "metrics_tuned") -> pd.DataFrame:
    """Return one row of val metrics per model, best PR-AUC first."""
    table = pd.DataFrame(
        {name: {"best_params": str(r["best_params"]), **r[which]} for name, r in results.items()}
    ).T
    return table.sort_values("pr_auc", ascending=False)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def feature_names(pipeline: Pipeline) -> list[str]:
    """Return the post-preprocessing feature names of a fitted pipeline."""
    return list(pipeline.named_steps["prep"].get_feature_names_out())


def logistic_coefficients(pipeline: Pipeline) -> pd.Series:
    """Coefficients (log-odds per unit / per standard deviation), largest magnitude first."""
    coef = pd.Series(pipeline.named_steps["model"].coef_[0], index=feature_names(pipeline))
    return coef.reindex(coef.abs().sort_values(ascending=False).index)


def tree_importances(pipeline: Pipeline) -> pd.Series:
    """Impurity-based feature importances, largest first."""
    imp = pd.Series(pipeline.named_steps["model"].feature_importances_, index=feature_names(pipeline))
    return imp.sort_values(ascending=False)


def naive_bayes_separation(pipeline: Pipeline) -> pd.Series:
    """Per-feature class separation: (mean_laundering - mean_legit) / pooled std."""
    nb = pipeline.named_steps["model"]
    pooled_sd = np.sqrt(nb.var_.mean(axis=0))
    sep = pd.Series((nb.theta_[1] - nb.theta_[0]) / pooled_sd, index=feature_names(pipeline))
    return sep.reindex(sep.abs().sort_values(ascending=False).index)
