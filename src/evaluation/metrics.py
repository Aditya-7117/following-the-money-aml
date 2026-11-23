"""
src/evaluation/metrics.py
Evaluation metrics for the laundering class (label 1).

- score: the model's suspicion score. Higher = more suspicious.
- flag:  a transaction is flagged (alerted) when score >= threshold.

If a value cannot be worked out, for example precision when nothing
is flagged, we return NaN and write it as "n/a". We never turn it into 0.
"""

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_metrics(y_true, scores, threshold):
    """Flag every score >= threshold, then measure the result."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)

    # labels=[0, 1] keeps the table 2x2 even if one class is missing
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        # zero_division=np.nan gives "n/a" instead of a fake 0
        "precision": precision_score(y_true, y_pred, zero_division=np.nan),
        "recall": recall_score(y_true, y_pred, zero_division=np.nan),
        "f1": f1_score(y_true, y_pred, zero_division=np.nan),
        "alert_fraction": y_pred.mean(),
    }


def average_precision(y_true, scores):
    """Area under the precision-recall curve (PR-AUC). NaN if there is no laundering."""
    if np.sum(y_true) == 0:
        return np.nan
    return average_precision_score(y_true, scores)


def roc_auc(y_true, scores):
    """Area under the ROC curve. NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, scores)


def recall_at_budget(df, budget=0.01):
    """Alert only the top 1% of scores and measure how much laundering we catch.

    df needs the columns: transaction_id, is_laundering, score.
    Ties in score are broken by the smaller transaction_id, so the result
    is the same every time. The labels are only used AFTER the alerts are
    chosen. This is a batch-ranking comparison, not a live threshold.
    """
    # round() stops float noise, e.g. 0.07 * 100 = 7.000000000000001
    n_alerts = math.ceil(round(budget * len(df), 9))

    ranked = df.sort_values(["score", "transaction_id"], ascending=[False, True])
    alerted = ranked.head(n_alerts)
    not_alerted = ranked.iloc[n_alerts:]

    caught = int(alerted["is_laundering"].sum())
    total = int(df["is_laundering"].sum())

    # If many rows share the cut-off score, the ID (not the model) decided
    # who got an alert. We count those rows so the report can say so.
    tied_left_out = 0
    if n_alerts > 0:
        cutoff_score = alerted["score"].iloc[-1]
        tied_left_out = int((not_alerted["score"] == cutoff_score).sum())

    return {
        "n_rows": len(df),
        "alerts_allowed": n_alerts,
        "laundering_rows": total,
        "caught": caught,
        "recall_at_budget": caught / total if total else np.nan,
        "tied_rows_left_out": tied_left_out,
    }
