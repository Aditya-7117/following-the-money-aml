"""
Evaluation metrics for the AML project (Priyanka, Person 4).

We always measure the LAUNDERING class (label 1).
- score: the model's suspicion score. Higher = more suspicious.
- flag:  a transaction is flagged (alerted) when score >= threshold.

If a value cannot be worked out, for example precision when nothing
is flagged, we return NaN and show it as "n/a". We never turn it into 0.
"""

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
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


def choose_threshold(y_val, val_scores):
    """Pick the threshold with the best F1. Use VALIDATION data only.

    If two thresholds give the same F1, we keep the higher one,
    because it raises fewer alerts.
    """
    if np.sum(y_val) == 0:
        raise ValueError("Validation data has no laundering rows")

    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)

    # the last precision/recall pair has no threshold, so we drop it
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)

    # thresholds are sorted low to high, so the last best one is the highest
    best = np.flatnonzero(f1 == f1.max())[-1]
    return thresholds[best]


def average_precision(y_true, scores):
    """Area under the precision-recall curve. NaN if there is no laundering."""
    if np.sum(y_true) == 0:
        return np.nan
    return average_precision_score(y_true, scores)


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

    caught = int(alerted["is_laundering"].sum())
    total = int(df["is_laundering"].sum())

    return {
        "n_rows": len(df),
        "alerts_allowed": n_alerts,
        "laundering_rows": total,
        "caught": caught,
        "recall_at_budget": caught / total if total else np.nan,
    }
