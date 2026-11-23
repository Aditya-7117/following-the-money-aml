"""
src/models/metrics.py
Validation metrics for heavily imbalanced binary classification.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def best_f1_threshold(y_true, scores) -> float:
    """Return the score threshold that maximises F1 on (y_true, scores).

    Predictions are positive when ``score >= threshold``.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall pair has no threshold; drop it.
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(f1))])


def evaluate_scores(y_true, scores, threshold: float) -> dict:
    """Return threshold-free and thresholded metrics as a flat dict."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
