"""Metric helpers with explicit sensitivity/specificity reporting."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float | int]:
    """Calculate thresholded classification metrics and ROC-AUC."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    auc = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
