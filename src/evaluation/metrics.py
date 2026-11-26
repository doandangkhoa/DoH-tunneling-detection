"""Binary metrics for benign (0) and tunnel (1)."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def binary_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = np.unique(y_true).size == 2
    return {
        "n": len(y_true), "benign": int(tn + fp), "tunnel": int(tp + fn),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": ratio(tn + tp, len(y_true)),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "fpr": ratio(fp, fp + tn),
        "ap": average_precision_score(y_true, y_score) if both_classes else np.nan,
        "roc_auc": roc_auc_score(y_true, y_score) if both_classes else np.nan,
    }
