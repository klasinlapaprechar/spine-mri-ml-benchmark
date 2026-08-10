"""Minimal metric helpers for the public smoke harness."""
from __future__ import annotations
import numpy as np

def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")
