"""Measures that matter in medicine, not just accuracy.

sensitivity = TP / (TP + FN)   of all sick patients, how many we caught
specificity = TN / (TN + FP)   of all healthy patients, how many we correctly cleared
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


# Turn raw model scores (logits) into probabilities. temperature > 1 makes them less confident
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------- several classes

# Accuracy, balanced accuracy, sensitivity (recall) and precision per class, confusion matrix, AUROC
def multiclass_metrics(y: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    pred = probs.argmax(1)
    class_numbers = list(range(len(classes)))
    precision, recall, _, support = precision_recall_fscore_support(y, pred, labels=class_numbers, zero_division=0)

    result = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class": {
            c: {"sensitivity": float(recall[i]), "precision": float(precision[i]), "support": int(support[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=class_numbers).tolist(),
        "calibration_error": calibration_error(y, probs),
    }

    # AUROC needs every class to be present
    if len(np.unique(y)) == len(classes) > 2:
        result["macro_auroc"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    elif len(np.unique(y)) == len(classes) == 2:
        result["macro_auroc"] = float(roc_auc_score(y, probs[:, 1]))

    return result


# ------------------------------------------------------------------------------------------- COVID yes / no

# Sensitivity, specificity, PPV, NPV and AUROC for y in {0, 1} and score = probability of COVID
def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    result = {
        "n": int(len(y)),
        "n_covid": int(y.sum()),
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(len(y), 1),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }

    # AUROC and sensitivity at 95% specificity need both COVID and non-COVID cases
    if 0 < y.sum() < len(y):
        result["auroc"] = float(roc_auc_score(y, score))
        result["sensitivity_at_95_specificity"] = sensitivity_at_specificity(y, score, 0.95)

    return result


# Best sensitivity reachable while keeping specificity >= target (e.g. 95% of healthy people cleared)
def sensitivity_at_specificity(y: np.ndarray, score: np.ndarray, target_specificity: float) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    allowed = (1 - fpr) >= target_specificity
    return float(tpr[allowed].max()) if allowed.any() else 0.0


# Highest threshold that still catches `target` of the COVID cases. Chosen on validation data only
def threshold_for_sensitivity(y: np.ndarray, score: np.ndarray, target: float = 0.95) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    reached = tpr >= target
    return float(np.clip(thresholds[reached].max(), 0.0, 1.0)) if reached.any() else 0.0


# ------------------------------------------------------------------------------------------- calibration

# Calibration error: average gap between how confident the model is and how often it is right (0 = perfect)
def calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    confidence = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)

    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


# Temperature scaling: find one number T that makes the validation probabilities honest (T > 1 = less confident)
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def loss_fn():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_t / log_t.exp(), y_t)
        loss.backward()
        return loss

    optimizer.step(loss_fn)
    return float(log_t.exp().clamp(0.05, 20).item())


# ------------------------------------------------------------------------------------------- confidence intervals

# 95% confidence interval by bootstrap: recompute the measure on many resamples of the test set.
# With `groups`, whole patients are resampled, because slices of one patient are not independent
def bootstrap_ci(metric: Callable[[np.ndarray], float], n: int, groups: np.ndarray | None = None,
                 n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if groups is not None:
        _, group_of_row = np.unique(groups, return_inverse=True)
        rows_of_group = [np.flatnonzero(group_of_row == g) for g in range(group_of_row.max() + 1)]

    values = []
    for _ in range(n_boot):
        if groups is None:
            rows = rng.integers(0, n, n)
        else:
            picked = rng.integers(0, len(rows_of_group), len(rows_of_group))
            rows = np.concatenate([rows_of_group[g] for g in picked])

        # A resample can contain a single class, where AUROC is undefined: skip it
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = metric(rows)
        except ValueError:
            continue
        if value is not None and not np.isnan(value):
            values.append(value)

    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
