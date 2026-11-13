"""Figures saved next to the results: confusion matrix, ROC curve, calibration plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, roc_curve


# Confusion matrix: rows = true class, columns = predicted class, with counts and row percentages
def plot_confusion(matrix: np.ndarray, classes: list[str], path: Path) -> None:
    fig = Figure(figsize=(1.4 * len(classes) + 2, 1.2 * len(classes) + 1.5))
    ax = fig.subplots()
    row_share = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)

    for i in range(len(classes)):
        for j in range(len(classes)):
            colour = "white" if row_share[i, j] > 0.6 else "black"
            ax.text(j, i, f"{matrix[i, j]}\n{row_share[i, j]:.0%}", ha="center", va="center", color=colour, fontsize=9)

    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    _save(fig, path)


# ROC curves: sensitivity against (1 - specificity) for every threshold. `curves` = {name: (y, score)}
def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig = Figure(figsize=(4.5, 4.5))
    ax = fig.subplots()

    for name, (y, score) in curves.items():
        if 0 < y.sum() < len(y):
            fpr, tpr, _ = roc_curve(y, score)
            ax.plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y, score):.3f})")

    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


# Calibration plot: how often the model is right vs. how confident it is, before and after temperature scaling
def plot_calibration(y: np.ndarray, probs: np.ndarray, probs_scaled: np.ndarray, path: Path, n_bins: int = 10) -> None:
    fig = Figure(figsize=(4.5, 4.5))
    ax = fig.subplots()
    edges = np.linspace(0, 1, n_bins + 1)

    for name, p in (("before", probs), ("after temperature scaling", probs_scaled)):
        confidence, correct = p.max(1), p.argmax(1) == y
        xs, ys = [], []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            in_bin = (confidence > low) & (confidence <= high)
            if in_bin.sum() >= 5:
                xs.append(confidence[in_bin].mean())
                ys.append(correct[in_bin].mean())
        ax.plot(xs, ys, marker="o", label=name)

    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    ax.set_xlabel("confidence")
    ax.set_ylabel("share of correct predictions")
    ax.legend(fontsize=8)
    _save(fig, path)


# Save a figure, creating its folder if needed
def _save(fig: Figure, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
