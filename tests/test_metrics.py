"""Tests for the measures."""

import numpy as np

from covid_ct_diagnosis.metrics import (
    binary_metrics,
    bootstrap_ci,
    calibration_error,
    fit_temperature,
    multiclass_metrics,
    softmax,
    threshold_for_sensitivity,
)


def test_binary_metrics_counts():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.7, 0.3, 0.2])
    m = binary_metrics(y, score, 0.5)

    assert m["confusion"] == {"tp": 2, "fn": 1, "fp": 1, "tn": 4}
    assert np.isclose(m["sensitivity"], 2 / 3)
    assert np.isclose(m["specificity"], 4 / 5)


# The whole point of the project in one test: 95% accuracy while catching no sick patient at all
def test_accuracy_can_hide_missed_patients():
    y = np.array([1] * 5 + [0] * 95)
    m = binary_metrics(y, np.zeros(100), 0.5)
    assert m["accuracy"] == 0.95
    assert m["sensitivity"] == 0.0


def test_threshold_reaches_the_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    score = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    threshold = threshold_for_sensitivity(y, score, 0.95)
    assert binary_metrics(y, score, threshold)["sensitivity"] >= 0.95


def test_perfect_predictions_have_no_calibration_error():
    y = np.array([0, 1, 2, 1])
    assert calibration_error(y, np.eye(3)[y]) == 0.0


# A model 3x too confident should get a temperature of about 3
def test_temperature_scaling_detects_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 3)) * 2
    y = np.array([rng.choice(3, p=p) for p in softmax(true_logits)])
    assert 2.5 < fit_temperature(true_logits * 3.0, y) < 3.5


def test_multiclass_metrics():
    y = np.array([0, 1, 2, 0, 1, 2])
    probs = np.eye(3)[[0, 1, 1, 0, 1, 2]] * 0.8 + 0.2 / 3
    m = multiclass_metrics(y, probs, ["a", "b", "c"])
    assert m["per_class"]["c"]["sensitivity"] == 0.5
    assert "macro_auroc" in m
    assert len(m["confusion_matrix"]) == 3


# 20 patients x 50 near-identical slices: resampling patients gives a much wider (more honest) interval
def test_patient_bootstrap_is_wider_for_correlated_slices():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 50)
    values = np.repeat(rng.random(20), 50)

    per_slice = bootstrap_ci(lambda i: values[i].mean(), len(values), n_boot=300)
    per_patient = bootstrap_ci(lambda i: values[i].mean(), len(values), groups=groups, n_boot=300)
    assert (per_patient[1] - per_patient[0]) > 3 * (per_slice[1] - per_slice[0])
