"""Measure a trained model: on its own test split, per patient, and on a dataset it has never seen.

    # Runs automatically at the end of training. For a cross-validation folder (fold_1 ... fold_5 inside),
    # it combines the results of all folds
    python -m covid_ct_diagnosis.evaluate --model-dir 3_honest_model/results/patient_split

    # Step 2: the baseline tested on the Tehran patients, as described in its settings file
    python -m covid_ct_diagnosis.evaluate --config 2_other_hospital_test/config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .metrics import (
    binary_metrics,
    bootstrap_ci,
    calibration_error,
    fit_temperature,
    multiclass_metrics,
    softmax,
    threshold_for_sensitivity,
)
from .plots import plot_calibration, plot_confusion, plot_roc
from .utils import save_json

COVID_DIAGNOSIS = "COVID-19"


# ------------------------------------------------------------------------------------------- small helpers

# Read the settings a model was trained with
def _load_cfg(model_dir: Path) -> dict:
    with open(model_dir / "config.yaml") as f:
        return yaml.safe_load(f)


# The model's raw scores as an array: one row per slice, one column per class
def _logits(df: pd.DataFrame, classes: list[str]) -> np.ndarray:
    return df[[f"logit_{c}" for c in classes]].to_numpy(dtype=np.float64)


# Patient ID of every row, or None when the dataset has no patient IDs
def _groups(df: pd.DataFrame) -> np.ndarray | None:
    if "group" in df and df["group"].notna().all():
        return df["group"].astype(str).to_numpy()
    return None


# AUROC, or None when only one class is present
def _auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else None


# Balanced accuracy = average of the per-class sensitivities
def _balanced_accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    pred = probs.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


# Mean of the k highest values, with k = `fraction` of the slices (at least 3)
def top_k_mean(values: np.ndarray, fraction: float) -> float:
    k = max(3, math.ceil(fraction * len(values)))
    return float(np.sort(values)[-k:].mean())


# ------------------------------------------------------------------------------------------- per slice

# Measures per slice: every class (on labelled slices), and COVID yes / no (on slices with a COVID target)
# `temperature` is one number, or one number per row when results of several runs are combined
def slice_report(df: pd.DataFrame, classes: list[str], positive_class: str, temperature: float | np.ndarray = 1.0,
                 threshold: float | None = None, plots_dir: Path | None = None) -> dict:
    report: dict = {}
    logits = _logits(df, classes)
    probs = softmax(logits)
    groups = _groups(df)

    # Every class, with a confidence interval that resamples whole patients when they are known
    labelled = df["label"].isin(classes).to_numpy()
    if labelled.any():
        y = df.loc[labelled, "label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        p = probs[labelled]
        g = groups[labelled] if groups is not None else None
        t = temperature[labelled][:, None] if np.ndim(temperature) else temperature
        scaled = softmax(logits[labelled], t)

        mc = multiclass_metrics(y, p, classes)
        mc["calibration_error_after_scaling"] = calibration_error(y, scaled)
        mc["balanced_accuracy_ci95"] = bootstrap_ci(lambda i: _balanced_accuracy(y[i], p[i]), len(y), g, n_boot=300)
        report["all_classes"] = mc

        if plots_dir:
            plot_confusion(np.array(mc["confusion_matrix"]), classes, plots_dir / "confusion_slices.png")
            plot_calibration(y, p, scaled, plots_dir / "calibration_slices.png")

    # COVID yes / no, using the probability of the COVID class as the score
    has_target = df["covid"].notna().to_numpy()
    if has_target.any():
        y = df.loc[has_target, "covid"].astype(int).to_numpy()
        score = probs[has_target, classes.index(positive_class)]
        g = groups[has_target] if groups is not None else None

        covid = {"at_0.5": binary_metrics(y, score, 0.5)}
        if threshold is not None:
            covid["at_val_threshold"] = binary_metrics(y, score, threshold)
        if 0 < y.sum() < len(y):
            covid["auroc_ci95"] = bootstrap_ci(lambda i: roc_auc_score(y[i], score[i]), len(y), g, n_boot=300)
            if plots_dir:
                plot_roc({"COVID vs. not COVID": (y, score)}, plots_dir / "roc_slices.png")
        report["covid_yes_no"] = covid

    return report


# ------------------------------------------------------------------------------------------- per patient

# One row per patient: the mean of its top-k most suspicious slice probabilities, for each lesion class
def patient_features(df: pd.DataFrame, classes: list[str], background: str, fraction: float) -> pd.DataFrame:
    probs = softmax(_logits(df, classes))
    lesion_classes = [c for c in classes if c != background]

    slices = pd.DataFrame({f"p_{c}": probs[:, classes.index(c)] for c in lesion_classes})
    slices["group"] = df["group"].astype(str).to_numpy()
    slices["diagnosis"] = df["diagnosis"].to_numpy()

    rows = []
    for patient_id, patient in slices.groupby("group"):
        row = {"group": patient_id, "diagnosis": patient["diagnosis"].iloc[0], "n_slices": len(patient)}
        for c in lesion_classes:
            row[f"topk_{c}"] = top_k_mean(patient[f"p_{c}"].to_numpy(), fraction)
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


# Patient diagnosis (COVID-19 / CAP / Normal) on the test patients, learned from the val patients
def patient_report(val: pd.DataFrame, test: pd.DataFrame, classes: list[str], settings: dict,
                   out_dir: Path) -> dict:
    fraction = settings["topk_fraction"]
    features_val = patient_features(val, classes, settings["background_class"], fraction)
    features_test = patient_features(test, classes, settings["background_class"], fraction)
    columns = [c for c in features_val.columns if c.startswith("topk_")]
    report: dict = {"n_val_patients": len(features_val), "n_test_patients": len(features_test)}

    # Simple rule, nothing learned: the more suspicious a patient's top-k slices, the more likely COVID
    rule_score = features_test[f"topk_{settings['positive_class']}"].to_numpy()
    report["rule"] = _covid_vs_others(features_test["diagnosis"], rule_score)

    # Learned combination: a small logistic regression fitted on the val patients turns the top-k scores
    # into probabilities for the three diagnoses
    x_val, x_test = features_val[columns].to_numpy(), features_test[columns].to_numpy()
    combiner = _new_combiner().fit(x_val, features_val["diagnosis"].to_numpy())
    diagnoses = list(combiner.classes_)
    y_val = np.array([diagnoses.index(d) for d in features_val["diagnosis"]])
    y_test = np.array([diagnoses.index(d) for d in features_test["diagnosis"]])

    # Honest percentages: the combiner is cautious, so its percentages are stretched by one temperature.
    # The temperature is fitted on out-of-sample val percentages, never on patients the combiner learned from
    logits_val_oof = _out_of_fold_logits(x_val, y_val, diagnoses)
    temperature = fit_temperature(logits_val_oof, y_val) if logits_val_oof is not None else 1.0
    logits_test = combiner.decision_function(x_test)
    probs_raw = softmax(logits_test)
    probs_test = softmax(logits_test, temperature)
    report["calibration"] = {
        "temperature": temperature,
        "error_before": calibration_error(y_test, probs_raw),
        "error_after": calibration_error(y_test, probs_test),
    }

    diagnosis = multiclass_metrics(y_test, probs_test, diagnoses)
    diagnosis["balanced_accuracy_ci95"] = bootstrap_ci(
        lambda i: _balanced_accuracy(y_test[i], probs_test[i]), len(y_test))
    report["diagnosis"] = diagnosis

    # COVID yes / no per patient. The threshold is chosen on the out-of-sample val percentages to catch 95% of
    # COVID cases
    covid_col = diagnoses.index(COVID_DIAGNOSIS)
    score_test = probs_test[:, covid_col]
    val_probs = softmax(logits_val_oof, temperature) if logits_val_oof is not None else combiner.predict_proba(x_val)
    score_val = val_probs[:, covid_col]
    is_covid_val = (features_val["diagnosis"] == COVID_DIAGNOSIS).astype(int).to_numpy()
    is_covid_test = (features_test["diagnosis"] == COVID_DIAGNOSIS).astype(int).to_numpy()
    threshold = threshold_for_sensitivity(is_covid_val, score_val, settings["target_sensitivity"])
    report["covid_yes_no"] = {
        "at_0.5": binary_metrics(is_covid_test, score_test, 0.5),
        "at_val_threshold": binary_metrics(is_covid_test, score_test, threshold),
        "auroc_ci95": bootstrap_ci(lambda i: roc_auc_score(is_covid_test[i], score_test[i]), len(is_covid_test)),
        **_covid_vs_others(features_test["diagnosis"], score_test),
    }

    # Figures, the per-patient table, and the combiner's numbers (reused by the demo export)
    plot_confusion(np.array(diagnosis["confusion_matrix"]), diagnoses, out_dir / "plots" / "confusion_patients.png")
    plot_roc(_roc_curves_by_diagnosis(features_test["diagnosis"], score_test), out_dir / "plots" / "roc_patients.png")
    features_test.assign(
        **{f"prob_{d}": probs_test[:, i] for i, d in enumerate(diagnoses)},
        **{f"raw_prob_{d}": probs_raw[:, i] for i, d in enumerate(diagnoses)},
    ).to_csv(out_dir / "patients_test.csv")
    report["combiner"] = {"features": columns, "diagnoses": diagnoses, "coef": combiner.coef_.tolist(),
                          "intercept": combiner.intercept_.tolist(), "temperature": temperature,
                          "covid_threshold": threshold}
    return report


# The small model that turns a patient's top-k scores into a diagnosis
def _new_combiner() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")


# Out-of-sample scores for the val patients: split them into up to 5 groups, fit the combiner on the others
# and score the held-out group, in turn. None when there are too few patients of some diagnosis to split
def _out_of_fold_logits(x: np.ndarray, y: np.ndarray, diagnoses: list[str]) -> np.ndarray | None:
    n_splits = min(5, int(np.bincount(y, minlength=len(diagnoses)).min()))
    if n_splits < 2:
        return None

    logits = np.zeros((len(y), len(diagnoses)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fit_rows, held_out in splitter.split(x, y):
        part = _new_combiner().fit(x[fit_rows], y[fit_rows])
        logits[held_out] = part.decision_function(x[held_out])
    return logits


# AUROC of COVID vs. all others, and of COVID vs. each other diagnosis separately
def _covid_vs_others(diagnosis: pd.Series, score: np.ndarray) -> dict:
    is_covid = (diagnosis == COVID_DIAGNOSIS).astype(int).to_numpy()
    result = {"auroc_covid_vs_rest": _auroc(is_covid, score)}
    for other in sorted(set(diagnosis) - {COVID_DIAGNOSIS}):
        keep = diagnosis.isin([COVID_DIAGNOSIS, other]).to_numpy()
        result[f"auroc_covid_vs_{other.lower()}"] = _auroc(is_covid[keep], score[keep])
    return result


# ROC curves for COVID vs. rest and COVID vs. each other diagnosis
def _roc_curves_by_diagnosis(diagnosis: pd.Series, score: np.ndarray) -> dict:
    is_covid = (diagnosis == COVID_DIAGNOSIS).astype(int).to_numpy()
    curves = {"COVID vs. rest": (is_covid, score)}
    for other in sorted(set(diagnosis) - {COVID_DIAGNOSIS}):
        keep = diagnosis.isin([COVID_DIAGNOSIS, other]).to_numpy()
        curves[f"COVID vs. {other}"] = (is_covid[keep], score[keep])
    return curves


# ------------------------------------------------------------------------------------------- entry points

# Evaluate a trained model on its own test split (called at the end of training)
def evaluate_run(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    cfg = _load_cfg(model_dir)
    classes, positive_class = cfg["classes"], cfg["positive_class"]
    val = pd.read_csv(model_dir / "preds_val.csv", dtype={"group": "string"})
    test = pd.read_csv(model_dir / "preds_test.csv", dtype={"group": "string"})

    # Calibration: fit the temperature on the labelled val slices
    labelled = val["label"].isin(classes).to_numpy()
    y_val = val.loc[labelled, "label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    temperature = fit_temperature(_logits(val[labelled], classes), y_val)

    # Threshold that catches 95% of COVID slices on val (only used to report sensitivity at that point)
    threshold = None
    has_target = val["covid"].notna().to_numpy()
    y_val_covid = val.loc[has_target, "covid"].astype(int).to_numpy()
    if 0 < y_val_covid.sum() < len(y_val_covid):
        score_val = softmax(_logits(val[has_target], classes))[:, classes.index(positive_class)]
        threshold = threshold_for_sensitivity(y_val_covid, score_val, cfg["eval"]["target_sensitivity"])

    results = {
        "name": cfg["name"],
        "split": cfg["data"]["split"],
        "temperature": temperature,
        "covid_threshold_from_val": threshold,
        "test_slices": slice_report(test, classes, positive_class, temperature, threshold, model_dir / "plots"),
    }

    patient_settings = cfg["eval"].get("patient_level")
    if patient_settings and patient_settings.get("enabled"):
        settings = {**patient_settings, "background_class": cfg["eval"]["background_class"],
                    "positive_class": positive_class}
        results["test_patients"] = patient_report(val, test, classes, settings, model_dir)

    # metrics.json is written last: its presence means the run is complete (cross-validation relies on this)
    save_json({"temperature": temperature, "threshold": threshold}, model_dir / "calibration.json")
    save_json(results, model_dir / "metrics.json")
    print_summary(results)
    return results


# Evaluate a trained model on another dataset it has never seen, optionally on a subset (`query`)
def evaluate_on_dataset(model_dir: str | Path, index_csv: str | Path, out_dir: str | Path, query: str | None = None,
                        per_patient: bool = True, device: str = "auto", name: str | None = None) -> dict:
    from torch.utils.data import DataLoader

    from .data import SliceDataset, build_transforms, load_index
    from .models import load_model, predict_logits
    from .train import save_predictions
    from .utils import pick_device

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg(model_dir)
    classes, positive_class = cfg["classes"], cfg["positive_class"]
    calibration = json.loads((model_dir / "calibration.json").read_text())

    # Predict every slice of the dataset (or of the chosen subset)
    df = load_index(index_csv)
    if query:
        df = df.query(query).copy()
    dev = pick_device(device)
    model, _ = load_model(model_dir / "model.pt", dev)
    dataset = SliceDataset(df, classes, build_transforms(cfg["data"]["img_size"], train=False))
    loader = DataLoader(dataset, batch_size=64, num_workers=cfg["train"]["num_workers"])
    preds = save_predictions(df, predict_logits(model, loader, dev), classes, out_dir / "preds.csv")

    results = {
        "name": name or cfg["name"], "model": cfg["name"], "tested_on": str(index_csv), "query": query,
        "slices": slice_report(preds, classes, positive_class, calibration["temperature"], calibration["threshold"],
                               out_dir / "plots"),
    }

    # Per patient when the dataset has patients: score = mean of the top-k COVID slice probabilities
    if per_patient and preds["group"].notna().all() and preds["diagnosis"].notna().all():
        probs = softmax(_logits(preds, classes))[:, classes.index(positive_class)]
        per_slice = pd.DataFrame({"group": preds["group"].astype(str), "diagnosis": preds["diagnosis"], "p": probs})
        per_patient = per_slice.groupby("group").agg(
            diagnosis=("diagnosis", "first"), score=("p", lambda s: top_k_mean(s.to_numpy(), 0.1)))
        results["patients"] = {"n_patients": len(per_patient),
                               **_covid_vs_others(per_patient["diagnosis"], per_patient["score"].to_numpy())}
        plot_roc(_roc_curves_by_diagnosis(per_patient["diagnosis"], per_patient["score"].to_numpy()),
                 out_dir / "plots" / "roc_patients.png")

    save_json(results, out_dir / "metrics.json")
    print_summary(results)
    return results


# Run every test listed in a settings file (used by Step 2): one results folder per test
def evaluate_from_config(config_path: str | Path, device: str = "auto") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    all_results = {}
    for test in cfg["tests"]:
        all_results[test["name"]] = evaluate_on_dataset(
            cfg["model_dir"], cfg["dataset"], Path(cfg["output_dir"]) / test["name"],
            query=test.get("query"), per_patient=test.get("per_patient", True), device=device,
            name=f"{cfg['name']} / {test['name']}",
        )
    return all_results


# ------------------------------------------------------------------------------------------- cross-validation

# Sensitivity, specificity, PPV, NPV and accuracy from added-up tp / fn / fp / tn counts
def _metrics_from_counts(counts: dict) -> dict:
    tp, fn, fp, tn = counts["tp"], counts["fn"], counts["fp"], counts["tn"]
    return {
        "n": tp + fn + fp + tn,
        "accuracy": (tp + tn) / max(tp + fn + fp + tn, 1),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "confusion": counts,
    }


# Add up the confusion counts that each fold got with its own validation threshold
def _pooled_threshold_results(fold_metrics: list[dict], *keys: str) -> dict | None:
    counts = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for m in fold_metrics:
        node = m
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return None
        for name in counts:
            counts[name] += node["confusion"][name]
    return _metrics_from_counts(counts)


# Slices that show an infection: AUROC of "COVID lesion" vs. "other lesion" (pneumonia), like Step 2's second test
def lesion_report(df: pd.DataFrame, classes: list[str], positive_class: str, background: str) -> dict:
    lesions = df[df["label"].isin([c for c in classes if c != background])]
    y = (lesions["label"] == positive_class).astype(int).to_numpy()
    score = softmax(_logits(lesions, classes))[:, classes.index(positive_class)]
    groups = _groups(lesions)
    result = {"n": len(lesions), "n_covid": int(y.sum()), "auroc": _auroc(y, score)}
    if result["auroc"] is not None:
        result["auroc_ci95"] = bootstrap_ci(lambda i: roc_auc_score(y[i], score[i]), len(y), groups, n_boot=300)
    return result


# Mean and spread (standard deviation) of each number across the folds
def _per_fold_summary(fold_metrics: list[dict], fold_lesions: list[dict], classes: list[str]) -> dict:
    wanted = {
        "slice_accuracy": ("test_slices", "all_classes", "accuracy"),
        "slice_balanced_accuracy": ("test_slices", "all_classes", "balanced_accuracy"),
        "slice_covid_auroc": ("test_slices", "covid_yes_no", "at_0.5", "auroc"),
        **{f"slice_sensitivity_{c}": ("test_slices", "all_classes", "per_class", c, "sensitivity") for c in classes},
        "patient_balanced_accuracy": ("test_patients", "diagnosis", "balanced_accuracy"),
        "patient_auroc_covid_vs_rest": ("test_patients", "covid_yes_no", "auroc_covid_vs_rest"),
        "patient_auroc_covid_vs_cap": ("test_patients", "covid_yes_no", "auroc_covid_vs_cap"),
        "patient_auroc_covid_vs_normal": ("test_patients", "covid_yes_no", "auroc_covid_vs_normal"),
        "patient_temperature": ("test_patients", "calibration", "temperature"),
    }

    def get(m: dict, keys: tuple):
        for key in keys:
            m = m.get(key) if isinstance(m, dict) else None
        return m

    summary = {}
    for name, keys in wanted.items():
        values = [get(m, keys) for m in fold_metrics]
        if all(v is not None for v in values):
            summary[name] = {"values": values, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}

    lesion_values = [r["auroc"] for r in fold_lesions]
    if all(v is not None for v in lesion_values):
        summary["lesion_auroc_covid_vs_other"] = {
            "values": lesion_values, "mean": float(np.mean(lesion_values)), "std": float(np.std(lesion_values, ddof=1))}
    return summary


# Patient results of all folds together: every patient once, each judged by the fold that tested it
def pooled_patient_report(patients: pd.DataFrame, fold_metrics: list[dict], positive_class: str, plots_dir: Path) -> dict:
    diagnoses = sorted(c.removeprefix("prob_") for c in patients.columns if c.startswith("prob_"))
    probs = patients[[f"prob_{d}" for d in diagnoses]].to_numpy()
    probs_raw = patients[[f"raw_prob_{d}" for d in diagnoses]].to_numpy()
    y = np.array([diagnoses.index(d) for d in patients["diagnosis"]])

    diagnosis = multiclass_metrics(y, probs, diagnoses)
    diagnosis["balanced_accuracy_ci95"] = bootstrap_ci(lambda i: _balanced_accuracy(y[i], probs[i]), len(y))

    is_covid = (patients["diagnosis"] == COVID_DIAGNOSIS).astype(int).to_numpy()
    score = patients[f"prob_{COVID_DIAGNOSIS}"].to_numpy()
    rule_score = patients[f"topk_{positive_class}"].to_numpy()
    report = {
        "n_patients": len(patients),
        "calibration": {
            "error_before": calibration_error(y, probs_raw),
            "error_after": calibration_error(y, probs),
            "mean_confidence_before": float(probs_raw.max(1).mean()),
            "mean_confidence_after": float(probs.max(1).mean()),
            "accuracy": float((probs.argmax(1) == y).mean()),
        },
        "rule": _covid_vs_others(patients["diagnosis"], rule_score),
        "diagnosis": diagnosis,
        "covid_yes_no": {
            "at_0.5": binary_metrics(is_covid, score, 0.5),
            "at_val_threshold": _pooled_threshold_results(fold_metrics, "test_patients", "covid_yes_no", "at_val_threshold"),
            "auroc_ci95": bootstrap_ci(lambda i: roc_auc_score(is_covid[i], score[i]), len(is_covid)),
            **_covid_vs_others(patients["diagnosis"], score),
        },
    }
    plot_confusion(np.array(diagnosis["confusion_matrix"]), diagnoses, plots_dir / "confusion_patients.png")
    plot_roc(_roc_curves_by_diagnosis(patients["diagnosis"], score), plots_dir / "roc_patients.png")
    plot_calibration(y, probs_raw, probs, plots_dir / "calibration_patients.png")
    return report


# Combine the folds of a cross-validation: the 5 test sets together = every slice (and patient) tested once.
# With refresh_folds, each fold is evaluated again first from its saved predictions (no retraining)
def evaluate_cross_validation(cv_dir: str | Path, refresh_folds: bool = False) -> dict:
    cv_dir = Path(cv_dir)
    cfg = _load_cfg(cv_dir)
    if refresh_folds:
        for fold_dir in sorted(cv_dir.glob("fold_*")):
            evaluate_run(fold_dir)
    classes, positive_class = cfg["classes"], cfg["positive_class"]
    background = cfg["eval"]["background_class"]
    fold_dirs = sorted(cv_dir.glob("fold_*"), key=lambda d: int(d.name.split("_")[1]))
    if len(fold_dirs) != cfg["data"]["cv_folds"] or not all((d / "metrics.json").exists() for d in fold_dirs):
        raise FileNotFoundError(f"Not every fold of {cv_dir} has finished")

    # Every fold's test predictions, with that fold's calibration temperature on each row
    tests, temperatures, fold_metrics, fold_lesions = [], [], [], []
    for number, fold_dir in enumerate(fold_dirs, start=1):
        test = pd.read_csv(fold_dir / "preds_test.csv", dtype={"group": "string"}).assign(fold=number)
        calibration = json.loads((fold_dir / "calibration.json").read_text())
        tests.append(test)
        temperatures.append(np.full(len(test), calibration["temperature"]))
        fold_metrics.append(json.loads((fold_dir / "metrics.json").read_text()))
        fold_lesions.append(lesion_report(test, classes, positive_class, background))
    pooled = pd.concat(tests, ignore_index=True)

    # Per slice: all folds together. With per-fold thresholds, the counts of each fold are added up
    plots_dir = cv_dir / "plots"
    test_slices = slice_report(pooled, classes, positive_class, np.concatenate(temperatures), None, plots_dir)
    if "covid_yes_no" in test_slices:
        test_slices["covid_yes_no"]["at_val_threshold"] = _pooled_threshold_results(
            fold_metrics, "test_slices", "covid_yes_no", "at_val_threshold")

    results = {
        "name": cfg["name"],
        "split": cfg["data"]["split"],
        "cv_folds": cfg["data"]["cv_folds"],
        "pooled": {
            "test_slices": test_slices,
            "covid_vs_other_lesions": lesion_report(pooled, classes, positive_class, background),
        },
    }

    # Per patient (split by patient only): every patient once, with the diagnosis from the fold that tested it
    patient_files = [d / "patients_test.csv" for d in fold_dirs]
    if all(f.exists() for f in patient_files):
        patients = pd.concat(
            [pd.read_csv(f).assign(fold=number) for number, f in enumerate(patient_files, start=1)], ignore_index=True)
        patients.to_csv(cv_dir / "patients.csv", index=False)
        results["pooled"]["test_patients"] = pooled_patient_report(patients, fold_metrics, positive_class, plots_dir)

    results["per_fold"] = _per_fold_summary(fold_metrics, fold_lesions, classes)
    save_json(results, cv_dir / "metrics.json")
    print_cv_summary(results)
    return results


# ------------------------------------------------------------------------------------------- printing

# Print the key numbers of a results dict in a readable way
def print_summary(results: dict) -> None:

    def num(x) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    def ci(pair) -> str:
        return "n/a" if not pair else f"[{pair[0]:.3f}, {pair[1]:.3f}]"

    title = f"{results['name']}" + (f" tested on {results['tested_on']}" if "tested_on" in results else "")
    print("\n" + "=" * 80 + f"\n{title}")

    slices = results.get("test_slices") or results.get("slices") or {}
    if "all_classes" in slices:
        mc = slices["all_classes"]
        print(f"SLICES  accuracy {num(mc['accuracy'])} | balanced accuracy {num(mc['balanced_accuracy'])} "
              f"95% CI {ci(mc['balanced_accuracy_ci95'])}")
        print("        sensitivity per class: " + ", ".join(
            f"{c} {num(v['sensitivity'])}" for c, v in mc["per_class"].items()))
        print(f"        calibration error {num(mc['calibration_error'])} -> "
              f"{num(mc['calibration_error_after_scaling'])} after temperature scaling")
    if "covid_yes_no" in slices:
        b = slices["covid_yes_no"]
        print(f"SLICES  COVID yes/no: AUROC {num(b['at_0.5'].get('auroc'))} 95% CI {ci(b.get('auroc_ci95'))} | "
              f"sensitivity {num(b['at_0.5']['sensitivity'])}, specificity {num(b['at_0.5']['specificity'])} at 0.5 | "
              f"sensitivity at 95% specificity {num(b['at_0.5'].get('sensitivity_at_95_specificity'))}")

    if "test_patients" in results:
        p = results["test_patients"]
        d = p["diagnosis"]
        print(f"PATIENTS ({p['n_test_patients']} test patients)  diagnosis balanced accuracy "
              f"{num(d['balanced_accuracy'])} 95% CI {ci(d['balanced_accuracy_ci95'])}")
        print("        sensitivity per diagnosis: " + ", ".join(
            f"{c} {num(v['sensitivity'])}" for c, v in d["per_class"].items()))
        cv = p["covid_yes_no"]
        print(f"        COVID yes/no AUROC {num(cv['auroc_covid_vs_rest'])} 95% CI {ci(cv['auroc_ci95'])} | "
              f"COVID vs. CAP {num(cv.get('auroc_covid_vs_cap'))} | COVID vs. Normal {num(cv.get('auroc_covid_vs_normal'))}")

    if "patients" in results:
        p = results["patients"]
        print(f"PATIENTS ({p['n_patients']})  " + ", ".join(
            f"{k.replace('auroc_', 'AUROC ').replace('_', ' ')} {num(v)}" for k, v in p.items() if k.startswith("auroc")))
    print("=" * 80)


# Print the key numbers of a cross-validation: all folds together, then the spread from fold to fold
def print_cv_summary(results: dict) -> None:

    def num(x) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    def ci(pair) -> str:
        return "n/a" if not pair else f"[{pair[0]:.3f}, {pair[1]:.3f}]"

    pooled, per_fold = results["pooled"], results["per_fold"]
    print("\n" + "=" * 80)
    print(f"{results['name']}: {results['cv_folds']}-fold cross-validation, split = {results['split']}")
    print("ALL FOLDS TOGETHER (every slice / patient tested once)")

    mc = pooled["test_slices"].get("all_classes", {})
    if mc:
        print(f"SLICES  accuracy {num(mc['accuracy'])} | balanced accuracy {num(mc['balanced_accuracy'])} "
              f"95% CI {ci(mc['balanced_accuracy_ci95'])}")
        print("        sensitivity per class: " + ", ".join(
            f"{c} {num(v['sensitivity'])}" for c, v in mc["per_class"].items()))
    covid = pooled["test_slices"].get("covid_yes_no")
    if covid:
        print(f"SLICES  COVID yes/no AUROC {num(covid['at_0.5'].get('auroc'))} 95% CI {ci(covid.get('auroc_ci95'))} | "
              f"sensitivity {num(covid['at_0.5']['sensitivity'])}, specificity {num(covid['at_0.5']['specificity'])} at 0.5")
    lesions = pooled["covid_vs_other_lesions"]
    print(f"SLICES  COVID lesion vs. pneumonia lesion AUROC {num(lesions['auroc'])} 95% CI {ci(lesions.get('auroc_ci95'))}")

    if "test_patients" in pooled:
        p = pooled["test_patients"]
        d, cv = p["diagnosis"], p["covid_yes_no"]
        print(f"PATIENTS ({p['n_patients']})  diagnosis balanced accuracy {num(d['balanced_accuracy'])} "
              f"95% CI {ci(d['balanced_accuracy_ci95'])}")
        print("        sensitivity per diagnosis: " + ", ".join(
            f"{c} {num(v['sensitivity'])}" for c, v in d["per_class"].items()))
        print(f"        COVID yes/no AUROC {num(cv['auroc_covid_vs_rest'])} 95% CI {ci(cv['auroc_ci95'])} | "
              f"COVID vs. CAP {num(cv.get('auroc_covid_vs_cap'))} | COVID vs. Normal {num(cv.get('auroc_covid_vs_normal'))}")
        cal = p["calibration"]
        print(f"        percentages: right for {cal['accuracy']:.0%} of patients, average confidence "
              f"{cal['mean_confidence_before']:.0%} -> {cal['mean_confidence_after']:.0%} | calibration error "
              f"{num(cal['error_before'])} -> {num(cal['error_after'])}")
        rule = p["rule"]
        print(f"        same rule as Step 2 (top-k slices): COVID vs. CAP {num(rule.get('auroc_covid_vs_cap'))} | "
              f"COVID vs. Normal {num(rule.get('auroc_covid_vs_normal'))}")

    print("FOLD TO FOLD (mean ± spread)")
    for name, v in per_fold.items():
        print(f"        {name}: {v['mean']:.3f} ± {v['std']:.3f}")
    print("=" * 80)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="settings file listing the tests to run (e.g. 2_other_hospital_test/config.yaml)")
    parser.add_argument("--model-dir", type=Path, help="folder with model.pt and config.yaml")
    parser.add_argument("--dataset", type=Path, help="index.csv of another dataset to test on")
    parser.add_argument("--out", type=Path, help="where to save the results of --dataset")
    parser.add_argument("--query", help="pandas query to test on a subset, e.g. \"diagnosis != 'CAP'\"")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--refresh-folds", action="store_true",
                        help="cross-validation folder: evaluate every fold again from its saved predictions first")
    args = parser.parse_args(argv)

    if args.config:
        evaluate_from_config(args.config, args.device)
    elif args.model_dir and (args.model_dir / "fold_1").exists():
        evaluate_cross_validation(args.model_dir, refresh_folds=args.refresh_folds)
    elif not args.model_dir:
        parser.error("give --config, or --model-dir")
    elif args.dataset:
        if not args.out:
            parser.error("--out is required with --dataset")
        evaluate_on_dataset(args.model_dir, args.dataset, args.out, args.query, device=args.device)
    else:
        evaluate_run(args.model_dir)


if __name__ == "__main__":
    main()
