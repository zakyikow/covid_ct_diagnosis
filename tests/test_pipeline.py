"""End-to-end tests on fake data: one run, a cross-validation, testing on another dataset, Grad-CAM, report,
and the export of demo patients."""

import json

import pandas as pd
import torch
import yaml

from covid_ct_diagnosis.evaluate import evaluate_cross_validation, evaluate_from_config, evaluate_on_dataset
from covid_ct_diagnosis.explain import gradcam_grid, label_sheet
from covid_ct_diagnosis.export_demo import export_demo
from covid_ct_diagnosis.models import build_model, grad_cam
from covid_ct_diagnosis.report import collect, to_markdown
from covid_ct_diagnosis.train import train


# Real settings, made tiny and fast: no pretrained weights, small images, CPU.
# Without `folds` it's a single run (like Step 1), with `folds` a cross-validation (like Step 3)
def tiny_config(index, output_dir, path="3_honest_model/patient_split.yaml", folds=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if folds:
        cfg["data"]["cv_folds"] = folds
    else:
        cfg["data"].pop("cv_folds", None)
        cfg["data"]["fractions"] = [0.6, 0.2, 0.2]
    cfg["output_dir"] = str(output_dir)
    cfg["data"]["index"] = str(index)
    cfg["data"]["img_size"] = 64
    cfg["model"]["pretrained"] = False
    cfg["train"].update(epochs=2, batch_size=16, num_workers=0, device="cpu")
    return cfg


def test_train_evaluate_explain(tehran_like, tmp_path):
    out = train(tiny_config(tehran_like, tmp_path / "3_honest_model" / "results" / "patient_split"))

    expected = ["model.pt", "config.yaml", "history.csv", "split_patients.csv", "preds_val.csv", "preds_test.csv",
                "metrics.json", "calibration.json", "patients_test.csv",
                "plots/confusion_slices.png", "plots/roc_patients.png"]
    for name in expected:
        assert (out / name).exists(), name

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["test_patients"]["n_test_patients"] > 0
    assert metrics["temperature"] > 0

    # Testing on another dataset (here the same fake data) gives slice and patient results
    other = evaluate_on_dataset(out, tehran_like, tmp_path / "other_dataset", device="cpu")
    assert "covid_yes_no" in other["slices"]
    assert "auroc_covid_vs_cap" in other["patients"]

    # Step 2 is described by a settings file: the real one, pointed at the fake data
    with open("2_other_hospital_test/config.yaml") as f:
        step2 = yaml.safe_load(f)
    step2.update(model_dir=str(out), dataset=str(tehran_like), output_dir=str(tmp_path / "2_other_hospital_test" / "results"))
    step2_path = tmp_path / "step2.yaml"
    step2_path.write_text(yaml.safe_dump(step2))

    results = evaluate_from_config(step2_path, device="cpu")
    assert set(results) == {"all_patients", "covid_vs_pneumonia_lesions"}
    assert "patients" in results["all_patients"]
    assert "patients" not in results["covid_vs_pneumonia_lesions"]

    assert gradcam_grid(out, per_cell=1).exists()
    assert label_sheet(tehran_like, "p009").exists()
    report = to_markdown(collect(tmp_path))
    assert "3_honest_model/results/patient_split" in report
    assert "2_other_hospital_test/results/all_patients" in report


# Cross-validation by patient: every patient tested once, combined results, resume, Grad-CAM across folds
def test_cross_validation_by_patient(tehran_like, tmp_path):
    out_dir = tmp_path / "3_honest_model" / "results" / "patient_split"
    out = train(tiny_config(tehran_like, out_dir, folds=3))

    for k in (1, 2, 3):
        assert (out / f"fold_{k}" / "metrics.json").exists()
    for name in ("config.yaml", "folds.csv", "metrics.json", "patients.csv", "plots/confusion_patients.png"):
        assert (out / name).exists(), name

    # Every patient is tested exactly once, by the fold it belongs to
    folds = pd.read_csv(out / "folds.csv")
    patients = pd.read_csv(out / "patients.csv")
    assert sorted(patients["group"]) == sorted(folds["group"])
    merged = patients.merge(folds, on="group", suffixes=("_tested", "_assigned"))
    assert (merged["fold_tested"] == merged["fold_assigned"]).all()

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["pooled"]["test_patients"]["n_patients"] == folds.shape[0]
    assert "covid_vs_other_lesions" in metrics["pooled"]
    assert len(metrics["per_fold"]["slice_balanced_accuracy"]["values"]) == 3

    # Calibrated patient percentages: same diagnoses as before calibration, and they still add up to 1
    diagnoses = ["CAP", "COVID-19", "Normal"]
    calibrated = patients[[f"prob_{d}" for d in diagnoses]].to_numpy()
    raw = patients[[f"raw_prob_{d}" for d in diagnoses]].to_numpy()
    assert (calibrated.argmax(1) == raw.argmax(1)).all()
    assert abs(calibrated.sum(1) - 1).max() < 1e-6
    assert "calibration" in metrics["pooled"]["test_patients"]

    # Re-evaluating from the saved predictions gives the same patient results
    again = evaluate_cross_validation(out, refresh_folds=True)["pooled"]["test_patients"]["diagnosis"]
    before = metrics["pooled"]["test_patients"]["diagnosis"]
    assert again["confusion_matrix"] == before["confusion_matrix"]
    assert again["balanced_accuracy"] == before["balanced_accuracy"]

    # Running it again skips the finished folds (their model files are not rewritten)
    before = (out / "fold_1" / "model.pt").stat().st_mtime_ns
    train(tiny_config(tehran_like, out_dir, folds=3))
    assert (out / "fold_1" / "model.pt").stat().st_mtime_ns == before

    assert gradcam_grid(out, per_cell=1).exists()
    report = to_markdown(collect(tmp_path))
    assert "3_honest_model/results/patient_split |" in report
    assert "fold_1" not in report

    # Demo export: the real Step 4 settings, pointed at this cross-validation and two of its patients
    with open("4_demo/config.yaml") as f:
        demo = yaml.safe_load(f)
    demo.update(cv_dir=str(out), index=str(tehran_like), output_dir=str(tmp_path / "demo"), patients=["p001", "p009"])
    demo.pop("summary")
    demo["page"].update(bars=5, top_slices=3, frame_size=32)
    demo_dir = export_demo(demo)

    assert [p["id"] for p in json.loads((demo_dir / "index.json").read_text())["patients"]] == ["p001", "p009"]
    patient = json.loads((demo_dir / "p009" / "patient.json").read_text())
    assert patient["n_slices"] == len(patient["slices"]) == 12
    assert len(patient["bars"]) == 5 and patient["bars"][-1]["last"] == 12
    assert patient["bars"][-1]["reading"] == patient["prediction"]["percents"]
    assert abs(sum(patient["prediction"]["percents"]) - 1) < 1e-2
    assert len(patient["top_slices"]) == 3
    for top in patient["top_slices"]:
        assert (demo_dir / top["image"]).exists() and (demo_dir / top["heatmap"]).exists()
    assert (demo_dir / patient["frames"]["image"]).exists()


# Cross-validation with a random split: no patient-level results, but the same slice results
def test_cross_validation_random_split(tehran_like, tmp_path):
    out_dir = tmp_path / "random_split"
    out = train(tiny_config(tehran_like, out_dir, path="3_honest_model/random_split.yaml", folds=3))
    metrics = json.loads((out / "metrics.json").read_text())
    assert "test_patients" not in metrics["pooled"]
    assert "all_classes" in metrics["pooled"]["test_slices"]
    assert not (out / "folds.csv").exists()


def test_gradcam_heatmap_shape_and_range():
    model = build_model(3, pretrained=False)
    cam, class_idx = grad_cam(model, torch.randn(1, 3, 64, 64))
    assert cam.shape == (64, 64)
    assert 0 <= cam.min() and cam.max() <= 1
    assert 0 <= class_idx < 3
