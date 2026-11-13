"""Export the demo patients for the web page (Step 4).

Each patient is run through the model of its own fold, the one that never saw it in training. The results are
saved as small JSON files and images in 4_demo/public/data/. The page only replays them: nothing runs in the
browser.

    python -m covid_ct_diagnosis.export_demo --config 4_demo/config.yaml

Written for each patient, in <output_dir>/<patient>/:
    patient.json         scores of every slice, the bars of the score strip, the diagnosis, the top slices
    frames.webp          one small image per bar, side by side, replayed while the scan is read
    slice_<n>.webp       each of the most suspicious slices (n counts from 1, top of the chest first)
    slice_<n>_heat.webp  its Grad-CAM heatmap, a transparent layer to put over the slice
And once: index.json (the patients) and summary.json (the numbers of the story and results sections).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
from PIL import Image

from .data import build_transforms, load_index
from .evaluate import top_k_mean
from .metrics import softmax
from .models import grad_cam, load_model
from .utils import load_config, save_json

# The recomputed scores must match the ones saved during the evaluation, up to small float differences
LOGIT_TOLERANCE = 0.05
PERCENT_TOLERANCE = 0.01


# Round every float of a list or array, to keep the JSON files small. Diagnosis percentages keep 4 decimals,
# so the page can round them to whole percents without rounding twice (0.9647 -> 96%, not 0.965 -> 97%)
def _round(values, digits: int = 3) -> list:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


# ------------------------------------------------------------------------------------------- diagnosis

# Diagnosis percentages of a patient from its slice probabilities, with the combiner saved by its fold.
# Same steps as the evaluation: mean of the top-k slices per lesion class, logistic regression, temperature
def diagnose(probs: np.ndarray, classes: list[str], combiner: dict, fraction: float) -> np.ndarray:
    features = [top_k_mean(probs[:, classes.index(name.removeprefix("topk_"))], fraction)
                for name in combiner["features"]]
    logits = np.asarray(combiner["coef"]) @ np.asarray(features) + np.asarray(combiner["intercept"])
    return softmax(logits[None, :], combiner["temperature"])[0]


# The score strip: the slices cut into `n_bars` groups, top of the chest first. Each bar keeps the highest score
# of each lesion class and the diagnosis from the slices read so far
def strip_bars(probs: np.ndarray, classes: list[str], lesion_classes: list[str], n_bars: int,
               combiner: dict, fraction: float) -> list[dict]:
    edges = np.linspace(0, len(probs), min(n_bars, len(probs)) + 1).round().astype(int)
    bars = []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        bars.append({
            "first": int(start) + 1,
            "last": int(end),
            "scores": {c: round(float(probs[start:end, classes.index(c)].max()), 3) for c in lesion_classes},
            "reading": _round(diagnose(probs[:end], classes, combiner, fraction), 4),
        })
    return bars


# The most suspicious slices: highest score of the lesion class that matches the diagnosis. When the diagnosis
# has no lesion class (healthy), the highest score of any lesion class. Returns (slice row, class) pairs
def top_slices(probs: np.ndarray, classes: list[str], lesion_classes: list[str], diagnosis_class: str | None,
               k: int) -> list[tuple[int, str]]:
    columns = [classes.index(c) for c in lesion_classes]
    if diagnosis_class:
        score, chosen = probs[:, classes.index(diagnosis_class)], np.full(len(probs), diagnosis_class)
    else:
        score = probs[:, columns].max(1)
        chosen = np.array(lesion_classes)[probs[:, columns].argmax(1)]
    rows = np.argsort(-score, kind="stable")[:k]
    return [(int(r), str(chosen[r])) for r in rows]


# ------------------------------------------------------------------------------------------- images

# A heatmap as a transparent colour layer: warm where the model looked, see-through where it didn't
def heatmap_layer(cam: np.ndarray, size: int) -> Image.Image:
    cam = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255
    rgba = colormaps["inferno"](cam)
    rgba[..., 3] = np.clip((cam - 0.15) / 0.85, 0, 1) * 0.85
    return Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")


# Several slices side by side in one image (a sprite sheet), so the page loads a single file
def frames_sheet(paths: list[str], size: int, columns: int) -> Image.Image:
    rows = -(-len(paths) // columns)
    sheet = Image.new("L", (columns * size, rows * size))
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            frame = img.convert("L").resize((size, size), Image.LANCZOS)
        sheet.paste(frame, ((i % columns) * size, (i // columns) * size))
    return sheet


# ------------------------------------------------------------------------------------------- one patient

# The slices of a patient, top of the chest first, with the scores its fold saved during the evaluation
def _patient_slices(index: pd.DataFrame, fold_dir: Path, patient_id: str, classes: list[str]) -> pd.DataFrame:
    slices = index[index["group"] == patient_id].sort_values("order").reset_index(drop=True)
    saved = pd.read_csv(fold_dir / "preds_test.csv", dtype={"group": "string"})
    saved = saved[saved["group"] == patient_id][["order", *[f"logit_{c}" for c in classes]]]
    slices = slices.merge(saved, on="order", how="left", validate="one_to_one")
    if slices.empty or slices[f"logit_{classes[0]}"].isna().any():
        raise ValueError(f"{patient_id} is not in the test predictions of {fold_dir}")
    return slices


# Scores of every slice, recomputed with the model of the fold
@torch.no_grad()
def _slice_logits(model, slices: pd.DataFrame, transform, batch_size: int = 64) -> np.ndarray:
    logits = []
    for start in range(0, len(slices), batch_size):
        images = []
        for path in slices["path"].iloc[start:start + batch_size]:
            with Image.open(path) as img:
                images.append(transform(img.convert("L")))
        logits.append(model(torch.stack(images)).numpy())
    return np.concatenate(logits)


# Run one patient through the model of its own fold, check it against the saved results, write its files
def export_patient(patient_id: str, patient: pd.Series, index: pd.DataFrame, cfg: dict, out_dir: Path,
                   models: dict) -> dict:
    fold = int(patient["fold"])
    fold_dir = Path(cfg["cv_dir"]) / f"fold_{fold}"
    if fold not in models:
        models[fold] = load_model(fold_dir / "model.pt", "cpu")
    model, model_cfg = models[fold]

    classes = model_cfg["classes"]
    background = model_cfg["eval"]["background_class"]
    lesion_classes = [c for c in classes if c != background]
    fraction = model_cfg["eval"]["patient_level"]["topk_fraction"]
    combiner = json.loads((fold_dir / "metrics.json").read_text())["test_patients"]["combiner"]
    diagnoses = combiner["diagnoses"]
    transform = build_transforms(model_cfg["data"]["img_size"], train=False)

    # The model gives the same scores as during the evaluation, and the same diagnosis percentages
    slices = _patient_slices(index, fold_dir, patient_id, classes)
    logits = _slice_logits(model, slices, transform)
    saved = slices[[f"logit_{c}" for c in classes]].to_numpy()
    if np.abs(logits - saved).max() > LOGIT_TOLERANCE:
        raise AssertionError(f"{patient_id}: the fold {fold} model doesn't reproduce the saved scores")
    probs = softmax(logits)
    percents = diagnose(probs, classes, combiner, fraction)
    expected = patient[[f"prob_{d}" for d in diagnoses]].to_numpy(dtype=float)
    if np.abs(percents - expected).max() > PERCENT_TOLERANCE:
        raise AssertionError(f"{patient_id}: diagnosis {percents} differs from patients.csv {expected}")
    predicted = diagnoses[int(percents.argmax())]

    page = cfg["page"]
    folder = out_dir / patient_id
    folder.mkdir(parents=True, exist_ok=True)
    bars = strip_bars(probs, classes, lesion_classes, page["bars"], combiner, fraction)
    # The frame replayed for each bar is its middle slice
    frame_paths = [slices["path"].iloc[(bar["first"] + bar["last"]) // 2 - 1] for bar in bars]
    frames_sheet(frame_paths, page["frame_size"], page["frame_columns"]).save(
        folder / "frames.webp", "WEBP", quality=page["image_quality"])

    # The most suspicious slices: the slice itself and its heatmap
    top = []
    diagnosis_class = cfg["diagnosis_classes"].get(predicted)
    for row, lesion_class in top_slices(probs, classes, lesion_classes, diagnosis_class, page["top_slices"]):
        n = row + 1
        with Image.open(slices["path"].iloc[row]) as img:
            img = img.convert("L")
            size = img.width
            img.save(folder / f"slice_{n}.webp", "WEBP", quality=page["image_quality"])
            tensor = transform(img).unsqueeze(0)
        cam, _ = grad_cam(model, tensor, classes.index(lesion_class))
        heatmap_layer(cam, size).save(folder / f"slice_{n}_heat.webp", "WEBP", quality=page["image_quality"])
        label = slices["label"].iloc[row]
        top.append({
            "slice": n,
            "score": round(float(probs[row, classes.index(lesion_class)]), 3),
            "radiologist_label": label if isinstance(label, str) else None,
            "image": f"{patient_id}/slice_{n}.webp",
            "heatmap": f"{patient_id}/slice_{n}_heat.webp",
        })

    result = {
        "id": patient_id,
        "diagnosis": patient["diagnosis"],
        "n_slices": len(slices),
        "slice_classes": classes,
        "diagnoses": diagnoses,
        "prediction": {"diagnosis": predicted, "percents": _round(percents, 4)},
        "slices": [{"slice": i + 1, "scores": _round(p)} for i, p in enumerate(probs)],
        "bars": bars,
        "frames": {"image": f"{patient_id}/frames.webp", "columns": page["frame_columns"], "count": len(bars)},
        "top_slices": top,
    }
    save_json(result, folder / "patient.json")
    print(f"{patient_id}: {patient['diagnosis']}, model says {predicted} {percents.max():.0%} (fold {fold})")
    return {"id": patient_id, "diagnosis": patient["diagnosis"], "n_slices": len(slices),
            "thumbnail": top[0]["image"], "data": f"{patient_id}/patient.json"}


# ------------------------------------------------------------------------------------------- summary

# The numbers shown in the story and results sections, read from each step's results (none typed by hand)
def results_summary(paths: dict) -> dict:
    def read(key: str) -> dict:
        return json.loads(Path(paths[key]).read_text())

    step1, step2 = read("step1"), read("step2")
    random_split, patient_split = read("step3_random")["pooled"], read("step3_patient")["pooled"]
    patients = patient_split["test_patients"]
    diagnosis = patients["diagnosis"]
    diagnoses = list(diagnosis["per_class"])
    confusion = np.array(diagnosis["confusion_matrix"])
    screening = patients["covid_yes_no"]["at_val_threshold"]["confusion"]

    return {
        "step1": {
            "accuracy": round(step1["test_slices"]["all_classes"]["accuracy"], 3),
            "n_slices": len(pd.read_csv(paths["step1_index"])),
        },
        "step2": {
            "covid_slices_missed": round(1 - step2["slices"]["covid_yes_no"]["at_0.5"]["sensitivity"], 3),
        },
        "step3": {
            "random_split": round(random_split["test_slices"]["all_classes"]["balanced_accuracy"], 3),
            "patient_split": round(patient_split["test_slices"]["all_classes"]["balanced_accuracy"], 3),
        },
        "honest_model": {
            "n_patients": patients["n_patients"],
            "balanced_accuracy": round(diagnosis["balanced_accuracy"], 3),
            "balanced_accuracy_ci95": _round(diagnosis["balanced_accuracy_ci95"]),
            "diagnoses": diagnoses,
            "confusion_matrix": confusion.tolist(),
            "per_diagnosis": {d: {"patients": int(confusion[i].sum()), "correct": int(confusion[i, i])}
                              for i, d in enumerate(diagnoses)},
            "screening": {
                "covid_caught": screening["tp"],
                "covid_patients": screening["tp"] + screening["fn"],
                "false_alarms": screening["fp"],
                "other_patients": screening["fp"] + screening["tn"],
            },
        },
    }


# ------------------------------------------------------------------------------------------- entry point

# Export every demo patient listed in the settings file, then the index and the summary numbers
def export_demo(cfg: dict) -> Path:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(cfg["index"])
    patients = pd.read_csv(Path(cfg["cv_dir"]) / "patients.csv", dtype={"group": "string"}).set_index("group")

    models: dict = {}
    exported = []
    for patient_id in cfg["patients"]:
        if patient_id not in patients.index:
            raise ValueError(f"{patient_id} is not in {cfg['cv_dir']}/patients.csv")
        exported.append(export_patient(patient_id, patients.loc[patient_id], index, cfg, out_dir, models))

    save_json({"patients": exported}, out_dir / "index.json")
    if cfg.get("summary"):
        save_json(results_summary(cfg["summary"]), out_dir / "summary.json")
    print(f"saved {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("overrides", nargs="*", help="settings to change, e.g. page.bars=30")
    args = parser.parse_args(argv)
    export_demo(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
