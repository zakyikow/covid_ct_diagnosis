"""Visual checks: Grad-CAM heatmaps of a trained model, and a sheet to check the Tehran slice labels.

    # Grad-CAM on test slices: correct and wrong predictions for each class. For a cross-validation folder,
    # the test slices of all folds are used, each drawn with the model of its own fold
    python -m covid_ct_diagnosis.explain gradcam --model-dir 3_honest_model/results/patient_split

    # Red border = slice labelled "infection visible" by the radiologists. They should sit where the lesions are
    python -m covid_ct_diagnosis.explain labels --index 0_data/processed/ctmd/index.csv --patient cap002
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from PIL import Image

from .data import build_transforms, load_index
from .metrics import softmax
from .models import grad_cam, load_model


# Pick the most confident correct and wrong test predictions for each true class, at most one slice per patient
def _pick_examples(preds: pd.DataFrame, classes: list[str], per_cell: int) -> list[tuple[str, list]]:
    preds = preds[preds["label"].isin(classes)].copy()
    probs = softmax(preds[[f"logit_{c}" for c in classes]].to_numpy(dtype=float))
    preds["pred"] = [classes[i] for i in probs.argmax(1)]
    preds["confidence"] = probs.max(1)

    picks = []
    for c in classes:
        rows = preds[preds["label"] == c].sort_values("confidence", ascending=False)
        right = _one_per_patient(rows[rows["pred"] == c]).head(per_cell)
        wrong = _one_per_patient(rows[rows["pred"] != c]).head(per_cell)
        right, wrong = list(right.itertuples()), list(wrong.itertuples())
        picks.append((c, right + [None] * (per_cell - len(right)) + wrong + [None] * (per_cell - len(wrong))))
    return picks


# Keep only the first (most confident) slice of each patient, so the examples aren't near-duplicates
def _one_per_patient(rows: pd.DataFrame) -> pd.DataFrame:
    if "group" not in rows or rows["group"].isna().all():
        return rows
    return rows.drop_duplicates(subset="group", keep="first")


# Test predictions with the folder of the model that made them: one run, or every fold of a cross-validation
def _test_predictions(model_dir: Path, preds_csv: Path | None) -> pd.DataFrame:
    if preds_csv:
        return pd.read_csv(preds_csv).assign(model_dir=str(model_dir))
    fold_dirs = sorted(model_dir.glob("fold_*"))
    if fold_dirs:
        return pd.concat([pd.read_csv(d / "preds_test.csv").assign(model_dir=str(d)) for d in fold_dirs],
                         ignore_index=True)
    return pd.read_csv(model_dir / "preds_test.csv").assign(model_dir=str(model_dir))


# Grid of Grad-CAM heatmaps: one row per true class, correct predictions on the left, wrong ones on the right
def gradcam_grid(model_dir: Path, preds_csv: Path | None = None, per_cell: int = 3, out: Path | None = None) -> Path:
    model_dir = Path(model_dir)
    with open(model_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    classes = cfg["classes"]

    picks = _pick_examples(_test_predictions(model_dir, preds_csv), classes, per_cell)
    transform = build_transforms(cfg["data"]["img_size"], train=False)

    # Each example is drawn with the model that predicted it (loaded once per model)
    models = {}

    def model_for(example):
        if example.model_dir not in models:
            models[example.model_dir], _ = load_model(Path(example.model_dir) / "model.pt", "cpu")
        return models[example.model_dir]

    n_cols = 2 * per_cell
    fig = Figure(figsize=(2.2 * n_cols, 2.4 * len(classes)))
    axes = fig.subplots(len(classes), n_cols, squeeze=False)

    for r, (true_class, examples) in enumerate(picks):
        for c, example in enumerate(examples):
            ax = axes[r, c]
            ax.axis("off")
            if example is None:
                continue

            # Heatmap for the class the model predicted, drawn over the slice
            img = Image.open(example.path).convert("L")
            cam, _ = grad_cam(model_for(example), transform(img).unsqueeze(0), classes.index(example.pred))
            ax.imshow(np.asarray(img.resize(cam.shape[::-1]), dtype=float) / 255, cmap="gray")
            ax.imshow(cam, cmap="jet", alpha=0.35)

            correct = example.pred == true_class
            ax.set_title(f"{'correct' if correct else 'wrong'}: true {true_class}\n"
                         f"predicted {example.pred} ({example.confidence:.2f})",
                         fontsize=7, color="green" if correct else "red")

    fig.suptitle(f"Grad-CAM, {cfg['name']}: correct (left) and wrong (right) predictions", fontsize=10)
    fig.tight_layout()
    out = Path(out or model_dir / "plots" / "gradcam.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    return out


# Contact sheet of all slices of one patient, with the radiologists' lesion slices framed in red
def label_sheet(index_csv: Path, patient: str, out: Path | None = None, n_cols: int = 12) -> Path:
    df = load_index(index_csv)
    slices = df[df["group"] == patient].sort_values("order")
    if slices.empty:
        raise ValueError(f"Patient {patient} not found in {index_csv}")

    n_rows = int(np.ceil(len(slices) / n_cols))
    fig = Figure(figsize=(n_cols * 1.1, n_rows * 1.15))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for ax, row in zip(axes.ravel(), slices.itertuples(), strict=False):
        ax.imshow(Image.open(row.path), cmap="gray")
        is_lesion = isinstance(row.label, str) and row.label != "no_lesion"
        if is_lesion:
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("red")
                spine.set_linewidth(2.5)
        ax.set_title(str(row.order), fontsize=6, color="red" if is_lesion else "black")

    diagnosis = slices["diagnosis"].iloc[0]
    fig.suptitle(f"{patient} ({diagnosis}), top of the chest first. Red = labelled lesion slice", fontsize=9)
    fig.tight_layout()
    out = Path(out or Path(index_csv).parent / f"label_check_{patient}.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    gradcam = commands.add_parser("gradcam")
    gradcam.add_argument("--model-dir", required=True, type=Path)
    gradcam.add_argument("--preds", type=Path, help="predictions file, default: preds_test.csv of the model")
    gradcam.add_argument("--per-cell", type=int, default=3)
    gradcam.add_argument("--out", type=Path)

    labels = commands.add_parser("labels")
    labels.add_argument("--index", required=True, type=Path)
    labels.add_argument("--patient", required=True)
    labels.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "gradcam":
        gradcam_grid(args.model_dir, args.preds, args.per_cell, args.out)
    else:
        label_sheet(args.index, args.patient, args.out)


if __name__ == "__main__":
    main()
