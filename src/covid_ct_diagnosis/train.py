"""Fine-tune MobileNetV3 on CT slices, then evaluate it on the test split.

    python -m covid_ct_diagnosis.train --config 1_reproduce_baseline/config.yaml
    python -m covid_ct_diagnosis.train --config 3_honest_model/patient_split.yaml train.epochs=3

One run writes to the `output_dir` of the settings file: model.pt, config.yaml, history.csv, preds_val.csv,
preds_test.csv, metrics.json, plots/ (and split_patients.csv for a split by patient).

With `cv_folds` in the settings (cross-validation), there is one such folder per run (fold_1 ... fold_5), and the
combined results of all runs are written to `output_dir` itself. A run that already finished is skipped, so an
interrupted cross-validation continues where it stopped.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    SliceDataset,
    assign_folds,
    assign_splits,
    build_transforms,
    check_no_patient_leakage,
    class_weights,
    load_index,
    split_for_fold,
)
from .metrics import multiclass_metrics, softmax
from .models import build_model, predict_logits
from .utils import load_config, pick_device, seed_everything


# Build a DataLoader for a set of slices, with training-time random changes if `train` is True
def make_loader(df: pd.DataFrame, cfg: dict, train: bool) -> DataLoader:
    transform = build_transforms(cfg["data"]["img_size"], train, cfg.get("augment"))
    dataset = SliceDataset(df, cfg["classes"], transform)
    workers = cfg["train"]["num_workers"]
    return DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=train, num_workers=workers,
                      persistent_workers=workers > 0)


# Load the index and keep the rows this experiment uses
def load_data(cfg: dict) -> pd.DataFrame:
    df = load_index(cfg["data"]["index"])
    if cfg["data"].get("query"):
        df = df.query(cfg["data"]["query"]).copy()
    return df


# Load the data and add a "split" column (train / val / test) for a single run
def load_and_split(cfg: dict) -> pd.DataFrame:
    df = load_data(cfg)
    df["split"] = assign_splits(df, cfg["data"]["split"], tuple(cfg["data"]["fractions"]), cfg["seed"])
    if cfg["data"]["split"] == "group":
        check_no_patient_leakage(df, df["split"])
    return df


# Learning rate schedule: warm up during the first 10% of steps, then decrease smoothly (cosine) to 0
def make_scheduler(optimizer, total_steps: int, warmup_steps: int):

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# One pass over the training data. Returns the average loss
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, desc: str) -> float:
    model.train()
    losses = []
    for images, targets, _ in tqdm(loader, desc=desc, leave=False):
        images, targets = images.to(device), targets.to(device)
        loss = criterion(model(images), targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses))


# Train with the settings: one run, or a full cross-validation when `cv_folds` is set
def train(cfg: dict) -> Path:
    if cfg["data"].get("cv_folds"):
        return train_cross_validation(cfg)

    out = Path(cfg["output_dir"])
    train_run(cfg, load_and_split(cfg), out)
    return out


# Cross-validation: one run per fold, each tested on a different fold, then the combined results
def train_cross_validation(cfg: dict) -> Path:
    from .evaluate import evaluate_cross_validation

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    n_folds = cfg["data"]["cv_folds"]
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # The folds, and for a split by patient, the list of which patient is in which fold
    df = load_data(cfg)
    folds = assign_folds(df, cfg["data"]["split"], n_folds, cfg["seed"])
    if cfg["data"]["split"] == "group":
        patients = df.assign(fold=folds + 1).groupby("group").agg(diagnosis=("diagnosis", "first"), fold=("fold", "first"))
        patients.to_csv(out / "folds.csv")
        print("patients per fold:", patients["fold"].value_counts().sort_index().to_dict())

    for k in range(n_folds):
        fold_dir = out / f"fold_{k + 1}"

        # A run is finished once its results are saved: skip it, so an interrupted cross-validation can resume
        if (fold_dir / "metrics.json").exists():
            print(f"\n==> fold {k + 1}/{n_folds} already finished, skipping")
            continue

        print(f"\n==> fold {k + 1}/{n_folds}: test = fold {k + 1}, validation = fold {(k + 1) % n_folds + 1}")
        fold_df = df.assign(split=split_for_fold(folds, k, n_folds))
        if cfg["data"]["split"] == "group":
            check_no_patient_leakage(fold_df, fold_df["split"])
        fold_cfg = {**cfg, "name": f"{cfg['name']} / fold {k + 1}", "output_dir": str(fold_dir), "fold": k + 1}
        train_run(fold_cfg, fold_df, fold_dir)

    evaluate_cross_validation(out)
    return out


# One run: train, keep the best model on the validation slices, predict val and test, then evaluate
def train_run(cfg: dict, df: pd.DataFrame, out: Path) -> Path:
    seed_everything(cfg["seed"])
    device = pick_device(cfg["train"]["device"])
    out.mkdir(parents=True, exist_ok=True)
    classes = cfg["classes"]

    # Keep only slices with a training label for training and model selection
    labelled = df["label"].isin(classes)
    train_df = df[(df["split"] == "train") & labelled]
    val_df = df[(df["split"] == "val") & labelled]
    print(f"device: {device}")
    print(df[labelled].groupby(["split", "label"]).size().unstack(fill_value=0))

    # Save which patient went where, so the split can be checked and reused. Only for a split by patient:
    # with a random split, every patient is spread over train, val and test
    if cfg["data"]["split"] == "group":
        patients = df.groupby("group").agg(diagnosis=("diagnosis", "first"), split=("split", "first"))
        patients.to_csv(out / "split_patients.csv")
        print("patients per split:", patients["split"].value_counts().to_dict())

    # Optional cap on the training set size, for quick tests
    max_train = cfg["train"].get("max_train_samples")
    if max_train and len(train_df) > max_train:
        train_df = train_df.sample(max_train, random_state=cfg["seed"])

    train_loader = make_loader(train_df, cfg, train=True)
    val_loader = make_loader(val_df, cfg, train=False)

    # Model, loss (rare classes weighted up), optimiser and learning rate schedule
    model = build_model(len(classes), cfg["model"]["pretrained"], cfg["model"]["drop_rate"]).to(device)
    weights = class_weights(train_loader.dataset.targets, len(classes), cfg["train"]["class_weight_power"])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["epochs"]
    total_steps = epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps=total_steps // 10)

    # Training loop with early stopping: stop when validation balanced accuracy hasn't improved for `patience` epochs
    best_score, best_epoch, history = -np.inf, 0, []
    y_val = np.array(val_loader.dataset.targets)
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, f"epoch {epoch}/{epochs}")
        val = multiclass_metrics(y_val, softmax(predict_logits(model, val_loader, device)), classes)

        history.append({
            "epoch": epoch, "train_loss": loss, "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"], "seconds": round(time.time() - start, 1),
        })
        print(f"epoch {epoch}: loss {loss:.4f} | val accuracy {val['accuracy']:.4f} | "
              f"val balanced accuracy {val['balanced_accuracy']:.4f} | {history[-1]['seconds']}s")

        if val["balanced_accuracy"] > best_score:
            best_score, best_epoch = val["balanced_accuracy"], epoch
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch}, out / "model.pt")
        elif epoch - best_epoch >= cfg["train"]["patience"]:
            print(f"early stopping: no improvement for {cfg['train']['patience']} epochs")
            break

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"best validation balanced accuracy {best_score:.4f} at epoch {best_epoch}")

    # Reload the best model and predict every val and test slice, including unlabelled ones
    # (those are needed for the patient-level diagnosis)
    model.load_state_dict(torch.load(out / "model.pt", map_location=device, weights_only=False)["state_dict"])
    for split in ("val", "test"):
        part = df[df["split"] == split]
        logits = predict_logits(model, make_loader(part, cfg, train=False), device)
        save_predictions(part, logits, classes, out / f"preds_{split}.csv")

    from .evaluate import evaluate_run

    evaluate_run(out)
    return out


# Save each slice's index row plus the model's raw score for every class
def save_predictions(df: pd.DataFrame, logits: np.ndarray, classes: list[str], path: Path) -> pd.DataFrame:
    preds = df.drop(columns=["split"], errors="ignore").reset_index(drop=True).copy()
    for i, c in enumerate(classes):
        preds[f"logit_{c}"] = logits[:, i]
    preds.to_csv(path, index=False)
    return preds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="key=value changes to the settings, e.g. train.epochs=3")
    args = parser.parse_args(argv)
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
