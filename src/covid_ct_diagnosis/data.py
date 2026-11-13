"""Loading slices, splitting them into train / val / test, and preparing images for the model.

The split is the heart of this project:
    random  shuffle slices. Slices of the same patient end up in train AND test, so the score is too
            optimistic. This is what most public projects on this data do.
    group   split by patient ID: each patient is entirely in train, val or test. The honest way.

Cross-validation repeats this 5 times with 5 folds, so that every slice (random) or every patient (group)
is in the test set exactly once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

# The pretrained model expects images normalised with the ImageNet colour statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Read an index.csv and turn its relative image paths into full paths
def load_index(index_csv: str | Path) -> pd.DataFrame:
    index_csv = Path(index_csv)
    df = pd.read_csv(index_csv, dtype={"group": "string", "label": "string", "diagnosis": "string"})
    df["path"] = [str(index_csv.parent / p) for p in df["path"]]
    return df


# ------------------------------------------------------------------------------------------- splits

# Number of items that go to train, and to train + val, for n items
def _cut_points(n: int, fractions: tuple[float, float, float]) -> tuple[int, int]:
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    return n_train, n_train + n_val


# Shuffle slices into train / val / test, keeping the class balance (the usual, leaky way)
def split_random(df: pd.DataFrame, fractions, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, dtype="object")

    for _, rows in df.groupby(df["label"].fillna("(none)")):
        idx = rows.index.to_numpy().copy()
        rng.shuffle(idx)
        a, b = _cut_points(len(idx), fractions)
        split[idx[a:b]] = "val"
        split[idx[b:]] = "test"

    return split


# Kind of each patient: diagnosis + whether it has slice labels. Splits and folds keep the same mix of kinds
def _patient_strata(df: pd.DataFrame) -> pd.Series:
    if df["group"].isna().any():
        raise ValueError("A split by patient needs a patient ID on every row")
    patients = df.groupby("group")
    has_labels = patients["label"].apply(lambda labels: labels.notna().any())
    return patients["diagnosis"].first().fillna("(none)") + "|" + has_labels.astype(str)


# Split patients into train / val / test so that no patient appears in two splits (the honest way)
def split_by_patient(df: pd.DataFrame, fractions, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, dtype="object")
    strata = _patient_strata(df)

    for _, patient_ids in strata.groupby(strata):
        ids = patient_ids.index.to_numpy().copy()
        rng.shuffle(ids)
        a, b = _cut_points(len(ids), fractions)
        split[df["group"].isin(ids[a:b])] = "val"
        split[df["group"].isin(ids[b:])] = "test"

    return split


# Pick the split method named in the settings file
def assign_splits(df: pd.DataFrame, method: str, fractions=(0.7, 0.15, 0.15), seed: int = 42) -> pd.Series:
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("The three split fractions must add up to 1")
    if method == "random":
        return split_random(df, fractions, seed)
    if method == "group":
        return split_by_patient(df, fractions, seed)
    raise ValueError(f"Unknown split method {method!r}, use 'random' or 'group'")


# ------------------------------------------------------------------------------------------- cross-validation

# Give every row a fold number 0 .. n_folds-1, dealing shuffled items into the folds in turn like cards,
# kind by kind, so every fold gets the same mix.
#   group   items are patients: all slices of a patient share one fold (the honest way)
#   random  items are slices: a patient's slices are spread over all folds (the usual, leaky way)
def assign_folds(df: pd.DataFrame, method: str, n_folds: int, seed: int = 42) -> pd.Series:
    if n_folds < 3:
        raise ValueError("Cross-validation needs at least 3 folds (train, val and test)")
    rng = np.random.default_rng(seed)
    position = 0

    if method == "group":
        strata = _patient_strata(df)
        fold_of_patient = {}
        for _, patient_ids in strata.groupby(strata):
            ids = patient_ids.index.to_numpy().copy()
            rng.shuffle(ids)
            for patient_id in ids:
                fold_of_patient[patient_id] = position % n_folds
                position += 1
        return df["group"].map(fold_of_patient).astype(int)

    if method == "random":
        folds = pd.Series(0, index=df.index, dtype=int)
        for _, rows in df.groupby(df["label"].fillna("(none)")):
            idx = rows.index.to_numpy().copy()
            rng.shuffle(idx)
            folds[idx] = (np.arange(len(idx)) + position) % n_folds
            position += len(idx)
        return folds

    raise ValueError(f"Unknown split method {method!r}, use 'random' or 'group'")


# The three parts of run k: test = fold k, validation = the next fold, training = all the others
def split_for_fold(folds: pd.Series, k: int, n_folds: int) -> pd.Series:
    split = pd.Series("train", index=folds.index, dtype="object")
    split[folds == k] = "test"
    split[folds == (k + 1) % n_folds] = "val"
    return split


# Stop with an error if any patient appears in more than one split
def check_no_patient_leakage(df: pd.DataFrame, split: pd.Series) -> None:
    if df["group"].isna().all():
        return
    splits_per_patient = split.groupby(df["group"]).nunique()
    leaked = splits_per_patient[splits_per_patient > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} patients appear in several splits, e.g. {list(leaked.index[:5])}")


# ------------------------------------------------------------------------------------------- images

# Image preparation. Training adds small random changes (crop, flip, rotation, brightness) so the
# model can't memorise exact pixels; evaluation only resizes
def build_transforms(img_size: int, train: bool, augment: dict | None = None) -> T.Compose:
    augment = augment or {}
    to_tensor = [T.Grayscale(3), T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    normalise = [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    if not train:
        return T.Compose([T.Resize((img_size, img_size), antialias=True), *to_tensor, *normalise])

    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(augment.get("crop_scale_min", 0.75), 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=augment.get("hflip", 0.5)),
        T.RandomAffine(degrees=augment.get("rotate", 10), translate=(0.05, 0.05)),
        T.ColorJitter(brightness=augment.get("brightness", 0.2), contrast=augment.get("contrast", 0.2)),
        *to_tensor,
        *normalise,
    ])


# One CT slice per item: (image tensor, class number or -1 when unlabelled, row number)
class SliceDataset(Dataset):

    def __init__(self, df: pd.DataFrame, classes: list[str], transform):
        self.paths = df["path"].tolist()
        self.transform = transform
        class_to_number = {c: i for i, c in enumerate(classes)}
        self.targets = [class_to_number.get(label, -1) if isinstance(label, str) else -1 for label in df["label"]]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as img:
            img = img.convert("L")
        return self.transform(img), self.targets[i], i


# Loss weights that give rare classes more importance (power 0.5 = a gentle rebalance)
def class_weights(targets: list[int], n_classes: int, power: float = 0.5) -> torch.Tensor:
    counts = np.bincount([t for t in targets if t >= 0], minlength=n_classes).astype(float)
    weights = (counts.sum() / np.maximum(counts, 1)) ** power
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)
