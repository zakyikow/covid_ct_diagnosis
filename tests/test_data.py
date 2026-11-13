"""Tests for data preparation and splits."""

import numpy as np
import pytest
from PIL import Image

from covid_ct_diagnosis.data import (
    assign_folds,
    assign_splits,
    check_no_patient_leakage,
    class_weights,
    load_index,
    split_for_fold,
)
from covid_ct_diagnosis.prepare import (
    _diagnosis_from_folder_name,
    _slice_label,
    file_number,
    pad_to_square_and_resize,
    prepare_ictcf,
    window_to_uint8,
)


# The honest split never puts a patient in two splits, and every split gets every diagnosis
def test_split_by_patient_has_no_leakage(tehran_like):
    df = load_index(tehran_like)
    split = assign_splits(df, "group", (0.6, 0.2, 0.2), seed=0)

    check_no_patient_leakage(df, split)
    assert set(split) == {"train", "val", "test"}
    for name in ("train", "val", "test"):
        assert set(df.loc[split == name, "diagnosis"]) == {"Normal", "COVID-19", "CAP"}


# The random split (the usual one in public projects) does leak patients, and the check catches it
def test_random_split_leaks_patients(tehran_like):
    df = load_index(tehran_like)
    split = assign_splits(df, "random", seed=0)
    with pytest.raises(AssertionError):
        check_no_patient_leakage(df, split)


# Cross-validation by patient: a patient's slices share one fold, every patient is tested and validated exactly
# once, no run puts a patient in two parts, and the folds have the same size and mix
def test_folds_by_patient(tehran_like):
    df = load_index(tehran_like)
    folds = assign_folds(df, "group", 5, seed=0)
    assert (folds.groupby(df["group"]).nunique() == 1).all()

    tested, validated = [], []
    for k in range(5):
        split = split_for_fold(folds, k, 5)
        check_no_patient_leakage(df, split)
        tested += df.loc[split == "test", "group"].unique().tolist()
        validated += df.loc[split == "val", "group"].unique().tolist()
    assert sorted(tested) == sorted(df["group"].unique())
    assert sorted(validated) == sorted(df["group"].unique())

    patients_per_fold = folds.groupby(df["group"]).first().value_counts()
    assert patients_per_fold.max() - patients_per_fold.min() <= 1
    for k in range(5):
        assert set(df.loc[folds == k, "diagnosis"]) == {"Normal", "COVID-19", "CAP"}


# Cross-validation with a random split: every slice tested once, and a patient's slices are spread over folds
def test_random_folds_spread_patients(tehran_like):
    df = load_index(tehran_like)
    folds = assign_folds(df, "random", 5, seed=0)
    tests = sum((split_for_fold(folds, k, 5) == "test").astype(int) for k in range(5))
    assert (tests == 1).all()
    assert (folds.groupby(df["group"]).nunique() > 1).any()


def test_cross_validation_needs_three_folds(tehran_like):
    with pytest.raises(ValueError):
        assign_folds(load_index(tehran_like), "group", 2)


def test_split_fractions_must_add_up_to_one(tehran_like):
    with pytest.raises(ValueError):
        assign_splits(load_index(tehran_like), "random", (0.5, 0.2, 0.2))


def test_class_weights_favour_the_rare_class():
    weights = class_weights([0] * 90 + [1] * 10, 2)
    assert weights[1] > weights[0]


# -1350 HU and below -> black, 150 HU and above -> white, the window's centre (-600 HU) -> mid grey
def test_lung_window():
    hu = np.array([[-2000, -1350, -600, 150, 3000]], dtype=float)
    assert window_to_uint8(hu).tolist() == [[0, 0, 128, 255, 255]]


# A wide image gets black bars above and below, not stretched
def test_padding_keeps_proportions():
    img = pad_to_square_and_resize(Image.new("L", (100, 50), 255), 32)
    pixels = np.asarray(img)
    assert pixels.shape == (32, 32)
    assert pixels[0, 16] == 0 and pixels[16, 16] == 255


def test_file_number():
    assert file_number("nCT7356.jpg") == 7356
    assert file_number("abc.png") == -1


def test_tehran_folder_names():
    assert _diagnosis_from_folder_name("COVID-19 subjects") == "COVID-19"
    assert _diagnosis_from_folder_name("Cap Cases") == "CAP"
    assert _diagnosis_from_folder_name("Normal subjects") == "Normal"
    assert _diagnosis_from_folder_name("Sample data") is None


# Normal patients: every slice is "no_lesion". COVID/CAP: follow the labels, or None without labels
def test_tehran_slice_labels():
    row = np.array([0, 1, 0])
    assert _slice_label("Normal", None, 5) == "no_lesion"
    assert _slice_label("COVID-19", row, 1) == "covid"
    assert _slice_label("CAP", row, 1) == "cap"
    assert _slice_label("CAP", row, 0) == "no_lesion"
    assert _slice_label("COVID-19", None, 1) is None


# Wuhan dataset: the "Original CT Scans" copy is used, images are resized, NiCT has no COVID target
def test_prepare_wuhan_dataset(wuhan_raw, tmp_path):
    out = tmp_path / "processed"
    df = prepare_ictcf(wuhan_raw, out, size=32, workers=1)

    assert len(df) == 30
    assert df["label"].value_counts().to_dict() == {"nCT": 10, "pCT": 10, "NiCT": 10}
    assert df.loc[df["label"] == "NiCT", "covid"].isna().all()
    assert Image.open(load_index(out / "index.csv")["path"].iloc[0]).size == (32, 32)
