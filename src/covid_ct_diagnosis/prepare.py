"""Turn the raw downloads into one common format: 256x256 grayscale PNGs plus an index.csv.

Columns of index.csv:
    path       PNG path, relative to the folder that holds index.csv
    label      training class of the slice (empty = no slice label, only used for patient-level results)
    covid      1 = COVID, 0 = not COVID, empty = undefined. Used to test a model on the other dataset
    group      patient ID (empty for the Wuhan dataset, which has none)
    diagnosis  patient diagnosis: COVID-19, CAP (pneumonia) or Normal (Tehran only)
    order      file number (Wuhan) or slice position from the top of the chest (Tehran)
    source     dataset name

Usage:
    python -m covid_ct_diagnosis.prepare ictcf --src 0_data/raw/ictcf --out 0_data/processed/ictcf
    python -m covid_ct_diagnosis.prepare ctmd  --src 0_data/raw/covid-ct-md --out 0_data/processed/ctmd
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
INDEX_COLUMNS = ["path", "label", "covid", "group", "diagnosis", "order", "source"]

# Lung window: centre -600 HU, width 1500 HU, so the range [-1350, 150] HU is mapped to black..white
LUNG_WINDOW = (-1350.0, 150.0)


# ------------------------------------------------------------------------------------------- image helpers

# Pad an image with black to make it square (keeps the anatomy's proportions), then resize it
def pad_to_square_and_resize(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("L")
    width, height = img.size

    if width != height:
        side = max(width, height)
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(img, ((side - width) // 2, (side - height) // 2))
        img = canvas

    return img.resize((size, size), Image.BILINEAR)


# Map Hounsfield units to 0..255 through the lung window
def window_to_uint8(hu: np.ndarray, window: tuple[float, float] = LUNG_WINDOW) -> np.ndarray:
    low, high = window
    scaled = (np.clip(hu.astype(np.float32), low, high) - low) / (high - low)
    return (scaled * 255.0 + 0.5).astype(np.uint8)


# Last number in a file name: "nCT7356.jpg" -> 7356, "abc.png" -> -1
def file_number(name: str) -> int:
    numbers = re.findall(r"\d+", Path(name).stem)
    return int(numbers[-1]) if numbers else -1


# Convert one image file to a resized PNG (skipped if it already exists, so a re-run is fast)
def _convert_image(job: tuple[Path, Path, int]) -> None:
    src, dst, size = job
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        pad_to_square_and_resize(img, size).save(dst)


# Convert many images, in parallel when workers > 1
def _convert_images(jobs: list[tuple[Path, Path, int]], workers: int, desc: str) -> None:
    if workers <= 1:
        for job in tqdm(jobs, desc=desc):
            _convert_image(job)
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(_convert_image, jobs, chunksize=64), total=len(jobs), desc=desc))


# Save the rows as index.csv, always with the same columns in the same order
def _write_index(rows: list[dict], out: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for column in INDEX_COLUMNS:
        if column not in df.columns:
            df[column] = None

    out.mkdir(parents=True, exist_ok=True)
    df[INDEX_COLUMNS].to_csv(out / "index.csv", index=False)
    return df[INDEX_COLUMNS]


# ------------------------------------------------------------------------------------------- Wuhan (iCTCF)

# Find the folder of each class anywhere under src. If a class exists twice, prefer the path containing `prefer`
def _find_class_folders(src: Path, names: list[str], prefer: str) -> dict[str, Path]:
    found: dict[str, list[Path]] = {name: [] for name in names}
    by_lower_name = {name.lower(): name for name in names}

    for folder in src.rglob("*"):
        if folder.is_dir() and folder.name.lower() in by_lower_name:
            found[by_lower_name[folder.name.lower()]].append(folder)

    result = {}
    for name, folders in found.items():
        if not folders:
            raise FileNotFoundError(f"No folder named {name!r} found under {src}")
        preferred = [f for f in folders if prefer.lower() in str(f).lower()]
        result[name] = sorted(preferred or folders, key=lambda p: len(p.parts))[0]
    return result


# Wuhan dataset (iCTCF): classes nCT (normal), NiCT (non-informative), pCT (COVID). No patient IDs
def prepare_ictcf(src: Path, out: Path, size: int = 256, workers: int = 4) -> pd.DataFrame:

    # The download contains an "Original CT Scans" copy and a "Preprocessed CT scans" copy: use the original
    class_folders = _find_class_folders(src, ["nCT", "NiCT", "pCT"], prefer="original")

    # A non-informative slice shows no lungs, so it is neither COVID nor not-COVID
    covid_by_label = {"pCT": 1, "nCT": 0, "NiCT": None}

    rows, jobs = [], []
    for label, folder in class_folders.items():
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        for f in files:
            rel = Path("images") / label / f"{f.stem}.png"
            jobs.append((f, out / rel, size))
            rows.append({
                "path": str(rel), "label": label, "covid": covid_by_label[label],
                "order": file_number(f.name), "source": "ictcf",
            })

    _convert_images(jobs, workers, "Wuhan dataset")
    df = _write_index(rows, out)
    print(f"\nWuhan dataset: {len(df)} slices\n{df['label'].value_counts().to_string()}")
    return df


# ------------------------------------------------------------------------------------------- Tehran (COVID-CT-MD)

DIAGNOSIS_BY_FOLDER_WORD = {"covid": "COVID-19", "cap": "CAP", "normal": "Normal"}


# "COVID-19 subjects" -> "COVID-19", "Cap Cases" -> "CAP", anything else -> None
def _diagnosis_from_folder_name(name: str) -> str | None:
    first_word = re.split(r"[\s_-]", name.lower())[0]
    for word, diagnosis in DIAGNOSIS_BY_FOLDER_WORD.items():
        if first_word.startswith(word):
            return diagnosis
    return None


# Find every patient: [(diagnosis, patient ID, folder)], from class folders that contain patient folders of DICOMs
def find_ctmd_patients(src: Path) -> list[tuple[str, str, Path]]:
    patients, seen = [], set()

    for class_folder in sorted(p for p in src.rglob("*") if p.is_dir()):
        diagnosis = _diagnosis_from_folder_name(class_folder.name)
        if diagnosis is None:
            continue

        for patient_folder in sorted(p for p in class_folder.iterdir() if p.is_dir()):
            has_dicoms = any(patient_folder.glob("*.dcm"))
            if has_dicoms and patient_folder.name not in seen:
                seen.add(patient_folder.name)
                patients.append((diagnosis, patient_folder.name, patient_folder))

    return patients


# Order a patient's slices from the top of the chest down (highest SliceLocation first)
def sorted_dicoms(patient_folder: Path) -> list[Path]:
    import pydicom

    # This is the order the radiologists' slice labels use: checked visually on sample patient cap002,
    # whose labelled lesions only line up with the visible ones in this order (see 0_data/README.md)
    keyed = []
    for f in patient_folder.glob("*.dcm"):
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        if "SliceLocation" in ds:
            z = float(ds.SliceLocation)
        elif "ImagePositionPatient" in ds:
            z = float(ds.ImagePositionPatient[2])
        else:
            z = -float(getattr(ds, "InstanceNumber", 0))
        keyed.append((z, file_number(f.name), f))

    keyed.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [f for _, _, f in keyed]


# Read a DICOM slice and convert its pixel values to Hounsfield units (HU)
def dicom_to_hu(path: Path) -> np.ndarray:
    import pydicom

    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return pixels * slope + intercept


# Map patient ID -> its row of radiologist slice labels (1 = infection visible on that slice)
def load_ctmd_slice_labels(src: Path) -> dict[str, np.ndarray]:

    def find(name: str) -> Path | None:
        hits = sorted(src.rglob(name))
        return hits[0] if hits else None

    labels_file = find("Slice-level-labels-updated-1.npy") or find("Slice-level-labels.npy")
    index_file = find("Index.csv")
    if labels_file is None or index_file is None:
        print("WARNING: slice labels or Index.csv not found, only patient diagnoses will be available")
        return {}

    # The label array's rows follow the row order of Index.csv. Its "Label Index" column skips a value,
    # so it must not be used as a row number
    labels = np.load(labels_file)
    index = pd.read_csv(index_file).sort_values("Label Index").reset_index(drop=True)
    if len(index) != labels.shape[0]:
        raise ValueError(f"Index.csv has {len(index)} rows but {labels_file.name} has {labels.shape[0]}")

    return {str(row["Folder/ID"]).strip(): labels[i] for i, row in index.iterrows()}


# The training class of slice k for a patient, or None when the patient has no slice labels
def _slice_label(diagnosis: str, labels_row: np.ndarray | None, k: int) -> str | None:
    if diagnosis == "Normal":
        return "no_lesion"
    if labels_row is None or k >= len(labels_row):
        return None
    if labels_row[k] != 1:
        return "no_lesion"
    return "covid" if diagnosis == "COVID-19" else "cap"


# Convert all slices of one patient and return their index rows
def _convert_patient(job: tuple) -> list[dict]:
    diagnosis, patient_id, folder, out, size, labels_row = job
    files = sorted_dicoms(folder)

    rows = []
    for k, f in enumerate(files):
        rel = Path("images") / patient_id / f"{k:04d}.png"
        dst = out / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(window_to_uint8(dicom_to_hu(f)))
            pad_to_square_and_resize(img, size).save(dst)

        label = _slice_label(diagnosis, labels_row, k)
        rows.append({
            "path": str(rel), "label": label, "covid": None if label is None else int(label == "covid"),
            "group": patient_id, "diagnosis": diagnosis, "order": k, "source": "ctmd",
        })
    return rows


# Tehran dataset (COVID-CT-MD): 305 patients with IDs, slice labels for 54 COVID + 25 CAP patients
def prepare_ctmd(src: Path, out: Path, size: int = 256, workers: int = 4) -> pd.DataFrame:
    patients = find_ctmd_patients(src)
    if not patients:
        raise FileNotFoundError(f"No patient folders with .dcm files found under {src}")

    slice_labels = load_ctmd_slice_labels(src)
    jobs = [(diag, pid, folder, out, size, slice_labels.get(pid)) for diag, pid, folder in patients]

    rows: list[dict] = []
    if workers <= 1:
        for job in tqdm(jobs, desc="Tehran patients"):
            rows.extend(_convert_patient(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for patient_rows in tqdm(pool.map(_convert_patient, jobs), total=len(jobs), desc="Tehran patients"):
                rows.extend(patient_rows)

    df = _write_index(rows, out)
    _print_ctmd_summary(df, slice_labels)
    return df


# Print counts and check that no lesion label points past a patient's last slice
def _print_ctmd_summary(df: pd.DataFrame, slice_labels: dict[str, np.ndarray]) -> None:
    patients = df.groupby("group").agg(diagnosis=("diagnosis", "first"), n_slices=("path", "size"))
    print("\nTehran patients per diagnosis:\n" + patients["diagnosis"].value_counts().to_string())
    print("\nSlices per label:\n" + df["label"].fillna("(no slice label)").value_counts().to_string())

    # A lesion label beyond the last slice would mean the slices and labels are misaligned
    misaligned = []
    for patient_id, row in slice_labels.items():
        if patient_id in patients.index:
            n_slices = int(patients.loc[patient_id, "n_slices"])
            if row[n_slices:].sum() > 0:
                misaligned.append(patient_id)

    if misaligned:
        print(f"\nWARNING: {len(misaligned)} patients have labels past their last slice: {misaligned[:5]}")
    else:
        print("\nOK: every lesion label falls on an existing slice.")


# ------------------------------------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["ictcf", "ctmd"])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    if args.dataset == "ictcf":
        prepare_ictcf(args.src, args.out, args.size, args.workers)
    else:
        prepare_ctmd(args.src, args.out, args.size, args.workers)


if __name__ == "__main__":
    main()
