"""Small fake datasets, so the whole pipeline can be tested without downloading anything."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from covid_ct_diagnosis.prepare import INDEX_COLUMNS


# A fake CT slice: a grey "body" disc, plus a bright blob whose position depends on the kind of slice
def fake_slice(rng: np.random.Generator, kind: str, size: int = 64) -> Image.Image:
    yy, xx = np.mgrid[:size, :size]
    img = 40 + 20 * rng.random((size, size))
    img[(yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.4) ** 2] += 60

    if kind in ("covid", "pCT"):
        img[(yy - size * 0.35) ** 2 + (xx - size * 0.3) ** 2 < (size * 0.1) ** 2] = 230
    if kind == "cap":
        img[(yy - size * 0.65) ** 2 + (xx - size * 0.7) ** 2 < (size * 0.1) ** 2] = 230
    if kind == "NiCT":
        img[:] = 30 + 10 * rng.random((size, size))

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


# A Tehran-like index: patients with IDs and diagnoses, and some COVID patients without slice labels
@pytest.fixture
def tehran_like(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    groups = [("Normal", 8, True), ("COVID-19", 8, True), ("COVID-19", 5, False), ("CAP", 8, True)]

    rows, patient_number = [], 0
    for diagnosis, n_patients, has_slice_labels in groups:
        for _ in range(n_patients):
            patient_number += 1
            patient_id = f"p{patient_number:03d}"

            for k in range(12):
                is_lesion = diagnosis != "Normal" and 3 <= k <= 8
                kind = "no_lesion" if not is_lesion else ("covid" if diagnosis == "COVID-19" else "cap")
                rel = Path("images") / patient_id / f"{k:04d}.png"
                (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
                fake_slice(rng, kind).save(tmp_path / rel)

                label = kind if (has_slice_labels or diagnosis == "Normal") else None
                rows.append({
                    "path": str(rel), "label": label, "covid": None if label is None else int(label == "covid"),
                    "group": patient_id, "diagnosis": diagnosis, "order": k, "source": "fake",
                })

    pd.DataFrame(rows)[INDEX_COLUMNS].to_csv(tmp_path / "index.csv", index=False)
    return tmp_path / "index.csv"


# A raw folder laid out like the Kaggle download of the Wuhan dataset (original + preprocessed copies)
@pytest.fixture
def wuhan_raw(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    root = tmp_path / "raw_ictcf"

    for copy in ("Original CT Scans", "Preprocessed CT scans"):
        for label in ("nCT", "pCT", "NiCT"):
            folder = root / copy / label
            folder.mkdir(parents=True)
            for i in range(10):
                fake_slice(rng, label, size=80).convert("RGB").save(folder / f"{label}{i + 1}.jpg")

    return root
