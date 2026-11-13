# Data

No images are stored in this repository. `bash scripts/run.sh step0` downloads both datasets into `0_data/raw/`
and converts them into `0_data/processed/<dataset>/`: 256×256 grayscale PNGs plus an `index.csv`.
`explore_data.ipynb` takes a first look at both.

## Sources

| | Wuhan dataset (iCTCF) | Tehran dataset (COVID-CT-MD) |
|---|---|---|
| Collected at | 2 hospitals in Wuhan, China | Babak Imaging Center, Tehran, Iran: one scanner (Siemens SOMATOM Scope) |
| Downloaded from | Kaggle `azaemon/preprocessed-ct-scans-for-covid19` | figshare collection 5129081, labels from github.com/ShahinSHH/COVID-CT-MD |
| What | 19,685 single slices: nCT 9,979 (normal), pCT 4,001 (COVID), NiCT 5,705 (non-informative) | 305 full scans (DICOM): 169 COVID-19, 60 CAP (regular pneumonia), 76 normal |
| Patients | about 104 (Ning et al. 2020) | 305 |
| Patient IDs | **no** | **yes**, one folder per patient |
| Slice labels | the class of each slice | radiologists marked infected slices for 54 COVID + 25 CAP patients |
| Licence (as listed by the source) | CC BY 4.0 | CC0 |
| Used for | Step 1: the baseline is trained here | Step 2: the baseline is tested on all 305 patients. Step 3: both trainings (random vs. by patient), each with 5-fold cross-validation, so every patient is tested once. Step 4: the demo |

## Preprocessing decisions

- **Image size.** Images are padded to a square with black, then resized to 256 px. Training then crops 224 px.
- **Tehran DICOMs → images.**
  - Pixel values are converted to Hounsfield units (HU): `pixel × RescaleSlope + RescaleIntercept`.
  - Then a lung window is applied: centre −600 HU, width 1500 HU, so [−1350, 150] HU is mapped to black → white.
- **Different looks.** The Wuhan images come already converted to 8-bit images, with an unknown window.
  So the two datasets don't look exactly alike. That difference is part of what Step 2 measures.
- **Tehran slice order.**
  - The dataset's authors say to sort slices by `SliceLocation`, but not in which direction.
  - Their labels count slices **from the top of the chest down**: highest `SliceLocation` first, which matches
    `InstanceNumber` 1, 2, 3…
  - Checked on sample patient `cap002`: in this order its 30 labelled slices (positions 89–118 of 131) are the
    lung bases, where the opacities are. Bottom-up order would put the labels on clean slices.
  - `explore_data.ipynb` redraws this check.
- **Tehran label rows** follow the row order of `Index.csv`. Its `Label Index` column skips a value
  (COVID 0–53, CAP 55–79), so it can't be used as a row number.

## Labels used by the code

| Dataset | `label` (training class) | `covid` (COVID yes / no, used in Step 2) |
|---|---|---|
| Wuhan | nCT / NiCT / pCT | pCT → 1, nCT → 0, NiCT → undefined |
| Tehran | per slice: no_lesion / covid / cap. Empty for the 150 patients without slice labels (115 COVID, 35 CAP) | covid lesion → 1, any other labelled slice → 0 |

Each Tehran patient's diagnosis (COVID-19 / CAP / Normal) is in the `diagnosis` column.

## Known weak points

- The Wuhan dataset has no patient IDs, so a random split leaks patients. Step 3 measures how much that matters,
  on the Tehran dataset where patient IDs exist.
- Tehran: COVID scans are from Feb–Apr 2020, CAP scans from 2018–2019 and normal scans from 2019–2020.
  All come from one scanner, but settings may have changed between those periods, which could help a model
  separate the classes without looking at the lungs.
