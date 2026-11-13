#!/usr/bin/env bash
# Download both datasets into 0_data/raw/. Safe to run again: finished downloads are skipped,
# and an interrupted Tehran download continues where it stopped.
#
# Needs a Kaggle API token in ~/.kaggle/access_token (kaggle.com > Settings > API > Create New Token).
set -euo pipefail

cd "$(dirname "$0")/.."
RAW=0_data/raw
KAGGLE=.venv/bin/kaggle

# Tehran dataset on figshare: one zip file, and its exact size, to check the download is complete
CTMD_URL=https://ndownloader.figshare.com/files/26069987
CTMD_BYTES=11273767727
CTMD_LABELS=https://raw.githubusercontent.com/ShahinSHH/COVID-CT-MD/main

mkdir -p "$RAW"


# Size of a file in bytes, on macOS and on Linux
file_size() {
  stat -f %z "$1" 2> /dev/null || stat -c %s "$1"
}


# Wuhan dataset (iCTCF) from Kaggle (~3.6 GB: an original and a preprocessed copy)
# A marker file is written only once a download has fully finished, so an interrupted one is redone
if [[ -f "$RAW/ictcf/.download_complete" ]]; then
  echo "==> Wuhan dataset already downloaded"
else
  echo "==> Downloading the Wuhan dataset from Kaggle"
  "$KAGGLE" datasets download -d azaemon/preprocessed-ct-scans-for-covid19 -p "$RAW/ictcf" --unzip
  touch "$RAW/ictcf/.download_complete"
fi


# Tehran dataset from figshare (~10.5 GB). curl -C - resumes a partial download
ZIP="$RAW/covid-ct-md.zip"
if [[ -f "$RAW/covid-ct-md/.unpack_complete" ]]; then
  echo "==> Tehran dataset already downloaded and unpacked"
else
  if [[ ! -f "$ZIP" ]] || [[ "$(file_size "$ZIP")" != "$CTMD_BYTES" ]]; then
    echo "==> Downloading the Tehran dataset (~10.5 GB, this takes a while)"
    curl -L -C - --retry 5 -o "$ZIP" "$CTMD_URL"
  fi

  if [[ "$(file_size "$ZIP")" != "$CTMD_BYTES" ]]; then
    echo "The download is incomplete ($(file_size "$ZIP") of $CTMD_BYTES bytes). Run step0 again to continue it."
    exit 1
  fi

  # ditto (macOS) handles very large zip files reliably; unzip elsewhere
  echo "==> Unpacking the Tehran dataset"
  mkdir -p "$RAW/covid-ct-md"
  if command -v ditto > /dev/null; then
    ditto -x -k "$ZIP" "$RAW/covid-ct-md"
  else
    unzip -q -o "$ZIP" -d "$RAW/covid-ct-md"
  fi
  touch "$RAW/covid-ct-md/.unpack_complete"
fi


# Radiologist slice labels and their index, published by the dataset's authors on GitHub
mkdir -p "$RAW/covid-ct-md"
curl -sSL -o "$RAW/covid-ct-md/Index.csv" "$CTMD_LABELS/Index.csv"
curl -sSL -o "$RAW/covid-ct-md/Slice-level-labels-updated-1.npy" "$CTMD_LABELS/Slice-level-labels-updated-1.npy"

echo "==> Downloads done. The zip ($ZIP, 10.5 GB) can be deleted once Step 0 has finished."
