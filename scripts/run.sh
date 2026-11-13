#!/usr/bin/env bash
# Every step of the project as one command. Run from the project folder:  bash scripts/run.sh <step>
#
#   step0          install everything, download both datasets (~14 GB), convert them, draw the label check
#   step1          the baseline: MobileNetV3 on the Wuhan dataset, random split (the usual ~99%)
#   step2          new hospital: the baseline tested on the Tehran patients (no training)
#   step3_random   Step 3, part 1: Tehran with the usual random split, 5-fold cross-validation (5 trainings)
#   step3_patient  Step 3, part 2: Tehran split by patient = the honest model, 5-fold cross-validation (5 trainings)
#                  The two differ only in the split: the gap between them = leakage.
#                  Interrupted? Run the same command again: finished folds are skipped
#   step4          demo data: the 6 demo patients, each through its own fold's model -> 4_demo/public/data/
#   report         one table with every result -> results.md
#   notebook       open 0_data/explore_data.ipynb
#   test           run the automatic tests
#
# On macOS, long steps keep the computer awake with `caffeinate` until they finish.
set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."
PY=.venv/bin/python
RAW=0_data/raw
PROC=0_data/processed
STEP=${1:-help}


# Re-run this script under `caffeinate -i` (macOS) so the computer doesn't sleep during a long step
keep_awake() {
  if [[ -z "${KEEP_AWAKE:-}" ]] && command -v caffeinate > /dev/null; then
    echo "(keeping the computer awake until this step finishes)"
    KEEP_AWAKE=1 exec caffeinate -i bash "$SCRIPT" "$STEP"
  fi
}


# Create .venv with Python 3.10+ and install the project, unless it already exists
setup() {
  if [[ -x "$PY" ]]; then
    return
  fi

  PYBIN=${PYBIN:-python3}
  if ! "$PYBIN" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo "Need Python 3.10 or newer, found $("$PYBIN" --version)."
    exit 1
  fi

  echo "==> Creating .venv and installing the project (a few minutes)"
  "$PYBIN" -m venv .venv
  .venv/bin/pip install -q -U pip
  .venv/bin/pip install -q -e ".[dev,data,notebook]"
}


# Stop with a clear message if an earlier step hasn't been run
require() {
  if [[ ! -e "$1" ]]; then
    echo "Missing $1. Run '$2' first."
    exit 1
  fi
}


case "$STEP" in

  step0)
    keep_awake
    setup
    bash scripts/download_data.sh

    echo "==> Converting the Wuhan dataset"
    $PY -m covid_ct_diagnosis.prepare ictcf --src $RAW/ictcf --out $PROC/ictcf --workers 6

    echo "==> Converting the Tehran dataset (DICOM -> images, slowest part)"
    $PY -m covid_ct_diagnosis.prepare ctmd --src $RAW/covid-ct-md --out $PROC/ctmd --workers 6

    echo "==> Drawing the slice-label check for patient cap002"
    $PY -m covid_ct_diagnosis.explain labels --index $PROC/ctmd/index.csv --patient cap002

    echo
    echo "Step 0 done. Label check: $PROC/ctmd/label_check_cap002.png"
    echo "Next: open 0_data/explore_data.ipynb (or: bash scripts/run.sh notebook), then bash scripts/run.sh step1"
    ;;

  step1)
    keep_awake
    require $PROC/ictcf/index.csv "bash scripts/run.sh step0"
    $PY -m covid_ct_diagnosis.train --config 1_reproduce_baseline/config.yaml
    echo
    echo "Step 1 done. Results: 1_reproduce_baseline/results/metrics.json"
    ;;

  step2)
    keep_awake
    require $PROC/ctmd/index.csv "bash scripts/run.sh step0"
    require 1_reproduce_baseline/results/model.pt "bash scripts/run.sh step1"

    echo "==> The baseline tested on the Tehran patients"
    $PY -m covid_ct_diagnosis.evaluate --config 2_other_hospital_test/config.yaml
    echo
    echo "Step 2 done. Results: 2_other_hospital_test/results/"
    ;;

  step3_random)
    keep_awake
    require $PROC/ctmd/index.csv "bash scripts/run.sh step0"
    echo "==> Step 3, part 1: random split (the usual, leaky way), 5 folds"
    $PY -m covid_ct_diagnosis.train --config 3_honest_model/random_split.yaml
    echo
    echo "Step 3 part 1 done. Results: 3_honest_model/results/random_split/metrics.json"
    echo "Next: bash scripts/run.sh step3_patient"
    ;;

  step3_patient)
    keep_awake
    require $PROC/ctmd/index.csv "bash scripts/run.sh step0"
    echo "==> Step 3, part 2: split by patient (the honest model), 5 folds"
    $PY -m covid_ct_diagnosis.train --config 3_honest_model/patient_split.yaml
    $PY -m covid_ct_diagnosis.explain gradcam --model-dir 3_honest_model/results/patient_split
    echo
    echo "Step 3 part 2 done. Results: 3_honest_model/results/patient_split/ (metrics.json, plots/)"
    ;;

  step4)
    require $PROC/ctmd/index.csv "bash scripts/run.sh step0"
    require 3_honest_model/results/patient_split/metrics.json "bash scripts/run.sh step3_patient"
    echo "==> Demo data: each demo patient through the model of its own fold"
    $PY -m covid_ct_diagnosis.export_demo --config 4_demo/config.yaml
    echo
    echo "Step 4 data done: 4_demo/public/data/ (index.json, summary.json, one folder per patient)"
    ;;

  report)
    $PY -m covid_ct_diagnosis.report > results.md
    cat results.md
    ;;

  notebook)
    setup
    .venv/bin/jupyter lab 0_data/explore_data.ipynb
    ;;

  test)
    setup
    $PY -m pytest
    ;;

  *)
    sed -n '2,16p' "$SCRIPT"
    ;;
esac
