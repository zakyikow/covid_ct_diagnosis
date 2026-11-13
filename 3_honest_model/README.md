# Step 3 · The leak (3a) and the honest model (3b)

> **3a. How much of the score is the leak?** **19 points**: 96% → 77% balanced accuracy per slice.<br>
> **3b. What is the honest score?** **84% balanced accuracy per patient**.

## What happens

- **Data:** the Tehran dataset (COVID-CT-MD), which has patient IDs.
- **Two trainings, identical except for the split** (an automatic test checks this):

  | `random_split.yaml` | `patient_split.yaml` |
  |---|---|
  | slices shuffled at random, the usual way | each patient only in training or only in test |
  | → the leaky score | → **the honest model** |

- **5-fold cross-validation:** the data is dealt into 5 folds. Each run tests on one fold, validates on the next
  and trains on the other 3. So every patient is tested exactly once, by a model that never saw it.
- **Per slice**, the model says: no infection, COVID-19 lesion or pneumonia lesion.
- **Per patient**, the scores of the 10% most suspicious slices become one diagnosis: COVID-19, pneumonia or
  healthy.

## Result

| 3a · Per slice | Random split | Split by patient |
|---|---:|---:|
| Balanced accuracy | 96% | **77%** |
| COVID-19 lesions found | 95% | 72% |
| Pneumonia lesions found | 97% | 64% |

| 3b · Per patient, honest model | Found |
|---|---:|
| COVID-19 | 130 of 169 (77%) |
| Pneumonia | 46 of 60 (77%) |
| Healthy | 75 of 76 (99%) |
| **Balanced accuracy** | **84%** (95% CI 80–88%) |

The mistakes, the screening setting and the heatmaps are in the [main README](../README.md#results).

## Files

`results/random_split/` and `results/patient_split/` have the same layout:

| File | What's inside |
|---|---|
| `metrics.json` | the 5 folds together, and each number per fold |
| `plots/` | confusion matrices, ROC curves, calibration, and Grad-CAM examples (by patient) |
| `patients.csv` | by patient: all 305 patients, with their scores, diagnosis and fold |
| `folds.csv` | by patient: the fold of each patient |
| `fold_1/` … `fold_5/` | each run: settings, training history, results, plots |

The trained models (17 MB each) and the predictions are not stored here. The 5 honest models and their
predictions are in the [release](https://github.com/zakyikow/covid_ct_diagnosis/releases).

## Run

```bash
bash scripts/run.sh step3_random     # after step0
bash scripts/run.sh step3_patient
```

Each command trains 5 models: by far the longest step. If it stops, the same command continues at the next fold.
