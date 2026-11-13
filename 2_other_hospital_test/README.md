# Step 2 · Test the baseline somewhere else

> **Question:** does the step 1 model still work on scans from another place?<br>
> **Answer:** no. It **misses 48% of the COVID-19 slices**, and it can't tell COVID-19 from pneumonia.

## What happens

- **Model:** the step 1 model, unchanged. Nothing is trained here.
- **Data:** all 305 patients of the Tehran dataset (COVID-CT-MD): another country, another scanner, and patients
  the model has never seen.
- **Score:** a slice's COVID-19 score is the model's probability for its COVID-19 class.

## Result

| Per slice | |
|---|---:|
| COVID-19 slices found (50% cut-off) | **52%** (1,991 of 3,815) |
| Pneumonia slices called COVID-19 | 59% (692 of 1,178) |
| AUROC, COVID-19 lesion vs. pneumonia lesion | **0.46**, a coin flip |

| Per patient (AUROC) | |
|---|---:|
| COVID-19 vs. healthy | 0.90 |
| COVID-19 vs. pneumonia | **0.54** |

It still roughly spots abnormal lungs, but it can't tell which infection it is. Likely reasons: it never saw
pneumonia in training, and the two datasets differ in scanners, image processing and patients.

> AUROC: the chance that a random case of one group scores higher than a random case of the other. 0.5 is a coin
> flip, 1 is perfect.

## Files

| File | What's inside |
|---|---|
| `config.yaml` | the two tests: all slices, and infected slices only |
| `results/all_patients/` | results per slice and per patient, ROC curves |
| `results/covid_vs_pneumonia_lesions/` | COVID-19 vs. pneumonia, on infected slices only |

The predictions are not stored; the command below recreates them.

## Run

```bash
bash scripts/run.sh step2    # after step0 and step1
```
