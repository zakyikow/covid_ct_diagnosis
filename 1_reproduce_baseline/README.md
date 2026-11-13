# Step 1 · Reproduce the 99% baseline

> **Question:** does the popular COVID-19 CT classifier really reach 99%?<br>
> **Answer:** yes, **99.3% accuracy**, but only because the test slices come from the same patients as the training
> slices.

## What happens

- **Data:** the Wuhan dataset (iCTCF), 19,685 CT slices of 3 kinds: normal, COVID-19, and no lungs visible.
- **Split:** slices shuffled at random into training, validation and test (70 / 15 / 15), as most public projects do.
- **Model:** MobileNetV3, pretrained on ImageNet, trained for 10 epochs.

## Result

| On the 2,953 test slices | |
|---|---:|
| Accuracy | **99.3%** (21 mistakes) |
| COVID-19 slices found | 594 of 600 |
| False alarms on normal slices | 0 of 1,497 |

The ~99% is reproduced. But the dataset has no patient IDs, so the same patients end up in both the training and
test sets. [Step 3](../3_honest_model/) measures how much that inflates the score.

## Files

| File | What's inside |
|---|---|
| `config.yaml` | the settings |
| `results/metrics.json` | all the results |
| `results/history.csv` | loss and validation score at each epoch |
| `results/calibration.json` | calibration temperature and COVID-19 cut-off, both chosen on validation |
| `results/plots/` | confusion matrix, ROC curve, calibration |

The trained model and the predictions are not stored; the command below recreates them.

## Run

```bash
bash scripts/run.sh step1    # after step0
```
