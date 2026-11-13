"""Tests for choosing the Grad-CAM examples."""

import pandas as pd

from covid_ct_diagnosis.explain import _pick_examples


# Several confident slices of one patient: only the most confident one may be picked
def test_examples_come_from_different_patients():
    preds = pd.DataFrame({
        "path": [f"slice{i}.png" for i in range(6)],
        "label": ["covid"] * 6,
        "group": ["p1", "p1", "p1", "p2", "p3", "p3"],
        "logit_no_lesion": [0.0] * 6,
        "logit_covid": [5.0, 4.9, 4.8, 3.0, 2.0, 1.9],
    })

    picks = dict(_pick_examples(preds, ["no_lesion", "covid"], per_cell=3))
    picked = [row for row in picks["covid"] if row is not None]

    assert [row.group for row in picked] == ["p1", "p2", "p3"]
    assert picked[0].path == "slice0.png"
