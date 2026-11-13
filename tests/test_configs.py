"""Checks on the settings files."""

import yaml


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


# The two Step 3 trainings must be identical except for the split, otherwise the comparison isn't fair
def test_step3_trainings_differ_only_in_the_split():
    random_split = _load("3_honest_model/random_split.yaml")
    patient_split = _load("3_honest_model/patient_split.yaml")

    for key in ("seed", "classes", "positive_class", "model", "augment", "train"):
        assert random_split[key] == patient_split[key], key

    assert random_split["data"]["split"] == "random"
    assert patient_split["data"]["split"] == "group"
    assert {**random_split["data"], "split": None} == {**patient_split["data"], "split": None}


# Step 2 tests the model trained in Step 1
def test_step2_uses_the_step1_model():
    step1 = _load("1_reproduce_baseline/config.yaml")
    step2 = _load("2_other_hospital_test/config.yaml")
    assert step2["model_dir"] == step1["output_dir"]


# Step 4 shows the honest model (the by-patient cross-validation of Step 3), with 6 different demo patients
def test_step4_uses_the_honest_model():
    patient_split = _load("3_honest_model/patient_split.yaml")
    demo = _load("4_demo/config.yaml")
    assert demo["cv_dir"] == patient_split["output_dir"]
    assert demo["index"] == patient_split["data"]["index"]
    assert len(set(demo["patients"])) == len(demo["patients"]) == 6
