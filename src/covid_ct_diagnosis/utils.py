"""Small helpers used by every step: settings files, randomness, device choice, saving results."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# Read a YAML settings file, then apply "key=value" overrides typed on the command line
def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")

        # Walk down nested keys such as "train.epochs"
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)

    return cfg


# Fix every random number generator so a run can be repeated exactly
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Pick the fastest available device: NVIDIA GPU, then Apple GPU (MPS), then CPU
def pick_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Save a dict as readable JSON (numpy numbers and paths are converted first)
def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_json_value)


# Convert values the json module doesn't know how to write
def _to_json_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Can't save {type(value)} as JSON")
