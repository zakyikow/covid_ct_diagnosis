"""Collect every results folder (metrics.json) into one markdown table: all the main results at a glance.

    python -m covid_ct_diagnosis.report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Follow a chain of keys in nested dicts, or return None if one is missing
def _get(d: dict, *keys):
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


# One table row per metrics.json found under the numbered step folders. For a cross-validation, one row with
# all folds together (the single folds are skipped)
def collect(root: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(root).glob("[0-9]_*/results/**/metrics.json")):
        if any(part.startswith("fold_") for part in path.parts):
            continue
        m = json.loads(path.read_text())
        m = m.get("pooled", m)
        slices = m.get("test_slices") or m.get("slices") or {}
        patients = m.get("test_patients", {}).get("covid_yes_no") or m.get("patients") or {}
        rows.append({
            "results": str(path.parent.relative_to(root)),
            "slice accuracy": _get(slices, "all_classes", "accuracy"),
            "slice balanced accuracy": _get(slices, "all_classes", "balanced_accuracy"),
            "COVID AUROC (slices)": _get(slices, "covid_yes_no", "at_0.5", "auroc"),
            "COVID sensitivity (slices, 0.5)": _get(slices, "covid_yes_no", "at_0.5", "sensitivity"),
            "COVID AUROC (patients)": patients.get("auroc_covid_vs_rest"),
            "COVID vs. CAP AUROC (patients)": patients.get("auroc_covid_vs_cap"),
            "COVID vs. CAP lesions AUROC (slices)": _get(m, "covid_vs_other_lesions", "auroc"),
        })
    return rows


# Format the rows as a markdown table
def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "_no results found_"
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = ["–" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args(argv)
    print(to_markdown(collect(args.root)))


if __name__ == "__main__":
    main()
