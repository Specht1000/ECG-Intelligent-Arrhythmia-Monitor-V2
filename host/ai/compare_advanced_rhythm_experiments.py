"""Compare advanced rhythm experiments without selecting a model from test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def load_experiment(path: Path) -> dict:
    report_path = path / "metrics.json" if path.is_dir() else path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validation = report["validation_metrics"]
    chapman = report["chapman_test_metrics"]
    ptbxl = report["ptbxl_fold10_metrics"]
    row = {
        "experiment": report_path.parent.name,
        "path": str(report_path.resolve()),
        "input_leads": ",".join(report["input_leads"]),
        "sampling_frequency_hz": report["sampling_frequency_hz"],
        "validation_macro_ap": validation["macro_average_precision"],
        "validation_macro_auroc": validation["macro_roc_auc"],
        "chapman_test_macro_ap": chapman["macro_average_precision"],
        "chapman_test_macro_auroc": chapman["macro_roc_auc"],
        "ptbxl_fold10_macro_ap": ptbxl["macro_average_precision"],
        "ptbxl_fold10_macro_auroc": ptbxl["macro_roc_auc"],
        "ptbxl_fold10_exact_match": ptbxl["exact_match_accuracy"],
    }
    for entry in ptbxl["per_class"]:
        row["ptbxl_{}_ap".format(entry["class_name"])] = entry["average_precision"]
        row["ptbxl_{}_sensitivity".format(entry["class_name"])] = entry["sensitivity"]
        row["ptbxl_{}_specificity".format(entry["class_name"])] = entry["specificity"]
    return row


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/advanced_rhythm_experiment_comparison.csv")
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    frame = pd.DataFrame([load_experiment(path) for path in args.experiments])
    frame = frame.sort_values("validation_macro_ap", ascending=False).reset_index(drop=True)
    frame.insert(0, "validation_rank", range(1, len(frame) + 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print("\nSelection rule: rank by validation macro AP; test metrics are reported only after selection.")
    print("Comparison written to {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
