"""Run controlled lead-count and sampling-frequency rhythm ablations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--pretraining-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-limit", type=int, default=12000)
    parser.add_argument("--validation-limit", type=int, default=3000)
    parser.add_argument("--cache-root", type=Path, default=Path(".cache/advanced_rhythm"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/rhythm_ablations"))
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Retrain configurations that already contain both metrics.json and model.pt.",
    )
    parser.add_argument(
        "--sampling-frequencies", nargs="+", type=int, choices=(100, 250, 500), default=(100, 250, 500)
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.epochs < 1 or args.pretraining_epochs < 0:
        raise ValueError("Epoch settings must be non-negative and include at least one supervised epoch")
    training_script = Path(__file__).with_name("train_advanced_rhythm_classifier.py")
    experiments = []
    configurations = [
        (("I", "II"), 100),
        (("I", "II", "III"), 100),
    ]
    configurations.extend(
        (("I", "II", "III"), frequency)
        for frequency in args.sampling_frequencies
        if frequency != 100
    )
    for leads, frequency in configurations:
        name = "{}lead_{}hz".format(len(leads), frequency)
        output_dir = args.output_root / name
        completed = (output_dir / "metrics.json").is_file() and (output_dir / "model.pt").is_file()
        if completed and not args.rerun_completed:
            print("Skipping completed {}".format(name), flush=True)
            experiments.append(output_dir)
            continue
        command = [
            sys.executable,
            str(training_script),
            "--input-leads",
            *leads,
            "--sampling-frequency",
            str(frequency),
            "--epochs",
            str(args.epochs),
            "--pretraining-epochs",
            str(args.pretraining_epochs),
            "--batch-size",
            str(args.batch_size),
            "--training-limit",
            str(args.training_limit),
            "--validation-limit",
            str(args.validation_limit),
            "--cache-root",
            str(args.cache_root),
            "--output-dir",
            str(output_dir),
            "--disable-distillation",
        ]
        print("Running {}".format(name), flush=True)
        subprocess.run(command, check=True)
        experiments.append(output_dir)
    comparison_script = Path(__file__).with_name("compare_advanced_rhythm_experiments.py")
    subprocess.run(
        [
            sys.executable,
            str(comparison_script),
            *[str(path) for path in experiments],
            "--output",
            str(args.output_root / "comparison.csv"),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
