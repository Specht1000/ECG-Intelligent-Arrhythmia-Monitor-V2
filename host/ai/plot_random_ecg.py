"""Plot a random ECG record from the local research datasets.

By default, the script randomly chooses between the two 12-lead datasets used by
ECG V2: Chapman-Shaoxing-Ningbo and PTB-XL. MIT-BIH can be selected explicitly
for a two-lead example.
"""

import argparse
import ast
import csv
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import wfdb


DATABASE_ROOT = PROJECT_ROOT / "database"
CHAPMAN_ROOT = (
    DATABASE_ROOT
    / "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
)
PTBXL_ROOT = (
    DATABASE_ROOT
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
MITBIH_ROOT = DATABASE_ROOT / "mit_bih"

CHAPMAN_EXCLUDED_RECORDS = {"JS01052", "JS23074"}
LEAD_DISPLAY_NAMES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


def _read_chapman_condition_map() -> Dict[str, str]:
    path = CHAPMAN_ROOT / "ConditionNames_SNOMED-CT.csv"
    conditions = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            conditions[row["Snomed_CT"].strip()] = row["Full Name"].strip()
    return conditions


def _read_chapman_diagnoses(header_path: Path) -> List[str]:
    condition_map = _read_chapman_condition_map()
    diagnosis_codes = []
    with header_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#Dx:"):
                diagnosis_codes = [code.strip() for code in line[4:].split(",")]
                break
    return [condition_map.get(code, "SNOMED CT {}".format(code)) for code in diagnosis_codes]


def _random_chapman_record(rng: random.Random) -> Tuple[Path, List[str]]:
    records_file = CHAPMAN_ROOT / "RECORDS"
    directories = [line.strip() for line in records_file.read_text().splitlines() if line.strip()]
    rng.shuffle(directories)

    for relative_directory in directories:
        directory = CHAPMAN_ROOT / Path(relative_directory)
        candidates = [
            path
            for path in directory.glob("*.hea")
            if path.stem not in CHAPMAN_EXCLUDED_RECORDS
        ]
        if candidates:
            header_path = rng.choice(candidates)
            return header_path.with_suffix(""), _read_chapman_diagnoses(header_path)

    raise FileNotFoundError("No usable Chapman WFDB records were found")


def _read_ptbxl_statements() -> Dict[str, str]:
    path = PTBXL_ROOT / "scp_statements.csv"
    statements = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if row:
                statements[row[0].strip()] = row[1].strip()
    return statements


def _random_ptbxl_record(rng: random.Random) -> Tuple[Path, List[str]]:
    metadata_path = PTBXL_ROOT / "ptbxl_database.csv"
    selected_row = None
    selected_count = 0

    # Reservoir sampling avoids retaining all 21,799 metadata rows in memory.
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for selected_count, row in enumerate(csv.DictReader(handle), start=1):
            if rng.randrange(selected_count) == 0:
                selected_row = row

    if selected_row is None:
        raise ValueError("PTB-XL metadata contains no records")

    statements = _read_ptbxl_statements()
    codes = ast.literal_eval(selected_row["scp_codes"])
    labels = [statements.get(code, code) for code in codes]
    return PTBXL_ROOT / Path(selected_row["filename_hr"]), labels


def _random_mitbih_record(rng: random.Random) -> Tuple[Path, List[str]]:
    candidates = sorted(
        path for path in MITBIH_ROOT.glob("*.hea") if path.stem.isdigit()
    )
    if not candidates:
        raise FileNotFoundError("No MIT-BIH records were found")
    return rng.choice(candidates).with_suffix(""), ["Beat annotations available separately"]


def choose_record(dataset: str, rng: random.Random) -> Tuple[str, Path, List[str]]:
    """Choose a random record and return dataset name, WFDB base path, and labels."""

    if dataset == "both-12-lead":
        dataset = rng.choice(("chapman", "ptbxl"))

    if dataset == "chapman":
        record_path, labels = _random_chapman_record(rng)
        return "Chapman-Shaoxing-Ningbo", record_path, labels
    if dataset == "ptbxl":
        record_path, labels = _random_ptbxl_record(rng)
        return "PTB-XL 1.0.3", record_path, labels
    if dataset == "mitbih":
        record_path, labels = _random_mitbih_record(rng)
        return "MIT-BIH Arrhythmia Database", record_path, labels

    raise ValueError("Unsupported dataset: {}".format(dataset))


def _lead_unit(record, index: int) -> str:
    if record.units and index < len(record.units):
        return record.units[index]
    return "amplitude"


def plot_record(
    dataset_name: str,
    record_path: Path,
    labels: Sequence[str],
    output_path: Path,
    show: bool = False,
) -> None:
    """Load and plot every lead in a WFDB record."""

    record = wfdb.rdrecord(str(record_path))
    if record.p_signal is None:
        raise ValueError("WFDB record has no calibrated physical signal")

    signals = np.asarray(record.p_signal)
    if signals.ndim != 2 or signals.shape[1] != len(record.sig_name):
        raise ValueError("Unexpected signal shape: {}".format(signals.shape))

    time_seconds = np.arange(signals.shape[0], dtype=float) / float(record.fs)
    lead_count = signals.shape[1]
    column_count = 2 if lead_count > 2 else 1
    row_count = int(np.ceil(lead_count / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15, max(5.0, row_count * 2.1)),
        squeeze=False,
        sharex=True,
    )
    flat_axes = axes.ravel()

    for index, axis in enumerate(flat_axes):
        if index >= lead_count:
            axis.set_visible(False)
            continue

        lead_name = LEAD_DISPLAY_NAMES.get(record.sig_name[index], record.sig_name[index])
        axis.plot(time_seconds, signals[:, index], color="#111827", linewidth=0.75)
        axis.set_title(lead_name, loc="left", fontsize=10, fontweight="bold")
        axis.set_ylabel(_lead_unit(record, index), fontsize=8)
        axis.set_xlim(time_seconds[0], time_seconds[-1])
        axis.xaxis.set_major_locator(MultipleLocator(1.0))
        axis.xaxis.set_minor_locator(MultipleLocator(0.2))
        axis.yaxis.set_minor_locator(MultipleLocator(0.1))
        axis.grid(which="major", color="#e7a6a6", linewidth=0.6, alpha=0.75)
        axis.grid(which="minor", color="#f6d4d4", linewidth=0.35, alpha=0.8)
        axis.tick_params(labelsize=8)

    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Time (s)")

    label_text = ", ".join(labels) if labels else "No label available"
    figure.suptitle(
        "Random ECG — {} — {}\nLabels: {}".format(
            dataset_name, record.record_name, label_text
        ),
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=160, bbox_inches="tight")
    print("Dataset: {}".format(dataset_name))
    print("Record: {}".format(record.record_name))
    print("Sampling frequency: {} Hz".format(record.fs))
    print("Shape: {} samples x {} leads".format(signals.shape[0], signals.shape[1]))
    print("Labels: {}".format(label_text))
    print("Plot saved to: {}".format(output_path.resolve()))

    plt.close(figure)

    if show:
        open_image(output_path)


def open_image(path: Path) -> None:
    """Open an image with the operating system's default viewer."""

    absolute_path = str(path.resolve())
    if os.name == "nt":
        os.startfile(absolute_path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", absolute_path])
    else:
        subprocess.Popen(["xdg-open", absolute_path])


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("both-12-lead", "chapman", "ptbxl", "mitbih"),
        default="both-12-lead",
        help="Dataset used for random selection",
    )
    parser.add_argument("--seed", type=int, help="Optional reproducible random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "plots" / "random_ecg.png",
        help="PNG output path",
    )
    parser.add_argument(
        "--show", action="store_true", help="Open the PNG in the default image viewer"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    rng = random.Random(args.seed)
    dataset_name, record_path, labels = choose_record(args.dataset, rng)
    plot_record(dataset_name, record_path, labels, args.output, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
