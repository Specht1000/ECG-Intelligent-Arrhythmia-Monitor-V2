"""Audit cross-dataset alignment of the approved six rhythm labels."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from advanced_rhythm_data import load_ptbxl_rhythm_metadata
from train_anomaly_baseline import parse_scp_codes
from train_rhythm_classifier import (
    CLASS_NAMES,
    DEFAULT_CHAPMAN_ROOT,
    DEFAULT_PTBXL_ROOT,
    PTBXL_CODE_TO_CLASS,
    build_or_load_chapman_index,
)


DEFAULT_OUTPUT_DIR = Path("artifacts/rhythm_label_alignment")
DEFAULT_CHAPMAN_INDEX = Path(".cache/chapman_rhythm_index_all_records.csv")


def _prevalence_rows(source: str, metadata: pd.DataFrame) -> list[dict]:
    return [
        {
            "source": source,
            "class": class_name,
            "positive_count": int(metadata[class_name].sum()),
            "record_count": int(len(metadata)),
            "prevalence": float(metadata[class_name].mean()),
        }
        for class_name in CLASS_NAMES
    ]


def _cooccurrence_rows(source: str, metadata: pd.DataFrame) -> list[dict]:
    rows = []
    for first, second in combinations(CLASS_NAMES, 2):
        both = (metadata[first].astype(bool) & metadata[second].astype(bool)).to_numpy()
        rows.append(
            {
                "source": source,
                "first_class": first,
                "second_class": second,
                "cooccurrence_count": int(both.sum()),
                "cooccurrence_prevalence": float(both.mean()),
            }
        )
    return rows


def _ptb_likelihood_rows(raw_metadata: pd.DataFrame) -> list[dict]:
    values = {class_name: [] for class_name in CLASS_NAMES}
    for serialized in raw_metadata["scp_codes"]:
        for code, likelihood in parse_scp_codes(serialized).items():
            if code in PTBXL_CODE_TO_CLASS:
                values[PTBXL_CODE_TO_CLASS[code]].append(float(likelihood))
    rows = []
    for class_name, likelihoods in values.items():
        array = np.asarray(likelihoods, dtype=np.float64)
        rows.append(
            {
                "class": class_name,
                "positive_count": int(len(array)),
                "zero_likelihood_count": int(np.sum(array == 0.0)),
                "below_100_likelihood_count": int(np.sum(array < 100.0)),
                "below_80_likelihood_count": int(np.sum(array < 80.0)),
                "median_likelihood": float(np.median(array)) if len(array) else None,
                "minimum_likelihood": float(np.min(array)) if len(array) else None,
            }
        )
    return rows


def audit(
    chapman_root: Path,
    ptbxl_root: Path,
    chapman_index: Path,
    output_dir: Path,
    workers: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    chapman = build_or_load_chapman_index(chapman_root, chapman_index, workers)
    ptbxl_raw = pd.read_csv(ptbxl_root / "ptbxl_database.csv")
    ptbxl = load_ptbxl_rhythm_metadata(ptbxl_root)
    prevalence = pd.DataFrame(
        _prevalence_rows("chapman", chapman) + _prevalence_rows("ptbxl", ptbxl)
    )
    cooccurrence = pd.DataFrame(
        _cooccurrence_rows("chapman", chapman) + _cooccurrence_rows("ptbxl", ptbxl)
    )
    likelihoods = pd.DataFrame(_ptb_likelihood_rows(ptbxl_raw))
    prevalence.to_csv(output_dir / "class_prevalence.csv", index=False)
    cooccurrence.to_csv(output_dir / "label_cooccurrence.csv", index=False)
    likelihoods.to_csv(output_dir / "ptbxl_label_likelihoods.csv", index=False)
    atrial_overlap = {
        source: int(
            (
                frame["atrial_fibrillation"].astype(bool)
                & frame["atrial_flutter"].astype(bool)
            ).sum()
        )
        for source, frame in (("chapman", chapman), ("ptbxl", ptbxl))
    }
    unlabeled = {
        source: int((frame.loc[:, CLASS_NAMES].sum(axis=1) == 0).sum())
        for source, frame in (("chapman", chapman), ("ptbxl", ptbxl))
    }
    multi_target = {
        source: int((frame.loc[:, CLASS_NAMES].sum(axis=1) > 1).sum())
        for source, frame in (("chapman", chapman), ("ptbxl", ptbxl))
    }
    report = {
        "scope": list(CLASS_NAMES),
        "record_counts": {"chapman": int(len(chapman)), "ptbxl": int(len(ptbxl))},
        "records_without_approved_target": unlabeled,
        "records_with_multiple_approved_targets": multi_target,
        "atrial_fibrillation_and_flutter_overlap": atrial_overlap,
        "ptbxl_target_annotation_likelihood_distribution": {
            row["class"]: {
                "zero_likelihood_count": row["zero_likelihood_count"],
                "below_100_likelihood_count": row["below_100_likelihood_count"],
                "below_80_likelihood_count": row["below_80_likelihood_count"],
            }
            for row in likelihoods.to_dict(orient="records")
        },
        "interpretation": (
            "This audit describes dataset annotations and possible semantic noise. It does "
            "not adjudicate a medical diagnosis and does not change the approved label map. "
            "PTB-XL rhythm-statement likelihood values are reported descriptively and are "
            "not treated as confidence thresholds."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapman-root", type=Path, default=DEFAULT_CHAPMAN_ROOT)
    parser.add_argument("--ptbxl-root", type=Path, default=DEFAULT_PTBXL_ROOT)
    parser.add_argument("--chapman-index", type=Path, default=DEFAULT_CHAPMAN_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=12)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = audit(
        args.chapman_root,
        args.ptbxl_root,
        args.chapman_index,
        args.output_dir,
        args.workers,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
