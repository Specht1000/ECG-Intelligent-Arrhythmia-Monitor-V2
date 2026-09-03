"""Analyze advanced rhythm errors and compare them with the frozen bipolar baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from train_rhythm_classifier import CLASS_NAMES


DEFAULT_ARTIFACT_ROOT = Path("artifacts/advanced_bipolar_rhythm_classifier")
DEFAULT_BASELINE_ROOT = Path("artifacts/bipolar_rhythm_classifier")
DEFAULT_OUTPUT_DIR = Path("artifacts/advanced_bipolar_rhythm_error_analysis")


def _probability_column(frame: pd.DataFrame, class_name: str) -> str:
    calibrated = class_name + "_calibrated_probability"
    return calibrated if calibrated in frame.columns else class_name + "_probability"


def summarize_predictions(frame: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    outcomes = []
    for class_name in CLASS_NAMES:
        labels = frame[class_name + "_reference"].to_numpy(dtype=np.int64)
        predictions = frame[class_name + "_prediction"].to_numpy(dtype=np.int64)
        probabilities = frame[_probability_column(frame, class_name)].to_numpy(dtype=np.float64)
        negatives = labels == 0
        specificity = float(np.mean(predictions[negatives] == 0)) if negatives.any() else 1.0
        summaries.append(
            {
                "dataset": dataset,
                "class": class_name,
                "positive_count": int(labels.sum()),
                "average_precision": float(average_precision_score(labels, probabilities)),
                "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
                "specificity": specificity,
                "precision": float(precision_score(labels, predictions, zero_division=0)),
                "false_positive_count": int(np.sum((labels == 0) & (predictions == 1))),
                "false_negative_count": int(np.sum((labels == 1) & (predictions == 0))),
            }
        )
        error_mask = labels != predictions
        identifier_column = "record_id" if "record_id" in frame.columns else "ecg_id"
        for row_index in np.flatnonzero(error_mask):
            outcomes.append(
                {
                    "dataset": dataset,
                    "identifier": str(frame.iloc[row_index][identifier_column]),
                    "class": class_name,
                    "reference": int(labels[row_index]),
                    "prediction": int(predictions[row_index]),
                    "probability": float(probabilities[row_index]),
                    "error_type": "false_positive" if labels[row_index] == 0 else "false_negative",
                    "error_confidence": float(
                        probabilities[row_index]
                        if labels[row_index] == 0
                        else 1.0 - probabilities[row_index]
                    ),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(outcomes)


def _no_target_false_positive_rate(frame: pd.DataFrame) -> tuple[int, float]:
    reference_columns = [class_name + "_reference" for class_name in CLASS_NAMES]
    prediction_columns = [class_name + "_prediction" for class_name in CLASS_NAMES]
    no_target = frame[reference_columns].sum(axis=1).to_numpy() == 0
    any_prediction = frame[prediction_columns].sum(axis=1).to_numpy() > 0
    return int(no_target.sum()), float(any_prediction[no_target].mean()) if no_target.any() else 0.0


def _atrial_cross_confusion(frame: pd.DataFrame) -> dict:
    af = frame["atrial_fibrillation_reference"].to_numpy(dtype=bool)
    afl = frame["atrial_flutter_reference"].to_numpy(dtype=bool)
    af_prediction = frame["atrial_fibrillation_prediction"].to_numpy(dtype=bool)
    afl_prediction = frame["atrial_flutter_prediction"].to_numpy(dtype=bool)
    af_only = af & ~afl
    afl_only = afl & ~af
    return {
        "af_only_count": int(af_only.sum()),
        "af_only_predicted_as_flutter_rate": float(afl_prediction[af_only].mean()) if af_only.any() else 0.0,
        "flutter_only_count": int(afl_only.sum()),
        "flutter_only_predicted_as_af_rate": float(af_prediction[afl_only].mean()) if afl_only.any() else 0.0,
    }


def analyze(artifact_root: Path, baseline_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "chapman_heldout": pd.read_csv(artifact_root / "chapman_test_predictions.csv"),
        "ptbxl_fold10": pd.read_csv(artifact_root / "ptbxl_fold10_predictions.csv"),
    }
    summary_frames = []
    error_frames = []
    no_target = {}
    atrial_confusion = {}
    for name, frame in datasets.items():
        summaries, errors = summarize_predictions(frame, name)
        summary_frames.append(summaries)
        error_frames.append(errors)
        count, false_positive_rate = _no_target_false_positive_rate(frame)
        no_target[name] = {
            "record_count": count,
            "any_false_positive_rate": false_positive_rate,
        }
        atrial_confusion[name] = _atrial_cross_confusion(frame)
    summaries = pd.concat(summary_frames, ignore_index=True)
    errors = pd.concat(error_frames, ignore_index=True)
    summaries.to_csv(output_dir / "per_class_metrics.csv", index=False)
    errors.to_csv(output_dir / "all_errors.csv", index=False)
    errors.sort_values("error_confidence", ascending=False).head(200).to_csv(
        output_dir / "highest_confidence_errors.csv", index=False
    )

    comparison_rows = []
    baseline_metrics_path = baseline_root / "metrics.json"
    if baseline_metrics_path.is_file():
        advanced_metrics = json.loads((artifact_root / "metrics.json").read_text(encoding="utf-8"))
        baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        pairs = (
            ("chapman_heldout", "chapman_test_metrics", "chapman_test_metrics"),
            ("ptbxl_fold10", "ptbxl_fold10_metrics", "ptbxl_external_metrics"),
        )
        for dataset, advanced_key, baseline_key in pairs:
            advanced_by_class = {
                row["class_name"]: row for row in advanced_metrics[advanced_key]["per_class"]
            }
            baseline_by_class = {
                row["class_name"]: row for row in baseline_metrics[baseline_key]["per_class"]
            }
            for class_name in CLASS_NAMES:
                advanced_ap = advanced_by_class[class_name]["average_precision"]
                baseline_ap = baseline_by_class[class_name]["average_precision"]
                comparison_rows.append(
                    {
                        "dataset": dataset,
                        "class": class_name,
                        "baseline_average_precision": baseline_ap,
                        "advanced_average_precision": advanced_ap,
                        "average_precision_change": advanced_ap - baseline_ap,
                    }
                )
        pd.DataFrame(comparison_rows).to_csv(
            output_dir / "comparison_with_bipolar_baseline.csv", index=False
        )
    report = {
        "no_target_exam_errors": no_target,
        "atrial_cross_confusion": atrial_confusion,
        "comparison_with_baseline_available": bool(comparison_rows),
        "interpretation_limit": (
            "This is an engineering error analysis. PTB-XL fold 10 is a held-out test for "
            "the advanced multi-dataset model, while it was an external dataset for the "
            "Chapman-only baseline; improvements are therefore not attributable to one method."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = analyze(args.artifact_root, args.baseline_root, args.output_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
