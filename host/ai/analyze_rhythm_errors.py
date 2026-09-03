"""Analyze errors, calibration, subgroups, and domain shift for the rhythm model."""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from analyze_anomaly_errors import (
    age_group,
    apply_temperature,
    fit_temperature,
    has_signal_quality_annotation,
)
from train_rhythm_classifier import CLASS_NAMES, DISPLAY_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "bipolar_rhythm_classifier"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "bipolar_rhythm_error_analysis"
DEFAULT_CHAPMAN_ROOT = (
    PROJECT_ROOT
    / "database"
    / "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
)
DEFAULT_PTBXL_ROOT = (
    PROJECT_ROOT
    / "database"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bin_count: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    error = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bin_count - 1 else probabilities < upper
        )
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return error


def calibration_bins(
    labels: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    dataset: str,
    class_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, 11)
    for calibration, probabilities in (
        ("raw", raw_probabilities),
        ("temperature_scaled", calibrated_probabilities),
    ):
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (probabilities >= lower) & (
                probabilities <= upper if index == 9 else probabilities < upper
            )
            rows.append(
                {
                    "dataset": dataset,
                    "class_name": class_name,
                    "calibration": calibration,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": int(mask.sum()),
                    "mean_probability": float(probabilities[mask].mean()) if mask.any() else math.nan,
                    "observed_rate": float(labels[mask].mean()) if mask.any() else math.nan,
                }
            )
    return pd.DataFrame(rows)


def calibration_metrics(
    labels: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
) -> Dict[str, float]:
    clipped_raw = np.clip(raw_probabilities, 1e-7, 1.0 - 1e-7)
    clipped_calibrated = np.clip(calibrated_probabilities, 1e-7, 1.0 - 1e-7)
    return {
        "raw_ece": expected_calibration_error(labels, raw_probabilities),
        "calibrated_ece": expected_calibration_error(labels, calibrated_probabilities),
        "raw_brier": float(brier_score_loss(labels, raw_probabilities)),
        "calibrated_brier": float(brier_score_loss(labels, calibrated_probabilities)),
        "raw_log_loss": float(log_loss(labels, clipped_raw, labels=[0, 1])),
        "calibrated_log_loss": float(log_loss(labels, clipped_calibrated, labels=[0, 1])),
    }


def _read_chapman_demographics(
    dataset_root: Path, record_id: str, record_base: str
) -> Dict[str, object]:
    lines = (dataset_root / record_base).with_suffix(".hea").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    metadata: Dict[str, str] = {}
    for line in lines[13:]:
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            metadata[key.strip().lower()] = value.strip()
    try:
        age = float(metadata.get("age", "nan"))
    except ValueError:
        age = math.nan
    sex = metadata.get("sex", "Unknown").strip().title()
    if sex not in ("Male", "Female"):
        sex = "Unknown"
    return {"record_id": record_id, "age": age, "sex_group": sex}


def add_chapman_metadata(
    predictions: pd.DataFrame,
    split_manifest: pd.DataFrame,
    dataset_root: Path,
    workers: int,
) -> pd.DataFrame:
    frame = predictions.merge(
        split_manifest[["record_id", "record_base"]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    if frame["record_base"].isna().any():
        raise ValueError("Some Chapman predictions do not match the split manifest")
    rows = list(frame[["record_id", "record_base"]].itertuples(index=False, name=None))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        demographics = list(
            executor.map(
                lambda item: _read_chapman_demographics(dataset_root, item[0], item[1]),
                rows,
            )
        )
    demographics_frame = pd.DataFrame(demographics)
    frame = frame.merge(demographics_frame, on="record_id", how="left", validate="one_to_one")
    frame["identifier"] = frame["record_id"].astype(str)
    frame["age_group"] = frame["age"].apply(
        lambda value: age_group(value) if pd.notna(value) else "Unknown"
    )
    frame["signal_quality_group"] = "Not annotated"
    return frame


def add_ptbxl_metadata(predictions: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    metadata = pd.read_csv(dataset_root / "ptbxl_database.csv")
    columns = [
        "ecg_id",
        "patient_id",
        "age",
        "sex",
        "site",
        "device",
        "baseline_drift",
        "static_noise",
        "burst_noise",
        "electrodes_problems",
    ]
    frame = predictions.merge(metadata[columns], on="ecg_id", how="left", validate="one_to_one")
    if frame["patient_id"].isna().any():
        raise ValueError("Some PTB-XL predictions do not match dataset metadata")
    frame["identifier"] = frame["ecg_id"].astype(str)
    frame["sex_group"] = frame["sex"].map({0: "Male", 1: "Female"}).fillna("Unknown")
    frame["age_group"] = frame["age"].apply(
        lambda value: age_group(value) if pd.notna(value) else "Unknown"
    )
    frame["signal_quality_group"] = frame.apply(
        lambda row: (
            "Annotated signal issue"
            if has_signal_quality_annotation(row)
            else "No annotated issue"
        ),
        axis=1,
    )
    return frame


def error_table(frame: pd.DataFrame, dataset: str, thresholds: Mapping[str, float]) -> pd.DataFrame:
    rows = []
    metadata_columns = [
        "identifier",
        "age",
        "age_group",
        "sex_group",
        "signal_quality_group",
        "target_label_scope",
    ]
    for class_name, display_name in zip(CLASS_NAMES, DISPLAY_NAMES):
        reference = frame["{}_reference".format(class_name)].to_numpy(dtype=np.int64)
        probability = frame["{}_probability".format(class_name)].to_numpy(dtype=np.float64)
        prediction = frame["{}_prediction".format(class_name)].to_numpy(dtype=np.int64)
        error_type = np.select(
            [
                (reference == 0) & (prediction == 0),
                (reference == 0) & (prediction == 1),
                (reference == 1) & (prediction == 0),
                (reference == 1) & (prediction == 1),
            ],
            ["true_negative", "false_positive", "false_negative", "true_positive"],
            default="invalid",
        )
        class_frame = frame[metadata_columns].copy()
        class_frame.insert(0, "dataset", dataset)
        class_frame.insert(2, "class_name", class_name)
        class_frame.insert(3, "display_name", display_name)
        class_frame["reference"] = reference
        class_frame["probability"] = probability
        class_frame["threshold"] = thresholds[class_name]
        class_frame["prediction"] = prediction
        class_frame["error_type"] = error_type
        class_frame["distance_from_threshold"] = np.abs(
            probability - thresholds[class_name]
        )
        class_frame["error_confidence"] = np.where(reference == 0, probability, 1.0 - probability)
        rows.append(class_frame)
    return pd.concat(rows, ignore_index=True)


def binary_summary(subset: pd.DataFrame) -> Dict[str, float]:
    labels = subset["reference"].to_numpy(dtype=np.int64)
    probabilities = subset["probability"].to_numpy(dtype=np.float64)
    predictions = subset["prediction"].to_numpy(dtype=np.int64)
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    return {
        "count": int(len(subset)),
        "positive_count": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else math.nan,
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": (
            float(true_negative / (true_negative + false_positive))
            if true_negative + false_positive
            else math.nan
        ),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": (
            float(average_precision_score(labels, probabilities))
            if len(np.unique(labels)) == 2
            else math.nan
        ),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) == 2
            else math.nan
        ),
    }


def subgroup_table(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    categories = {
        "sex": "sex_group",
        "age": "age_group",
        "signal_quality": "signal_quality_group",
        "target_label_scope": "target_label_scope",
    }
    for (dataset, class_name, display_name), class_frame in errors.groupby(
        ["dataset", "class_name", "display_name"], observed=True
    ):
        for category, column in categories.items():
            for group, subset in class_frame.groupby(column, observed=True):
                if group in ("Unknown", "Not annotated"):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "class_name": class_name,
                        "display_name": display_name,
                        "category": category,
                        "group": group,
                        **binary_summary(subset),
                    }
                )
    return pd.DataFrame(rows)


def threshold_proximity_table(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, class_name), subset in errors.groupby(
        ["dataset", "class_name"], observed=True
    ):
        incorrect = subset["reference"] != subset["prediction"]
        for margin in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20):
            near = subset["distance_from_threshold"] < margin
            rows.append(
                {
                    "dataset": dataset,
                    "class_name": class_name,
                    "margin": margin,
                    "near_threshold_count": int(near.sum()),
                    "near_threshold_fraction": float(near.mean()),
                    "fraction_of_all_errors_near_threshold": (
                        float((near & incorrect).sum() / incorrect.sum())
                        if incorrect.any()
                        else math.nan
                    ),
                    "error_rate_near_threshold": (
                        float(incorrect[near].mean()) if near.any() else math.nan
                    ),
                    "error_rate_away_from_threshold": (
                        float(incorrect[~near].mean()) if (~near).any() else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def domain_shift_table(errors: pd.DataFrame) -> pd.DataFrame:
    summaries: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (dataset, class_name), subset in errors.groupby(
        ["dataset", "class_name"], observed=True
    ):
        summaries[(dataset, class_name)] = binary_summary(subset)
    rows = []
    for class_name, display_name in zip(CLASS_NAMES, DISPLAY_NAMES):
        primary = summaries[("chapman_test", class_name)]
        external = summaries[("ptbxl_external", class_name)]
        rows.append(
            {
                "class_name": class_name,
                "display_name": display_name,
                "chapman_prevalence": primary["prevalence"],
                "ptbxl_prevalence": external["prevalence"],
                "prevalence_ratio_ptbxl_to_chapman": (
                    external["prevalence"] / primary["prevalence"]
                ),
                "chapman_average_precision": primary["average_precision"],
                "ptbxl_average_precision": external["average_precision"],
                "average_precision_change": (
                    external["average_precision"] - primary["average_precision"]
                ),
                "chapman_sensitivity": primary["sensitivity"],
                "ptbxl_sensitivity": external["sensitivity"],
                "sensitivity_change": external["sensitivity"] - primary["sensitivity"],
                "chapman_precision": primary["precision"],
                "ptbxl_precision": external["precision"],
                "precision_change": external["precision"] - primary["precision"],
            }
        )
    return pd.DataFrame(rows)


def label_interaction_table(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for reference_class in CLASS_NAMES:
        reference_mask = frame["{}_reference".format(reference_class)] == 1
        for output_class in CLASS_NAMES:
            probabilities = frame.loc[
                reference_mask, "{}_probability".format(output_class)
            ]
            predictions = frame.loc[
                reference_mask, "{}_prediction".format(output_class)
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "reference_class": reference_class,
                    "output_class": output_class,
                    "reference_support": int(reference_mask.sum()),
                    "mean_output_probability": float(probabilities.mean()),
                    "positive_output_rate": float(predictions.mean()),
                    "is_target_diagonal": reference_class == output_class,
                }
            )
    return pd.DataFrame(rows)


def true_label_cooccurrence_table(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    labels = frame[["{}_reference".format(name) for name in CLASS_NAMES]].to_numpy(
        dtype=np.int64
    )
    counts = labels.T @ labels
    rows = []
    for first_index, first_class in enumerate(CLASS_NAMES):
        for second_index, second_class in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "dataset": dataset,
                    "first_class": first_class,
                    "second_class": second_class,
                    "cooccurrence_count": int(counts[first_index, second_index]),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(
    domain_shift: pd.DataFrame,
    calibration: pd.DataFrame,
    errors: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    positions = np.arange(len(CLASS_NAMES))
    width = 0.38
    labels = domain_shift["display_name"]

    axes[0, 0].bar(
        positions - width / 2,
        domain_shift["chapman_average_precision"],
        width,
        label="Chapman held-out",
    )
    axes[0, 0].bar(
        positions + width / 2,
        domain_shift["ptbxl_average_precision"],
        width,
        label="PTB-XL external",
    )
    axes[0, 0].set(title="Average precision and domain shift", ylim=(0, 1))
    axes[0, 0].legend()

    axes[0, 1].bar(
        positions - width / 2,
        domain_shift["chapman_prevalence"],
        width,
        label="Chapman held-out",
    )
    axes[0, 1].bar(
        positions + width / 2,
        domain_shift["ptbxl_prevalence"],
        width,
        label="PTB-XL external",
    )
    axes[0, 1].set(title="Positive-label prevalence", ylim=(0, 1))
    axes[0, 1].legend()

    external_calibration = calibration.loc[calibration["dataset"] == "ptbxl_external"]
    axes[1, 0].bar(
        positions - width / 2,
        external_calibration["raw_ece"],
        width,
        label="Raw",
    )
    axes[1, 0].bar(
        positions + width / 2,
        external_calibration["calibrated_ece"],
        width,
        label="Temperature-scaled",
    )
    axes[1, 0].set(title="PTB-XL expected calibration error", ylabel="ECE")
    axes[1, 0].legend()

    error_counts = (
        errors.loc[errors["error_type"].isin(("false_positive", "false_negative"))]
        .groupby(["class_name", "error_type"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(CLASS_NAMES)
    )
    axes[1, 1].bar(
        positions - width / 2,
        error_counts.get("false_positive", pd.Series(0, index=CLASS_NAMES)),
        width,
        label="False positive",
    )
    axes[1, 1].bar(
        positions + width / 2,
        error_counts.get("false_negative", pd.Series(0, index=CLASS_NAMES)),
        width,
        label="False negative",
    )
    axes[1, 1].set(title="Combined held-out error counts")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
    figure.suptitle("Rhythm classifier error analysis")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chapman-root", type=Path, default=DEFAULT_CHAPMAN_ROOT)
    parser.add_argument("--ptbxl-root", type=Path, default=DEFAULT_PTBXL_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    artifact_root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads((artifact_root / "metrics.json").read_text(encoding="utf-8"))
    thresholds = {name: float(report["thresholds"][name]) for name in CLASS_NAMES}
    validation = pd.read_csv(artifact_root / "chapman_validation_predictions.csv")
    chapman_test = pd.read_csv(artifact_root / "chapman_test_predictions.csv")
    external = pd.read_csv(artifact_root / "ptbxl_external_predictions.csv")
    manifest = pd.read_csv(artifact_root / "chapman_split_manifest.csv")

    chapman_test = add_chapman_metadata(
        chapman_test, manifest, args.chapman_root.resolve(), args.workers
    )
    external = add_ptbxl_metadata(external, args.ptbxl_root.resolve())
    reference_columns = ["{}_reference".format(name) for name in CLASS_NAMES]
    prediction_columns = ["{}_prediction".format(name) for name in CLASS_NAMES]
    for frame in (chapman_test, external):
        frame["target_label_scope"] = np.where(
            frame[reference_columns].sum(axis=1) > 0,
            "At least one target label",
            "No target label",
        )
    errors = pd.concat(
        [
            error_table(chapman_test, "chapman_test", thresholds),
            error_table(external, "ptbxl_external", thresholds),
        ],
        ignore_index=True,
    )
    errors.to_csv(output_dir / "all_class_outcomes.csv", index=False)
    errors.loc[errors["error_type"].isin(("false_positive", "false_negative"))].to_csv(
        output_dir / "all_errors.csv", index=False
    )
    highest_confidence = (
        errors.loc[errors["error_type"].isin(("false_positive", "false_negative"))]
        .sort_values("error_confidence", ascending=False)
        .groupby(["dataset", "class_name", "error_type"], observed=True)
        .head(25)
    )
    highest_confidence.to_csv(output_dir / "highest_confidence_errors.csv", index=False)

    calibration_rows = []
    calibration_bin_frames = []
    temperatures: Dict[str, float] = {}
    for class_name, display_name in zip(CLASS_NAMES, DISPLAY_NAMES):
        validation_labels = validation["{}_reference".format(class_name)].to_numpy(dtype=np.int64)
        validation_probabilities = validation[
            "{}_probability".format(class_name)
        ].to_numpy(dtype=np.float64)
        temperature = fit_temperature(validation_labels, validation_probabilities)
        temperatures[class_name] = temperature
        for dataset, frame in (("chapman_test", chapman_test), ("ptbxl_external", external)):
            labels = frame["{}_reference".format(class_name)].to_numpy(dtype=np.int64)
            probabilities = frame["{}_probability".format(class_name)].to_numpy(dtype=np.float64)
            calibrated = apply_temperature(probabilities, temperature)
            calibration_rows.append(
                {
                    "dataset": dataset,
                    "class_name": class_name,
                    "display_name": display_name,
                    "temperature": temperature,
                    "positive_count": int(labels.sum()),
                    **calibration_metrics(labels, probabilities, calibrated),
                }
            )
            calibration_bin_frames.append(
                calibration_bins(labels, probabilities, calibrated, dataset, class_name)
            )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(output_dir / "calibration_metrics.csv", index=False)
    pd.concat(calibration_bin_frames, ignore_index=True).to_csv(
        output_dir / "calibration_bins.csv", index=False
    )

    subgroups = subgroup_table(errors)
    subgroups.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    proximity = threshold_proximity_table(errors)
    proximity.to_csv(output_dir / "threshold_proximity.csv", index=False)
    domain_shift = domain_shift_table(errors)
    domain_shift.to_csv(output_dir / "domain_shift.csv", index=False)
    interactions = pd.concat(
        [
            label_interaction_table(chapman_test, "chapman_test"),
            label_interaction_table(external, "ptbxl_external"),
        ],
        ignore_index=True,
    )
    interactions.to_csv(output_dir / "label_interactions.csv", index=False)
    cooccurrences = pd.concat(
        [
            true_label_cooccurrence_table(chapman_test, "chapman_test"),
            true_label_cooccurrence_table(external, "ptbxl_external"),
        ],
        ignore_index=True,
    )
    cooccurrences.to_csv(output_dir / "true_label_cooccurrence.csv", index=False)
    plot_summary(domain_shift, calibration, errors, output_dir / "summary.png")

    weakest_external = domain_shift.sort_values("ptbxl_average_precision").iloc[0]
    largest_ap_drop = domain_shift.sort_values("average_precision_change").iloc[0]
    calibration_improvement_count = int(
        (calibration["calibrated_ece"] < calibration["raw_ece"]).sum()
    )
    external_without_target = external[reference_columns].sum(axis=1) == 0
    external_any_prediction = external[prediction_columns].sum(axis=1) > 0
    afib_on_flutter = interactions.loc[
        (interactions["dataset"] == "chapman_test")
        & (interactions["reference_class"] == "atrial_flutter")
        & (interactions["output_class"] == "atrial_fibrillation")
    ].iloc[0]
    flutter_on_afib = interactions.loc[
        (interactions["dataset"] == "chapman_test")
        & (interactions["reference_class"] == "atrial_fibrillation")
        & (interactions["output_class"] == "atrial_flutter")
    ].iloc[0]
    summary = {
        "analysis": "Six-label rhythm error, calibration, subgroup, and domain-shift analysis",
        "intended_use": "Exploratory research analysis only; no clinical abstention policy is defined.",
        "data_policy": {
            "temperature_fitting": "One temperature per class fitted on Chapman validation predictions only.",
            "evaluation": "Calibration is reported on Chapman held-out test and PTB-XL external fold 10.",
            "threshold_proximity": "Descriptive analysis only; margins are not an approved abstention policy.",
        },
        "temperatures": temperatures,
        "record_counts": {
            "chapman_test": int(len(chapman_test)),
            "ptbxl_external": int(len(external)),
            "total_binary_outcomes": int(len(errors)),
            "total_errors": int(
                errors["error_type"].isin(("false_positive", "false_negative")).sum()
            ),
        },
        "findings": {
            "weakest_external_average_precision_class": str(weakest_external["class_name"]),
            "weakest_external_average_precision": float(weakest_external["ptbxl_average_precision"]),
            "largest_average_precision_drop_class": str(largest_ap_drop["class_name"]),
            "largest_average_precision_change": float(largest_ap_drop["average_precision_change"]),
            "temperature_scaling_ece_improvements_out_of_12": calibration_improvement_count,
            "ptbxl_no_target_exam_count": int(external_without_target.sum()),
            "ptbxl_no_target_exam_any_false_positive_rate": float(
                external_any_prediction[external_without_target].mean()
            ),
            "chapman_afib_positive_rate_on_flutter_records": float(
                afib_on_flutter["positive_output_rate"]
            ),
            "chapman_flutter_positive_rate_on_afib_records": float(
                flutter_on_afib["positive_output_rate"]
            ),
        },
        "limitations": [
            "Subgroup metrics are retrospective and may have small positive support.",
            "Chapman has no separate patient identifier and no signal-quality annotation used here.",
            "PTB-XL atrial-flutter support is only seven positive records.",
            "Temperature scaling does not correct label or acquisition domain shift.",
            "No uncertainty margin or abstention policy has been clinically approved.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("Analyzed {:,} binary outcomes".format(len(errors)))
    print("Recorded {:,} false-positive or false-negative outcomes".format(summary["record_counts"]["total_errors"]))
    print(
        "Temperature scaling improved ECE in {}/12 dataset-class evaluations".format(
            calibration_improvement_count
        )
    )
    print(
        "Weakest external AP: {} ({:.4f})".format(
            weakest_external["display_name"], weakest_external["ptbxl_average_precision"]
        )
    )
    print("Artifacts written to {}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
