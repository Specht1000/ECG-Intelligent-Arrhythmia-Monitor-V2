"""Analyze errors, subgroups, calibration, and uncertainty for the anomaly baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_anomaly_baseline import (
    DEFAULT_DATASET_ROOT,
    expected_calibration_error,
    load_diagnostic_class_map,
    load_low_resolution_record,
    parse_scp_codes,
)


DEFAULT_ARTIFACT_ROOT = Path("artifacts/anomaly_baseline")
AGE_GROUP_ORDER = ("0-39", "40-59", "60-79", "80+ (includes privacy-coded 90+)")
LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def age_group(age: float) -> str:
    """Group age while treating PTB-XL HIPAA-coded ages above 120 as 90+."""

    normalized_age = 90.0 if age > 120 else age
    if normalized_age < 40:
        return "0-39"
    if normalized_age < 60:
        return "40-59"
    if normalized_age < 80:
        return "60-79"
    return "80+ (includes privacy-coded 90+)"


def has_annotation(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(str(value).strip().strip(","))


def has_signal_quality_annotation(row: pd.Series) -> bool:
    fields = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems")
    return any(has_annotation(row[field]) for field in fields)


def probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def fit_temperature(labels: np.ndarray, probabilities: np.ndarray) -> float:
    logits = probability_to_logit(probabilities)

    def negative_log_likelihood(temperature: float) -> float:
        calibrated = np.clip(sigmoid(logits / temperature), 1e-7, 1.0 - 1e-7)
        return float(
            -np.mean(labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated))
        )

    result = minimize_scalar(negative_log_likelihood, bounds=(0.05, 10.0), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization failed: {}".format(result.message))
    return float(result.x)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    return sigmoid(probability_to_logit(probabilities) / temperature)


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Dict[str, object]:
    labels = labels.astype(np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    result: Dict[str, object] = {
        "count": int(len(labels)),
        "abnormal_prevalence": float(labels.mean()) if len(labels) else math.nan,
        "accuracy": float(accuracy_score(labels, predictions)) if len(labels) else math.nan,
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if len(np.unique(labels)) == 2
            else math.nan
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)) if len(labels) else math.nan,
        "precision": (
            float(precision_score(labels, predictions, zero_division=0)) if len(labels) else math.nan
        ),
        "sensitivity": (
            float(recall_score(labels, predictions, zero_division=0)) if len(labels) else math.nan
        ),
        "specificity": (
            float(true_negative / (true_negative + false_positive))
            if true_negative + false_positive
            else math.nan
        ),
        "confusion_matrix": matrix.tolist(),
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["average_precision"] = float(average_precision_score(labels, probabilities))
    else:
        result["roc_auc"] = math.nan
        result["average_precision"] = math.nan
    return result


def calibration_table(
    labels: np.ndarray, raw_probabilities: np.ndarray, calibrated_probabilities: np.ndarray
) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0.0, 1.0, 11)
    for calibration_name, probabilities in (
        ("raw", raw_probabilities),
        ("temperature_scaled", calibrated_probabilities),
    ):
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (probabilities >= lower) & (
                probabilities <= upper if index == len(edges) - 2 else probabilities < upper
            )
            rows.append(
                {
                    "calibration": calibration_name,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": int(mask.sum()),
                    "mean_probability": float(probabilities[mask].mean()) if mask.any() else math.nan,
                    "observed_abnormal_rate": float(labels[mask].mean()) if mask.any() else math.nan,
                }
            )
    return pd.DataFrame(rows)


def uncertainty_curve(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> pd.DataFrame:
    rows = []
    for margin in np.linspace(0.0, 0.35, 71):
        retained = np.abs(probabilities - threshold) >= margin
        if retained.sum() < 2 or len(np.unique(labels[retained])) < 2:
            continue
        metrics = binary_metrics(labels[retained], probabilities[retained], threshold)
        rows.append(
            {
                "margin": float(margin),
                "coverage": float(retained.mean()),
                "retained_count": int(retained.sum()),
                "uncertain_count": int((~retained).sum()),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
            }
        )
    return pd.DataFrame(rows)


def select_uncertainty_margin(validation_curve: pd.DataFrame) -> Tuple[float, str]:
    eligible = validation_curve.loc[
        (validation_curve["balanced_accuracy"] >= 0.90) & (validation_curve["coverage"] >= 0.50)
    ]
    if not eligible.empty:
        selected = eligible.sort_values(["coverage", "margin"], ascending=[False, True]).iloc[0]
        return float(selected["margin"]), "Validation balanced accuracy >= 0.90 with maximum coverage"
    selected = validation_curve.sort_values(
        ["balanced_accuracy", "coverage"], ascending=[False, False]
    ).iloc[0]
    return float(selected["margin"]), "Maximum validation balanced accuracy (target 0.90 unavailable)"


def add_analysis_columns(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    class_map: Mapping[str, str],
    threshold: float,
) -> pd.DataFrame:
    frame = predictions.merge(metadata, on="ecg_id", how="left", validate="one_to_one")
    if frame["patient_id"].isna().any():
        raise ValueError("Some predictions do not match PTB-XL metadata")

    codes: List[str] = []
    classes: List[str] = []
    for value in frame["scp_codes"]:
        record_codes = parse_scp_codes(value)
        codes.append("|".join(sorted(record_codes)))
        classes.append("|".join(sorted({class_map[code] for code in record_codes if code in class_map})))
    frame["diagnostic_codes"] = codes
    frame["diagnostic_classes"] = classes
    frame["predicted_label"] = (frame["abnormal_probability"] >= threshold).astype(np.int64)
    frame["error_type"] = np.select(
        [
            (frame["reference_label"] == 0) & (frame["predicted_label"] == 0),
            (frame["reference_label"] == 0) & (frame["predicted_label"] == 1),
            (frame["reference_label"] == 1) & (frame["predicted_label"] == 0),
            (frame["reference_label"] == 1) & (frame["predicted_label"] == 1),
        ],
        ["true_negative", "false_positive", "false_negative", "true_positive"],
        default="invalid",
    )
    frame["sex_group"] = frame["sex"].map({0: "Male", 1: "Female"}).fillna("Unknown")
    frame["age_group"] = frame["age"].apply(age_group)
    frame["signal_quality_group"] = frame.apply(
        lambda row: "Annotated signal issue" if has_signal_quality_annotation(row) else "No annotated issue",
        axis=1,
    )
    frame["distance_from_threshold"] = np.abs(frame["abnormal_probability"] - threshold)
    return frame


def subgroup_metrics(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    group_columns = {
        "sex": "sex_group",
        "age": "age_group",
        "signal_quality": "signal_quality_group",
    }
    for category, column in group_columns.items():
        for group, subset in frame.groupby(column, observed=True):
            metrics = binary_metrics(
                subset["reference_label"].to_numpy(),
                subset["abnormal_probability"].to_numpy(),
                threshold,
            )
            rows.append({"category": category, "group": group, **metrics})
    return pd.DataFrame(rows)


def label_detection_metrics(
    frame: pd.DataFrame,
    statements: pd.DataFrame,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    predictions = frame["abnormal_probability"].to_numpy() >= threshold
    superclass_rows = []
    for diagnostic_class in ("MI", "STTC", "CD", "HYP"):
        mask = frame["diagnostic_classes"].str.split("|").apply(
            lambda values: diagnostic_class in values
        )
        superclass_rows.append(
            {
                "diagnostic_class": diagnostic_class,
                "count": int(mask.sum()),
                "detection_sensitivity": float(predictions[mask].mean()) if mask.any() else math.nan,
                "mean_abnormal_probability": (
                    float(frame.loc[mask, "abnormal_probability"].mean()) if mask.any() else math.nan
                ),
                "false_negative_count": int((~predictions[mask]).sum()),
            }
        )

    code_rows = []
    diagnostic_statements = statements.loc[statements["diagnostic"] == 1.0]
    code_sets = frame["diagnostic_codes"].str.split("|")
    for code, statement in diagnostic_statements.iterrows():
        if statement.get("diagnostic_class") == "NORM":
            continue
        mask = code_sets.apply(lambda values: str(code) in values)
        if mask.sum() < 10:
            continue
        code_rows.append(
            {
                "scp_code": str(code),
                "description": statement.get("description", ""),
                "diagnostic_class": statement.get("diagnostic_class", ""),
                "count": int(mask.sum()),
                "detection_sensitivity": float(predictions[mask].mean()),
                "mean_abnormal_probability": float(frame.loc[mask, "abnormal_probability"].mean()),
                "false_negative_count": int((~predictions[mask]).sum()),
            }
        )
    code_frame = pd.DataFrame(code_rows).sort_values(
        ["detection_sensitivity", "count"], ascending=[True, False]
    )
    return pd.DataFrame(superclass_rows), code_frame


def plot_summary(
    subgroup_frame: pd.DataFrame,
    superclass_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    validation_uncertainty: pd.DataFrame,
    test_uncertainty: pd.DataFrame,
    selected_margin: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    bars = superclass_frame.sort_values("detection_sensitivity")
    axes[0, 0].barh(bars["diagnostic_class"], bars["detection_sensitivity"], color="#4472C4")
    axes[0, 0].set(xlim=(0, 1), xlabel="Detection sensitivity", title="Sensitivity by diagnostic superclass")
    for index, row in enumerate(bars.itertuples()):
        axes[0, 0].text(row.detection_sensitivity + 0.01, index, "n={}".format(row.count), va="center")

    subgroup_plot = subgroup_frame.loc[subgroup_frame["count"] >= 20].copy()
    subgroup_plot["label"] = subgroup_plot["category"] + ": " + subgroup_plot["group"]
    subgroup_plot = subgroup_plot.sort_values("balanced_accuracy")
    axes[0, 1].barh(subgroup_plot["label"], subgroup_plot["balanced_accuracy"], color="#70AD47")
    axes[0, 1].set(xlim=(0.5, 1), xlabel="Balanced accuracy", title="Held-out subgroup performance")

    for name, subset in calibration_frame.groupby("calibration", sort=False):
        valid = subset["count"] > 0
        axes[1, 0].plot(
            subset.loc[valid, "mean_probability"],
            subset.loc[valid, "observed_abnormal_rate"],
            marker="o",
            label=name.replace("_", " ").title(),
        )
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1, 0].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed abnormal rate",
        title="Test reliability diagram",
    )
    axes[1, 0].legend()

    axes[1, 1].plot(
        validation_uncertainty["coverage"],
        validation_uncertainty["balanced_accuracy"],
        label="Validation",
    )
    axes[1, 1].plot(
        test_uncertainty["coverage"], test_uncertainty["balanced_accuracy"], label="Test"
    )
    selected = test_uncertainty.iloc[(test_uncertainty["margin"] - selected_margin).abs().argsort()[:1]]
    axes[1, 1].scatter(selected["coverage"], selected["balanced_accuracy"], color="red", zorder=3)
    axes[1, 1].set(
        xlabel="Coverage after rejecting uncertain exams",
        ylabel="Balanced accuracy on retained exams",
        title="Exploratory uncertainty rejection",
    )
    axes[1, 1].legend()

    figure.suptitle("PTB-XL anomaly baseline error analysis")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def diagnostic_title(row: pd.Series, statements: pd.DataFrame, maximum_codes: int = 6) -> str:
    labels = []
    for code in row["diagnostic_codes"].split("|"):
        if code in statements.index:
            description = str(statements.loc[code].get("description", code))
            labels.append("{} ({})".format(code, description))
        else:
            labels.append(code)
    if len(labels) > maximum_codes:
        labels = labels[:maximum_codes] + ["..."]
    return ", ".join(labels)


def plot_false_negative_examples(
    false_negatives: pd.DataFrame,
    dataset_root: Path,
    statements: pd.DataFrame,
    threshold: float,
    pdf_path: Path,
    montage_path: Path,
    example_count: int = 10,
) -> None:
    examples = false_negatives.nsmallest(example_count, "abnormal_probability").copy()
    time_seconds = np.arange(1000) / 100.0
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for _, row in examples.iterrows():
            waveform = load_low_resolution_record(dataset_root / row["filename_lr"])
            figure, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
            for lead_index, axis in enumerate(axes.flat):
                axis.plot(time_seconds, waveform[lead_index], linewidth=0.7, color="#1f4e79")
                axis.set_title(LEAD_NAMES[lead_index], loc="left", fontsize=9)
                axis.grid(alpha=0.2)
            figure.supxlabel("Time (s)")
            figure.supylabel("Amplitude (mV)")
            figure.suptitle(
                "False negative ECG {} | abnormal probability {:.3f} | threshold {:.3f}\n{}".format(
                    int(row["ecg_id"]), row["abnormal_probability"], threshold, diagnostic_title(row, statements)
                ),
                fontsize=11,
            )
            figure.tight_layout(rect=(0.03, 0.03, 1, 0.93))
            pdf.savefig(figure)
            plt.close(figure)

    montage_leads = ((1, "II"), (6, "V1"), (10, "V5"))
    figure, axes = plt.subplots(len(examples), len(montage_leads), figsize=(15, 2.0 * len(examples)), sharex=True)
    if len(examples) == 1:
        axes = np.asarray([axes])
    for row_index, (_, row) in enumerate(examples.iterrows()):
        waveform = load_low_resolution_record(dataset_root / row["filename_lr"])
        for column, (lead_index, lead_name) in enumerate(montage_leads):
            axis = axes[row_index, column]
            axis.plot(time_seconds, waveform[lead_index], linewidth=0.7, color="#1f4e79")
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(lead_name)
            if column == 0:
                axis.set_ylabel(
                    "ECG {}\np={:.3f}".format(int(row["ecg_id"]), row["abnormal_probability"]),
                    rotation=0,
                    ha="right",
                    va="center",
                )
    figure.supxlabel("Time (s)")
    figure.suptitle("Most confident false negatives: representative leads")
    figure.tight_layout(rect=(0.08, 0.03, 1, 0.98))
    figure.savefig(montage_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/anomaly_error_analysis"))
    parser.add_argument("--false-negative-examples", type=int, default=10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    artifact_root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_report = json.loads((artifact_root / "metrics.json").read_text(encoding="utf-8"))
    raw_threshold = float(baseline_report["validation_metrics"]["threshold"])
    metadata = pd.read_csv(dataset_root / "ptbxl_database.csv")
    statements = pd.read_csv(dataset_root / "scp_statements.csv", index_col=0)
    statements.index = statements.index.astype(str)
    class_map = load_diagnostic_class_map(dataset_root / "scp_statements.csv")
    validation_predictions = pd.read_csv(artifact_root / "validation_predictions.csv")
    test_predictions = pd.read_csv(artifact_root / "test_predictions.csv")

    validation_labels = validation_predictions["reference_label"].to_numpy(dtype=np.int64)
    validation_raw = validation_predictions["abnormal_probability"].to_numpy(dtype=np.float64)
    test_labels = test_predictions["reference_label"].to_numpy(dtype=np.int64)
    test_raw = test_predictions["abnormal_probability"].to_numpy(dtype=np.float64)

    temperature = fit_temperature(validation_labels, validation_raw)
    validation_calibrated = apply_temperature(validation_raw, temperature)
    test_calibrated = apply_temperature(test_raw, temperature)
    calibrated_threshold = float(apply_temperature(np.asarray([raw_threshold]), temperature)[0])

    test_frame = add_analysis_columns(test_predictions, metadata, class_map, raw_threshold)
    subgroup_frame = subgroup_metrics(test_frame, raw_threshold)
    superclass_frame, code_frame = label_detection_metrics(
        test_frame, statements, raw_threshold
    )
    calibration_frame = calibration_table(test_labels, test_raw, test_calibrated)
    validation_uncertainty = uncertainty_curve(
        validation_labels, validation_calibrated, calibrated_threshold
    )
    test_uncertainty = uncertainty_curve(test_labels, test_calibrated, calibrated_threshold)
    selected_margin, margin_rule = select_uncertainty_margin(validation_uncertainty)

    validation_retained = np.abs(validation_calibrated - calibrated_threshold) >= selected_margin
    test_retained = np.abs(test_calibrated - calibrated_threshold) >= selected_margin
    uncertainty_report = {
        "selection_rule": margin_rule,
        "selected_on": "validation",
        "calibrated_probability_margin": selected_margin,
        "calibrated_lower_bound": calibrated_threshold - selected_margin,
        "calibrated_upper_bound": calibrated_threshold + selected_margin,
        "validation_coverage": float(validation_retained.mean()),
        "validation_metrics_on_retained": binary_metrics(
            validation_labels[validation_retained],
            validation_calibrated[validation_retained],
            calibrated_threshold,
        ),
        "test_coverage": float(test_retained.mean()),
        "test_uncertain_count": int((~test_retained).sum()),
        "test_metrics_on_retained": binary_metrics(
            test_labels[test_retained], test_calibrated[test_retained], calibrated_threshold
        ),
    }

    test_frame["calibrated_abnormal_probability"] = test_calibrated
    test_frame["uncertainty_decision"] = np.where(
        np.abs(test_calibrated - calibrated_threshold) < selected_margin,
        "uncertain",
        np.where(test_calibrated >= calibrated_threshold, "abnormal", "normal"),
    )

    raw_calibration = {
        "brier_score": float(np.mean(np.square(test_raw - test_labels))),
        "expected_calibration_error_10_bins": expected_calibration_error(test_labels, test_raw),
    }
    calibrated_calibration = {
        "temperature_selected_on_validation": temperature,
        "calibrated_threshold_equivalent_to_raw_threshold": calibrated_threshold,
        "brier_score": float(np.mean(np.square(test_calibrated - test_labels))),
        "expected_calibration_error_10_bins": expected_calibration_error(
            test_labels, test_calibrated
        ),
    }

    report = {
        "analysis": "PTB-XL anomaly baseline held-out error analysis",
        "intended_use": "Research analysis only; not a diagnostic or clinical-use result.",
        "test_record_count": int(len(test_frame)),
        "raw_threshold": raw_threshold,
        "overall_test_metrics": binary_metrics(test_labels, test_raw, raw_threshold),
        "error_type_counts": {
            key: int(value) for key, value in test_frame["error_type"].value_counts().items()
        },
        "calibration": {
            "raw": raw_calibration,
            "temperature_scaled": calibrated_calibration,
        },
        "uncertainty": uncertainty_report,
        "privacy_coded_age_count": int((test_frame["age"] > 120).sum()),
        "subgroups": subgroup_frame.to_dict(orient="records"),
        "diagnostic_superclasses": superclass_frame.to_dict(orient="records"),
        "lowest_sensitivity_diagnostic_codes_minimum_support_10": code_frame.head(15).to_dict(
            orient="records"
        ),
        "limitations": [
            "Subgroup results are descriptive and were not adjusted for diagnosis prevalence or confounding.",
            "Signal-quality groups use annotation presence, not a validated signal-quality index.",
            "Temperature scaling and the uncertainty margin were selected on validation and remain exploratory.",
            "The binary target covers broad diagnostic abnormalities and is not an arrhythmia taxonomy.",
        ],
    }

    (output_dir / "error_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    test_frame.to_csv(output_dir / "test_error_cases.csv", index=False)
    subgroup_frame.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    superclass_frame.to_csv(output_dir / "diagnostic_superclass_metrics.csv", index=False)
    code_frame.to_csv(output_dir / "diagnostic_code_metrics.csv", index=False)
    calibration_frame.to_csv(output_dir / "calibration_bins.csv", index=False)
    validation_uncertainty.assign(split="validation").to_csv(
        output_dir / "validation_uncertainty_curve.csv", index=False
    )
    test_uncertainty.assign(split="test").to_csv(
        output_dir / "test_uncertainty_curve.csv", index=False
    )

    plot_summary(
        subgroup_frame,
        superclass_frame,
        calibration_frame,
        validation_uncertainty,
        test_uncertainty,
        selected_margin,
        output_dir / "error_analysis.png",
    )
    false_negatives = test_frame.loc[test_frame["error_type"] == "false_negative"]
    plot_false_negative_examples(
        false_negatives,
        dataset_root,
        statements,
        raw_threshold,
        output_dir / "false_negative_examples.pdf",
        output_dir / "false_negative_montage.png",
        example_count=args.false_negative_examples,
    )

    print("False negatives: {}".format(len(false_negatives)))
    print("Temperature: {:.4f}".format(temperature))
    print(
        "Test ECE: {:.4f} raw, {:.4f} temperature-scaled".format(
            raw_calibration["expected_calibration_error_10_bins"],
            calibrated_calibration["expected_calibration_error_10_bins"],
        )
    )
    print(
        "Exploratory uncertainty coverage: {:.2%}; retained balanced accuracy: {:.4f}".format(
            uncertainty_report["test_coverage"],
            uncertainty_report["test_metrics_on_retained"]["balanced_accuracy"],
        )
    )
    print("Outputs written to {}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
