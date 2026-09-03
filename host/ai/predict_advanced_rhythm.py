"""Run the advanced bipolar rhythm model with research quality and uncertainty gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from advanced_rhythm_components import (
    AdvancedRhythmNet,
    assess_signal_quality,
    extract_rhythm_features,
)
from advanced_rhythm_data import load_wfdb_selected_record
from analyze_anomaly_errors import apply_temperature


DEFAULT_MODEL = Path("artifacts/advanced_bipolar_rhythm_classifier/model.pt")


def _load_numpy_record(
    path: Path,
    expected_leads: Sequence[str],
    expected_sample_count: int,
) -> np.ndarray:
    waveform = np.load(path, allow_pickle=False).astype(np.float32)
    if (
        tuple(expected_leads) == ("I", "II", "III")
        and waveform.shape in ((2, expected_sample_count), (expected_sample_count, 2))
    ):
        if waveform.shape == (expected_sample_count, 2):
            waveform = waveform.T
        waveform = np.vstack((waveform, waveform[1] - waveform[0])).astype(np.float32)
    if waveform.shape == (expected_sample_count, len(expected_leads)):
        waveform = waveform.T
    expected_shape = (len(expected_leads), expected_sample_count)
    if waveform.shape != expected_shape:
        raise ValueError(
            "Expected NumPy waveform shape {} or its transpose, received {}".format(
                expected_shape, waveform.shape
            )
        )
    return waveform


def predict(
    record: Path,
    model_path: Path,
    uncertainty_margin: float = 0.05,
) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != "AdvancedRhythmNet":
        raise ValueError("Unsupported advanced-model checkpoint: {}".format(model_path))
    class_names = tuple(checkpoint["class_names"])
    display_names = tuple(checkpoint["display_names"])
    lead_names = tuple(checkpoint["lead_order"])
    sampling_frequency_hz = int(checkpoint["sampling_frequency_hz"])
    sample_count = int(checkpoint["sample_count"])
    if record.suffix.lower() == ".npy":
        raw_shape = np.load(record, mmap_mode="r", allow_pickle=False).shape
        lead_iii_reconstructed = (
            lead_names == ("I", "II", "III")
            and raw_shape in ((2, sample_count), (sample_count, 2))
        )
        waveform = _load_numpy_record(record, lead_names, sample_count)
        input_kind = "selected-lead NumPy array in mV"
    else:
        waveform = load_wfdb_selected_record(record, lead_names, sampling_frequency_hz)
        input_kind = "compatible 12-lead WFDB source with selected bipolar leads"
        lead_iii_reconstructed = False
    quality = assess_signal_quality(waveform, sampling_frequency_hz, lead_names)
    rhythm_features = extract_rhythm_features(waveform, sampling_frequency_hz, lead_names)
    waveform_means = np.asarray(checkpoint["waveform_means_mv"], dtype=np.float32)[:, None]
    waveform_deviations = np.asarray(
        checkpoint["waveform_standard_deviations_mv"], dtype=np.float32
    )[:, None]
    feature_means = np.asarray(checkpoint["feature_means"], dtype=np.float32)
    feature_deviations = np.asarray(checkpoint["feature_standard_deviations"], dtype=np.float32)
    normalized_waveform = (waveform - waveform_means) / waveform_deviations
    normalized_features = (rhythm_features - feature_means) / feature_deviations
    model = AdvancedRhythmNet(len(lead_names), len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(normalized_waveform[None]),
            torch.from_numpy(normalized_features[None]),
        )[0]
        raw_probabilities = torch.sigmoid(logits).numpy()
    temperatures = np.asarray(checkpoint["temperatures"], dtype=np.float64)
    probabilities = np.asarray(
        [
            apply_temperature(raw_probabilities[index : index + 1], temperatures[index])[0]
            for index in range(len(class_names))
        ],
        dtype=np.float64,
    )
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float64)
    margins = np.abs(probabilities - thresholds)
    predictions = []
    for index, (class_name, display_name) in enumerate(zip(class_names, display_names)):
        predictions.append(
            {
                "class": class_name,
                "display_name": display_name,
                "raw_probability": float(raw_probabilities[index]),
                "calibrated_probability": float(probabilities[index]),
                "decision_threshold": float(thresholds[index]),
                "threshold_margin": float(margins[index]),
                "positive": bool(probabilities[index] >= thresholds[index]),
            }
        )
    uncertainty_reasons = []
    if not quality.acceptable:
        uncertainty_reasons.append("experimental_signal_quality_gate_failed")
    if np.any(margins < uncertainty_margin):
        uncertainty_reasons.append("prediction_close_to_decision_threshold")
    status = "inconclusive" if uncertainty_reasons else "result_available"
    if "III" not in lead_names:
        lead_iii_source = "not_applicable"
    elif lead_iii_reconstructed:
        lead_iii_source = "reconstructed_as_ii_minus_i"
    else:
        lead_iii_source = "input"
    return {
        "record": str(record.resolve()),
        "input_kind": input_kind,
        "input_leads": list(lead_names),
        "lead_iii_source": lead_iii_source,
        "sampling_frequency_hz": sampling_frequency_hz,
        "status": status,
        "uncertainty_reasons": uncertainty_reasons,
        "positive_labels": (
            [] if status == "inconclusive" else [item["class"] for item in predictions if item["positive"]]
        ),
        "predictions": predictions,
        "signal_quality": quality.to_dict(),
        "deterministic_rhythm_features": dict(
            zip(checkpoint["feature_names"], rhythm_features.astype(float).tolist())
        ),
        "model": "Advanced multi-dataset bipolar limb-lead rhythm classifier",
        "intended_use": "Research only; not a diagnosis or clinical-use output.",
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--uncertainty-margin", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not 0.0 <= args.uncertainty_margin <= 0.5:
        raise ValueError("Uncertainty margin must be between 0 and 0.5")
    result = predict(args.record, args.model, args.uncertainty_margin)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
