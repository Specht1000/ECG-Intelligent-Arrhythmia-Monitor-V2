"""Explain one advanced bipolar-rhythm prediction with integrated gradients."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from torch import nn

from advanced_rhythm_components import AdvancedRhythmNet, extract_rhythm_features
from advanced_rhythm_data import load_wfdb_selected_record
from analyze_anomaly_errors import apply_temperature
from explain_rhythm_prediction import integrated_gradients, select_class_index
from predict_advanced_rhythm import DEFAULT_MODEL, _load_numpy_record


DEFAULT_OUTPUT_DIR = Path("artifacts/advanced_bipolar_rhythm_explanations")


class WaveformExplanationWrapper(nn.Module):
    """Bind deterministic features so attribution is computed over the waveform only."""

    def __init__(self, model: AdvancedRhythmNet, features: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("features", features)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.features.expand(waveform.shape[0], -1)
        return self.model(waveform, features)


def _load_waveform(record: Path, checkpoint: dict) -> np.ndarray:
    lead_names = tuple(checkpoint["lead_order"])
    sample_count = int(checkpoint["sample_count"])
    if record.suffix.lower() == ".npy":
        return _load_numpy_record(record, lead_names, sample_count)
    return load_wfdb_selected_record(
        record,
        lead_names,
        int(checkpoint["sampling_frequency_hz"]),
    )


def _calibrated_probabilities(logits: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    raw_probabilities = 1.0 / (1.0 + np.exp(-logits))
    return np.asarray(
        [
            apply_temperature(raw_probabilities[index : index + 1], temperatures[index])[0]
            for index in range(len(raw_probabilities))
        ],
        dtype=np.float64,
    )


def _plot_explanation(
    waveform: np.ndarray,
    absolute_attribution: np.ndarray,
    lead_names: Sequence[str],
    sampling_frequency_hz: int,
    title: str,
    output_path: Path,
) -> None:
    time_seconds = np.arange(waveform.shape[1]) / float(sampling_frequency_hz)
    scale = max(float(np.quantile(absolute_attribution, 0.995)), np.finfo(np.float32).eps)
    normalized_attribution = np.clip(absolute_attribution / scale, 0.0, 1.0)
    figure, axes = plt.subplots(
        len(lead_names),
        1,
        figsize=(16, max(5, 2.7 * len(lead_names))),
        sharex=True,
        squeeze=False,
    )
    visible_axes = []
    for lead_index, lead_name in enumerate(lead_names):
        axis = axes[lead_index, 0]
        visible_axes.append(axis)
        axis.plot(time_seconds, waveform[lead_index], color="#606060", linewidth=0.7)
        axis.scatter(
            time_seconds,
            waveform[lead_index],
            c=normalized_attribution[lead_index],
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            s=5,
            linewidths=0,
        )
        axis.set_ylabel("{} (mV)".format(lead_name))
        axis.grid(alpha=0.15)
    visible_axes[-1].set_xlabel("Time (s)")
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="Reds"),
        ax=visible_axes,
        fraction=0.015,
        pad=0.02,
    )
    colorbar.set_label("Normalized absolute attribution")
    figure.suptitle(title)
    figure.subplots_adjust(top=0.93, right=0.91, hspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def explain(
    record: Path,
    model_path: Path,
    class_name: Optional[str],
    steps: int,
    output_path: Optional[Path],
) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != "AdvancedRhythmNet":
        raise ValueError("Unsupported advanced-model checkpoint: {}".format(model_path))
    class_names = tuple(checkpoint["class_names"])
    display_names = tuple(checkpoint["display_names"])
    lead_names = tuple(checkpoint["lead_order"])
    sampling_frequency_hz = int(checkpoint["sampling_frequency_hz"])
    waveform = _load_waveform(record, checkpoint)
    features = extract_rhythm_features(waveform, sampling_frequency_hz, lead_names)
    waveform_means = np.asarray(checkpoint["waveform_means_mv"], dtype=np.float32)[:, None]
    waveform_deviations = np.asarray(
        checkpoint["waveform_standard_deviations_mv"], dtype=np.float32
    )[:, None]
    feature_means = np.asarray(checkpoint["feature_means"], dtype=np.float32)
    feature_deviations = np.asarray(
        checkpoint["feature_standard_deviations"], dtype=np.float32
    )
    normalized_waveform = torch.from_numpy(
        ((waveform - waveform_means) / waveform_deviations)[None]
    )
    normalized_features = torch.from_numpy(
        ((features - feature_means) / feature_deviations)[None]
    )
    model = AdvancedRhythmNet(len(lead_names), len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    wrapper = WaveformExplanationWrapper(model, normalized_features)
    wrapper.eval()
    with torch.no_grad():
        logits = wrapper(normalized_waveform)[0].numpy()
    probabilities = _calibrated_probabilities(
        logits, np.asarray(checkpoint["temperatures"], dtype=np.float64)
    )
    class_index = select_class_index(class_names, probabilities, class_name)
    attribution = integrated_gradients(wrapper, normalized_waveform, class_index, steps)
    absolute_attribution = attribution.detach().abs().numpy()
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float64)
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / "{}_{}_integrated_gradients.png".format(
            record.stem, class_names[class_index]
        )
    output_path = output_path.resolve()
    _plot_explanation(
        waveform,
        absolute_attribution,
        lead_names,
        sampling_frequency_hz,
        "Integrated gradients - {} | calibrated probability {:.3f} | threshold {:.3f}".format(
            display_names[class_index], probabilities[class_index], thresholds[class_index]
        ),
        output_path,
    )
    lead_attribution = absolute_attribution.sum(axis=1)
    lead_attribution /= max(float(lead_attribution.sum()), np.finfo(np.float32).eps)
    result = {
        "record": str(record.resolve()),
        "explained_class": class_names[class_index],
        "display_name": display_names[class_index],
        "calibrated_probability": float(probabilities[class_index]),
        "decision_threshold": float(thresholds[class_index]),
        "positive": bool(probabilities[class_index] >= thresholds[class_index]),
        "integration_steps": steps,
        "relative_absolute_attribution_by_lead": {
            lead: float(value) for lead, value in zip(lead_names, lead_attribution)
        },
        "plot": str(output_path),
        "interpretation_limit": (
            "Integrated gradients measures model sensitivity relative to the normalized-zero "
            "waveform baseline while deterministic features are fixed. It does not prove "
            "medical causality or clinical correctness."
        ),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--class-name")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.steps < 2:
        raise ValueError("Integrated gradients requires at least two integration steps")
    result = explain(args.record, args.model, args.class_name, args.steps, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
