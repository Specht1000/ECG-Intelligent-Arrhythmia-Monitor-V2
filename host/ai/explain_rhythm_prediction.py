"""Create an experimental integrated-gradients view for one rhythm prediction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from predict_rhythm import DEFAULT_MODEL, format_predictions, normalize_record_base
from train_anomaly_baseline import load_low_resolution_record
from train_rhythm_classifier import RhythmECGNet


DEFAULT_OUTPUT_DIR = Path("artifacts/rhythm_explanations")


def integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    class_index: int,
    steps: int = 32,
) -> torch.Tensor:
    if inputs.ndim != 3 or inputs.shape[0] != 1:
        raise ValueError("Integrated gradients expects one batched ECG input")
    if steps < 2:
        raise ValueError("Integrated gradients requires at least two integration steps")
    baseline = torch.zeros_like(inputs)
    alphas = torch.linspace(0.0, 1.0, steps + 1, dtype=inputs.dtype).view(-1, 1, 1)
    path_inputs = baseline + alphas * (inputs - baseline)
    path_inputs.requires_grad_(True)
    logits = model(path_inputs)[:, class_index]
    gradients = torch.autograd.grad(logits.sum(), path_inputs)[0]
    average_gradients = (
        0.5 * gradients[0]
        + gradients[1:-1].sum(dim=0)
        + 0.5 * gradients[-1]
    ) / steps
    return (inputs[0] - baseline[0]) * average_gradients


def load_model_and_input(
    record_base: Path, model_path: Path
) -> Tuple[dict, RhythmECGNet, np.ndarray, torch.Tensor, np.ndarray]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != "RhythmECGNet":
        raise ValueError("Unsupported model checkpoint: {}".format(model_path))
    if checkpoint.get("sampling_frequency_hz") != 100 or checkpoint.get("sample_count") != 1000:
        raise ValueError("The checkpoint does not describe a 100 Hz, 10-second input")
    model = RhythmECGNet(class_count=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    waveform = load_low_resolution_record(normalize_record_base(record_base))
    means = np.asarray(checkpoint["normalization_means_mv"], dtype=np.float32)[:, None]
    standard_deviations = np.asarray(
        checkpoint["normalization_standard_deviations_mv"], dtype=np.float32
    )[:, None]
    normalized = torch.from_numpy((waveform - means) / standard_deviations).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(normalized)).squeeze(0).numpy()
    return checkpoint, model, waveform, normalized, probabilities


def select_class_index(
    class_names: Sequence[str], probabilities: np.ndarray, requested_class: Optional[str]
) -> int:
    if requested_class is None:
        return int(np.argmax(probabilities))
    if requested_class not in class_names:
        raise ValueError(
            "Unknown class '{}'; choose one of {}".format(
                requested_class, ", ".join(class_names)
            )
        )
    return class_names.index(requested_class)


def plot_explanation(
    waveform: np.ndarray,
    absolute_attribution: np.ndarray,
    lead_names: Sequence[str],
    display_name: str,
    probability: float,
    threshold: float,
    output_path: Path,
) -> None:
    time_seconds = np.arange(waveform.shape[1]) / 100.0
    scale = float(np.quantile(absolute_attribution, 0.995))
    normalized_attribution = np.clip(
        absolute_attribution / max(scale, np.finfo(np.float32).eps), 0.0, 1.0
    )
    figure, axes = plt.subplots(6, 2, figsize=(16, 14), sharex=True)
    for lead_index, axis in enumerate(axes.flat):
        axis.plot(time_seconds, waveform[lead_index], color="#606060", linewidth=0.7, zorder=1)
        axis.scatter(
            time_seconds,
            waveform[lead_index],
            c=normalized_attribution[lead_index],
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            s=5,
            linewidths=0,
            zorder=2,
        )
        axis.set_ylabel("{} (mV)".format(lead_names[lead_index]))
        axis.grid(alpha=0.15)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    scalar_mappable = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="Reds")
    colorbar = figure.colorbar(scalar_mappable, ax=list(axes.flat), fraction=0.015, pad=0.02)
    colorbar.set_label("Normalized absolute attribution")
    figure.suptitle(
        "Integrated gradients - {} | probability {:.3f} | threshold {:.3f}".format(
            display_name, probability, threshold
        )
    )
    figure.subplots_adjust(top=0.94, right=0.91, hspace=0.28, wspace=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def explain(
    record_base: Path,
    model_path: Path,
    class_name: Optional[str],
    steps: int,
    output_path: Optional[Path],
) -> dict:
    normalized_base = normalize_record_base(record_base)
    checkpoint, model, waveform, normalized, probabilities = load_model_and_input(
        normalized_base, model_path
    )
    class_names = tuple(checkpoint["class_names"])
    display_names = tuple(checkpoint["display_names"])
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float32)
    class_index = select_class_index(class_names, probabilities, class_name)
    attribution = integrated_gradients(model, normalized, class_index, steps=steps)
    absolute_attribution = attribution.detach().abs().numpy()

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / "{}_{}_integrated_gradients.png".format(
            normalized_base.name, class_names[class_index]
        )
    output_path = output_path.resolve()
    plot_explanation(
        waveform,
        absolute_attribution,
        tuple(checkpoint["lead_order"]),
        display_names[class_index],
        float(probabilities[class_index]),
        float(thresholds[class_index]),
        output_path,
    )
    predictions = format_predictions(
        class_names, display_names, probabilities, thresholds
    )
    lead_attribution = absolute_attribution.sum(axis=1)
    lead_attribution = lead_attribution / max(
        float(lead_attribution.sum()), np.finfo(np.float32).eps
    )
    result = {
        "record": str(normalized_base.resolve()),
        "explained_class": class_names[class_index],
        "display_name": display_names[class_index],
        "probability": float(probabilities[class_index]),
        "decision_threshold": float(thresholds[class_index]),
        "positive": bool(probabilities[class_index] >= thresholds[class_index]),
        "integration_steps": steps,
        "relative_absolute_attribution_by_lead": {
            lead: float(value)
            for lead, value in zip(checkpoint["lead_order"], lead_attribution)
        },
        "all_predictions": predictions,
        "plot": str(output_path),
        "interpretation_limit": (
            "Integrated gradients measures model sensitivity relative to the training-mean "
            "baseline; it does not prove medical causality or clinical correctness."
        ),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record",
        type=Path,
        help="PTB-XL records100 base path, .hea path, or .dat path",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--class-name", choices=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = explain(args.record, args.model, args.class_name, args.steps, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
