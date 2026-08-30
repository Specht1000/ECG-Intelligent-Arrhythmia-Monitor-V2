"""Run the experimental anomaly baseline on one PTB-XL 100 Hz record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from train_anomaly_baseline import AnomalyECGNet, load_low_resolution_record


DEFAULT_MODEL = Path("artifacts/anomaly_baseline/model.pt")


def normalize_record_base(path: Path) -> Path:
    if path.suffix.lower() in (".hea", ".dat"):
        return path.with_suffix("")
    return path


def predict(record_base: Path, model_path: Path) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != "AnomalyECGNet":
        raise ValueError("Unsupported model checkpoint: {}".format(model_path))
    if checkpoint.get("sampling_frequency_hz") != 100 or checkpoint.get("sample_count") != 1000:
        raise ValueError("The checkpoint does not describe a 100 Hz, 10-second input")

    model = AnomalyECGNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    waveform = load_low_resolution_record(normalize_record_base(record_base))
    means = np.asarray(checkpoint["normalization_means_mv"], dtype=np.float32)[:, None]
    standard_deviations = np.asarray(
        checkpoint["normalization_standard_deviations_mv"], dtype=np.float32
    )[:, None]
    normalized = (waveform - means) / standard_deviations
    with torch.no_grad():
        logits = model(torch.from_numpy(normalized).unsqueeze(0))
        probability = float(torch.sigmoid(logits).item())

    threshold = float(checkpoint["decision_threshold"])
    return {
        "record": str(normalize_record_base(record_base).resolve()),
        "predicted_label": "abnormal" if probability >= threshold else "normal",
        "abnormal_probability": probability,
        "decision_threshold": threshold,
        "model": "PTB-XL experimental anomaly baseline",
        "intended_use": "Research benchmark only; not a diagnostic or clinical-use output.",
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record",
        type=Path,
        help="PTB-XL records100 base path, .hea path, or .dat path",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = predict(args.record, args.model)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
