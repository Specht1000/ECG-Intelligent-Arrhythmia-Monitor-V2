"""Train the approved six-label rhythm benchmark on Chapman and test on PTB-XL."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_anomaly_baseline import ResidualBlock, load_low_resolution_record, parse_scp_codes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CLASS_NAMES = (
    "sinus_rhythm",
    "sinus_bradycardia",
    "sinus_tachycardia",
    "sinus_arrhythmia",
    "atrial_fibrillation",
    "atrial_flutter",
)
DISPLAY_NAMES = (
    "Sinus rhythm",
    "Sinus bradycardia",
    "Sinus tachycardia",
    "Sinus arrhythmia",
    "Atrial fibrillation",
    "Atrial flutter",
)
CHAPMAN_CODE_TO_CLASS = {
    "426783006": "sinus_rhythm",
    "426177001": "sinus_bradycardia",
    "427084000": "sinus_tachycardia",
    "427393009": "sinus_arrhythmia",
    "164889003": "atrial_fibrillation",
    "164890007": "atrial_flutter",
}
PTBXL_CODE_TO_CLASS = {
    "SR": "sinus_rhythm",
    "SBRAD": "sinus_bradycardia",
    "STACH": "sinus_tachycardia",
    "SARRH": "sinus_arrhythmia",
    "AFIB": "atrial_fibrillation",
    "AFLT": "atrial_flutter",
}
EXCLUDED_CHAPMAN_RECORDS = frozenset(("JS01052", "JS23074"))
GAIN_PATTERN = re.compile(
    r"^(?P<gain>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\((?P<baseline>[+-]?\d+)\))?/(?P<unit>\S+)$"
)


@dataclass(frozen=True)
class RhythmTrainingConfig:
    seed: int = 20260831
    epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    waveform_workers: int = 12
    data_loader_workers: int = 0
    sampling_frequency_hz: int = 100
    sample_count: int = 1000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def label_vector(labels: Iterable[str]) -> np.ndarray:
    selected = set(labels)
    return np.asarray([class_name in selected for class_name in CLASS_NAMES], dtype=np.float32)


def _parse_chapman_header(path: Path, dataset_root: Path) -> Optional[Dict[str, object]]:
    if path.stem in EXCLUDED_CHAPMAN_RECORDS:
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    first = lines[0].split()
    if len(first) < 4 or (int(first[1]), float(first[2]), int(first[3])) != (12, 500.0, 5000):
        raise ValueError("Unexpected Chapman record structure: {}".format(path))

    gains: List[float] = []
    baselines: List[float] = []
    leads: List[str] = []
    for line in lines[1:13]:
        fields = line.split()
        if len(fields) < 9:
            raise ValueError("Incomplete Chapman signal definition: {}".format(path))
        match = GAIN_PATTERN.match(fields[2])
        if match is None or match.group("unit") != "mV":
            raise ValueError("Unsupported Chapman gain or unit: {}".format(path))
        gains.append(float(match.group("gain")))
        baselines.append(
            float(match.group("baseline")) if match.group("baseline") is not None else float(fields[4])
        )
        leads.append(fields[-1])
    if tuple(leads) != LEADS:
        raise ValueError("Unexpected Chapman lead order: {}".format(path))

    diagnosis_codes: List[str] = []
    for line in lines[13:]:
        if line.startswith("#Dx:"):
            diagnosis_codes = [code.strip() for code in line[4:].split(",") if code.strip()]
            break
    labels = sorted({CHAPMAN_CODE_TO_CLASS[code] for code in diagnosis_codes if code in CHAPMAN_CODE_TO_CLASS})
    return {
        "record_id": path.stem,
        "record_base": str(path.relative_to(dataset_root).with_suffix("")),
        "diagnosis_codes": "|".join(diagnosis_codes),
        "labels": "|".join(labels),
        "gains": json.dumps(gains, separators=(",", ":")),
        "baselines": json.dumps(baselines, separators=(",", ":")),
        **{class_name: int(class_name in labels) for class_name in CLASS_NAMES},
    }


def build_or_load_chapman_index(
    dataset_root: Path, index_path: Path, workers: int
) -> pd.DataFrame:
    if index_path.is_file():
        frame = pd.read_csv(index_path, keep_default_na=False)
        required = {"record_id", "record_base", "labels", "gains", "baselines", *CLASS_NAMES}
        if required.issubset(frame.columns) and not frame.empty:
            return frame

    header_paths = sorted((dataset_root / "WFDBRecords").rglob("*.hea"))
    index_path.parent.mkdir(parents=True, exist_ok=True)

    def parse(path: Path) -> Optional[Dict[str, object]]:
        return _parse_chapman_header(path, dataset_root)

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for result in tqdm(
            executor.map(parse, header_paths), total=len(header_paths), desc="Indexing Chapman headers"
        ):
            if result is not None:
                rows.append(result)
    frame = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
    frame.to_csv(index_path, index=False)
    return frame


def make_chapman_splits(metadata: pd.DataFrame, seed: int) -> Dict[str, np.ndarray]:
    indices = np.arange(len(metadata))
    combinations = metadata["labels"].astype(str)
    counts = combinations.value_counts()
    stratification = combinations.where(combinations.map(counts) >= 10, "rare_combination")
    training, temporary = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=stratification,
    )
    temporary_stratification = stratification.iloc[temporary]
    temporary_counts = temporary_stratification.value_counts()
    temporary_stratification = temporary_stratification.where(
        temporary_stratification.map(temporary_counts) >= 2, "rare_combination"
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=temporary_stratification,
    )
    return {
        "training": np.sort(training),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }


def cache_fingerprint(metadata: pd.DataFrame) -> str:
    payload = "\n".join(
        "{}|{}|{}".format(row.record_id, row.record_base, row.labels)
        for row in metadata.itertuples()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_chapman_waveform(dataset_root: Path, row: Mapping[str, object]) -> np.ndarray:
    matrix = loadmat(
        (dataset_root / str(row["record_base"])).with_suffix(".mat"),
        variable_names=("val",),
    )["val"].astype(np.float32)
    if matrix.shape != (12, 5000):
        raise ValueError("Unexpected Chapman waveform shape for {}".format(row["record_id"]))
    gains = np.asarray(json.loads(str(row["gains"])), dtype=np.float32)[:, None]
    baselines = np.asarray(json.loads(str(row["baselines"])), dtype=np.float32)[:, None]
    physical_mv = (matrix - baselines) / gains
    downsampled = resample_poly(physical_mv, up=1, down=5, axis=1)
    if downsampled.shape != (12, 1000):
        raise ValueError("Unexpected downsampled shape for {}".format(row["record_id"]))
    return downsampled.astype(np.float32)


def build_or_load_chapman_cache(
    dataset_root: Path,
    metadata: pd.DataFrame,
    cache_path: Path,
    workers: int,
) -> np.memmap:
    manifest_path = cache_path.with_suffix(".json")
    expected_manifest = {
        "format_version": 1,
        "metadata_fingerprint": cache_fingerprint(metadata),
        "shape": [len(metadata), 12, 1000],
        "dtype": "float16",
        "unit": "mV",
        "lead_order": list(LEADS),
        "source_sampling_frequency_hz": 500,
        "sampling_frequency_hz": 100,
        "resampling": "scipy.signal.resample_poly up=1 down=5",
    }
    valid = False
    if cache_path.is_file() and manifest_path.is_file():
        try:
            valid = json.loads(manifest_path.read_text(encoding="utf-8")) == expected_manifest
        except (OSError, ValueError):
            valid = False
    if not valid:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".building.npy")
        cache = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=(len(metadata), 12, 1000)
        )
        rows = metadata.to_dict(orient="records")
        chunk_size = 256
        progress = tqdm(total=len(rows), desc="Building Chapman waveform cache")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for start in range(0, len(rows), chunk_size):
                chunk_rows = rows[start : start + chunk_size]
                waveforms = list(
                    executor.map(lambda row: _load_chapman_waveform(dataset_root, row), chunk_rows)
                )
                cache[start : start + len(waveforms)] = np.stack(waveforms).astype(np.float16)
                progress.update(len(waveforms))
        progress.close()
        cache.flush()
        del cache
        temporary.replace(cache_path)
        manifest_path.write_text(json.dumps(expected_manifest, indent=2) + "\n", encoding="utf-8")
    return np.load(cache_path, mmap_mode="r")


def build_or_load_ptbxl_external_cache(
    dataset_root: Path, metadata: pd.DataFrame, cache_path: Path, workers: int
) -> np.memmap:
    fingerprint_payload = "\n".join(
        "{}|{}".format(row.ecg_id, row.filename_lr) for row in metadata.itertuples()
    )
    expected_manifest = {
        "format_version": 1,
        "fingerprint": hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest(),
        "shape": [len(metadata), 12, 1000],
        "dtype": "float16",
        "unit": "mV",
    }
    manifest_path = cache_path.with_suffix(".json")
    valid = False
    if cache_path.is_file() and manifest_path.is_file():
        try:
            valid = json.loads(manifest_path.read_text(encoding="utf-8")) == expected_manifest
        except (OSError, ValueError):
            valid = False
    if not valid:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".building.npy")
        cache = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=(len(metadata), 12, 1000)
        )
        record_paths = [dataset_root / filename for filename in metadata["filename_lr"]]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for index, waveform in enumerate(
                tqdm(
                    executor.map(load_low_resolution_record, record_paths),
                    total=len(record_paths),
                    desc="Building PTB-XL external cache",
                )
            ):
                cache[index] = waveform.astype(np.float16)
        cache.flush()
        del cache
        temporary.replace(cache_path)
        manifest_path.write_text(json.dumps(expected_manifest, indent=2) + "\n", encoding="utf-8")
    return np.load(cache_path, mmap_mode="r")


def load_ptbxl_external_metadata(dataset_root: Path) -> pd.DataFrame:
    metadata = pd.read_csv(dataset_root / "ptbxl_database.csv")
    metadata = metadata.loc[metadata["strat_fold"] == 10].copy().reset_index(drop=True)
    labels = []
    for value in metadata["scp_codes"]:
        codes = parse_scp_codes(value)
        selected = {PTBXL_CODE_TO_CLASS[code] for code in codes if code in PTBXL_CODE_TO_CLASS}
        labels.append(label_vector(selected))
    vectors = np.stack(labels)
    for index, class_name in enumerate(CLASS_NAMES):
        metadata[class_name] = vectors[:, index].astype(np.int64)
    metadata["labels"] = [
        "|".join(class_name for class_name, selected in zip(CLASS_NAMES, vector) if selected)
        for vector in vectors
    ]
    return metadata


def compute_normalization(
    waveforms: np.memmap, training_indices: Sequence[int], chunk_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(12, dtype=np.float64)
    squared_sums = np.zeros(12, dtype=np.float64)
    count = 0
    indices = np.asarray(training_indices, dtype=np.int64)
    for start in tqdm(range(0, len(indices), chunk_size), desc="Computing normalization"):
        batch = np.asarray(waveforms[indices[start : start + chunk_size]], dtype=np.float32)
        sums += batch.sum(axis=(0, 2), dtype=np.float64)
        squared_sums += np.square(batch, dtype=np.float32).sum(axis=(0, 2), dtype=np.float64)
        count += batch.shape[0] * batch.shape[2]
    means = sums / count
    variance = np.maximum(squared_sums / count - np.square(means), 1e-12)
    return means.astype(np.float32), np.sqrt(variance).astype(np.float32)


class RhythmDataset(Dataset):
    def __init__(
        self,
        waveforms: np.memmap,
        labels: np.ndarray,
        identifiers: Sequence[object],
        indices: Sequence[int],
        means: np.ndarray,
        standard_deviations: np.ndarray,
    ) -> None:
        self.waveforms = waveforms
        self.labels = labels.astype(np.float32)
        self.identifiers = np.asarray(identifiers)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.means = means[:, None]
        self.standard_deviations = standard_deviations[:, None]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item])
        waveform = np.asarray(self.waveforms[index], dtype=np.float32)
        waveform = (waveform - self.means) / self.standard_deviations
        return (
            torch.from_numpy(waveform),
            torch.from_numpy(self.labels[index]),
            int(index),
        )


class RhythmECGNet(nn.Module):
    def __init__(self, class_count: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 24, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            ResidualBlock(24, 24),
            ResidualBlock(24, 48, stride=2),
            ResidualBlock(48, 96, stride=2),
            ResidualBlock(96, 128, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, class_count)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best = np.flatnonzero(f1_values == np.nanmax(f1_values))
    return float(thresholds[best[np.argmin(np.abs(thresholds[best] - 0.5))]])


def select_thresholds(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(
        [select_f1_threshold(labels[:, index], probabilities[:, index]) for index in range(len(CLASS_NAMES))],
        dtype=np.float64,
    )


def multilabel_metrics(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> Dict[str, object]:
    predictions = probabilities >= thresholds[None, :]
    per_class = []
    for index, (class_name, display_name) in enumerate(zip(CLASS_NAMES, DISPLAY_NAMES)):
        reference = labels[:, index].astype(np.int64)
        predicted = predictions[:, index].astype(np.int64)
        positive_count = int(reference.sum())
        negative_count = int(len(reference) - positive_count)
        true_positive = int(((reference == 1) & (predicted == 1)).sum())
        true_negative = int(((reference == 0) & (predicted == 0)).sum())
        metrics = {
            "class_name": class_name,
            "display_name": display_name,
            "threshold": float(thresholds[index]),
            "positive_count": positive_count,
            "prevalence": float(reference.mean()),
            "roc_auc": (
                float(roc_auc_score(reference, probabilities[:, index]))
                if positive_count and negative_count
                else math.nan
            ),
            "average_precision": (
                float(average_precision_score(reference, probabilities[:, index]))
                if positive_count
                else math.nan
            ),
            "f1": float(f1_score(reference, predicted, zero_division=0)),
            "precision": float(precision_score(reference, predicted, zero_division=0)),
            "sensitivity": float(recall_score(reference, predicted, zero_division=0)),
            "specificity": float(true_negative / negative_count) if negative_count else math.nan,
            "true_positive": true_positive,
            "false_positive": int(((reference == 0) & (predicted == 1)).sum()),
            "false_negative": int(((reference == 1) & (predicted == 0)).sum()),
            "true_negative": true_negative,
        }
        per_class.append(metrics)
    valid_auc = [entry["roc_auc"] for entry in per_class if not math.isnan(entry["roc_auc"])]
    valid_ap = [entry["average_precision"] for entry in per_class if not math.isnan(entry["average_precision"])]
    return {
        "macro_roc_auc": float(np.mean(valid_auc)),
        "macro_average_precision": float(np.mean(valid_ap)),
        "macro_f1": float(np.mean([entry["f1"] for entry in per_class])),
        "micro_f1": float(f1_score(labels.ravel(), predictions.ravel(), zero_division=0)),
        "exact_match_accuracy": float(np.mean(np.all(labels == predictions, axis=1))),
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_function: nn.Module, device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses = 0.0
    count = 0
    labels_list = []
    probabilities_list = []
    indices_list = []
    for waveforms, labels, indices in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        logits = model(waveforms)
        losses += float(loss_function(logits, labels).item()) * len(labels)
        count += len(labels)
        labels_list.append(labels.cpu().numpy())
        probabilities_list.append(torch.sigmoid(logits).cpu().numpy())
        indices_list.append(indices.numpy())
    return (
        losses / count,
        np.concatenate(labels_list),
        np.concatenate(probabilities_list),
        np.concatenate(indices_list),
    )


def prediction_frame(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    identifier_column: str,
) -> pd.DataFrame:
    result = pd.DataFrame({identifier_column: metadata.iloc[indices][identifier_column].to_numpy()})
    for class_index, class_name in enumerate(CLASS_NAMES):
        result["{}_reference".format(class_name)] = labels[:, class_index].astype(np.int64)
        result["{}_probability".format(class_name)] = probabilities[:, class_index]
        result["{}_prediction".format(class_name)] = (
            probabilities[:, class_index] >= thresholds[class_index]
        ).astype(np.int64)
    return result


def plot_evaluation(
    history: pd.DataFrame,
    chapman_metrics: Mapping[str, object],
    external_metrics: Mapping[str, object],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(history["epoch"], history["training_loss"], label="Training")
    axes[0, 0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0, 0].set(title="Loss", xlabel="Epoch", ylabel="Weighted binary cross-entropy")
    axes[0, 0].legend()

    labels = list(DISPLAY_NAMES)
    positions = np.arange(len(labels))
    width = 0.38
    chapman_classes = chapman_metrics["per_class"]
    external_classes = external_metrics["per_class"]
    axes[0, 1].bar(
        positions - width / 2,
        [entry["average_precision"] for entry in chapman_classes],
        width,
        label="Chapman held-out",
    )
    axes[0, 1].bar(
        positions + width / 2,
        [entry["average_precision"] for entry in external_classes],
        width,
        label="PTB-XL external",
    )
    axes[0, 1].set(title="Average precision", ylim=(0, 1), xticks=positions, xticklabels=labels)
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[0, 1].legend()

    axes[1, 0].bar(
        positions - width / 2,
        [entry["sensitivity"] for entry in chapman_classes],
        width,
        label="Chapman held-out",
    )
    axes[1, 0].bar(
        positions + width / 2,
        [entry["sensitivity"] for entry in external_classes],
        width,
        label="PTB-XL external",
    )
    axes[1, 0].set(title="Sensitivity", ylim=(0, 1), xticks=positions, xticklabels=labels)
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].legend()

    support = [entry["positive_count"] for entry in external_classes]
    axes[1, 1].bar(positions, support, color="#70AD47")
    axes[1, 1].set(title="PTB-XL external positive support", xticks=positions, xticklabels=labels)
    axes[1, 1].tick_params(axis="x", rotation=35)
    for index, count in enumerate(support):
        axes[1, 1].text(index, count, str(count), ha="center", va="bottom")

    figure.suptitle("Six-label rhythm benchmark")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapman-root", type=Path, default=DEFAULT_CHAPMAN_ROOT)
    parser.add_argument("--ptbxl-root", type=Path, default=DEFAULT_PTBXL_ROOT)
    parser.add_argument(
        "--index", type=Path, default=Path(".cache/chapman_rhythm_index_all_records.csv")
    )
    parser.add_argument(
        "--chapman-cache",
        type=Path,
        default=Path(".cache/chapman_rhythm_waveforms_all_records.npy"),
    )
    parser.add_argument("--ptbxl-cache", type=Path, default=Path(".cache/ptbxl_rhythm_external.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rhythm_classifier"))
    parser.add_argument("--epochs", type=int, default=RhythmTrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=RhythmTrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=RhythmTrainingConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=RhythmTrainingConfig.seed)
    parser.add_argument("--waveform-workers", type=int, default=RhythmTrainingConfig.waveform_workers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = RhythmTrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        waveform_workers=args.waveform_workers,
    )
    set_seed(config.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chapman_root = args.chapman_root.resolve()
    ptbxl_root = args.ptbxl_root.resolve()

    chapman_metadata = build_or_load_chapman_index(
        chapman_root, args.index.resolve(), config.waveform_workers
    )
    splits = make_chapman_splits(chapman_metadata, config.seed)
    chapman_waveforms = build_or_load_chapman_cache(
        chapman_root,
        chapman_metadata,
        args.chapman_cache.resolve(),
        config.waveform_workers,
    )
    ptbxl_metadata = load_ptbxl_external_metadata(ptbxl_root)
    ptbxl_waveforms = build_or_load_ptbxl_external_cache(
        ptbxl_root,
        ptbxl_metadata,
        args.ptbxl_cache.resolve(),
        config.waveform_workers,
    )

    means, standard_deviations = compute_normalization(
        chapman_waveforms, splits["training"]
    )
    chapman_labels = chapman_metadata.loc[:, CLASS_NAMES].to_numpy(dtype=np.float32)
    ptbxl_labels = ptbxl_metadata.loc[:, CLASS_NAMES].to_numpy(dtype=np.float32)
    datasets = {
        split: RhythmDataset(
            chapman_waveforms,
            chapman_labels,
            chapman_metadata["record_id"],
            indices,
            means,
            standard_deviations,
        )
        for split, indices in splits.items()
    }
    external_indices = np.arange(len(ptbxl_metadata))
    datasets["external_ptbxl"] = RhythmDataset(
        ptbxl_waveforms,
        ptbxl_labels,
        ptbxl_metadata["ecg_id"],
        external_indices,
        means,
        standard_deviations,
    )
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "training"),
            num_workers=config.data_loader_workers,
        )
        for split, dataset in datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RhythmECGNet().to(device)
    training_labels = chapman_labels[splits["training"]]
    positive_counts = training_labels.sum(axis=0)
    negative_counts = len(training_labels) - positive_counts
    positive_weights = np.minimum(negative_counts / np.maximum(positive_counts, 1), 25.0)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    print("Device: {}".format(device))
    print("Chapman eligible records: {:,}".format(len(chapman_metadata)))
    print("Split sizes: {}".format({key: len(value) for key, value in splits.items()}))
    print("PTB-XL external records: {:,}".format(len(ptbxl_metadata)))
    print("Trainable parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    history: List[Dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_map = -math.inf
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(loaders["training"], desc="Epoch {}/{}".format(epoch, config.epochs))
        for waveforms, labels, _ in progress:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            sample_count += len(labels)
            progress.set_postfix(loss="{:.4f}".format(loss_sum / sample_count))
        training_loss = loss_sum / sample_count
        validation_loss, validation_labels, validation_probabilities, _ = evaluate(
            model, loaders["validation"], loss_function, device
        )
        validation_map = float(
            np.mean(
                [
                    average_precision_score(validation_labels[:, index], validation_probabilities[:, index])
                    for index in range(len(CLASS_NAMES))
                ]
            )
        )
        scheduler.step(validation_map)
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "validation_macro_average_precision": validation_map,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            "Epoch {:02d}: train loss {:.4f}, validation loss {:.4f}, validation macro AP {:.4f}".format(
                epoch, training_loss, validation_loss, validation_map
            )
        )
        if validation_map > best_validation_map + 1e-4:
            best_validation_map = validation_map
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print("Early stopping after epoch {}".format(epoch))
                break

    model.load_state_dict(best_state)
    validation_loss, validation_labels, validation_probabilities, validation_indices = evaluate(
        model, loaders["validation"], loss_function, device
    )
    thresholds = select_thresholds(validation_labels, validation_probabilities)
    test_loss, test_labels, test_probabilities, test_indices = evaluate(
        model, loaders["test"], loss_function, device
    )
    external_loss, external_labels, external_probabilities, external_indices_output = evaluate(
        model, loaders["external_ptbxl"], loss_function, device
    )
    validation_metrics = multilabel_metrics(validation_labels, validation_probabilities, thresholds)
    test_metrics = multilabel_metrics(test_labels, test_probabilities, thresholds)
    external_metrics = multilabel_metrics(external_labels, external_probabilities, thresholds)
    validation_metrics["loss"] = validation_loss
    test_metrics["loss"] = test_loss
    external_metrics["loss"] = external_loss

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    prediction_frame(
        chapman_metadata,
        validation_indices,
        validation_labels,
        validation_probabilities,
        thresholds,
        "record_id",
    ).to_csv(output_dir / "chapman_validation_predictions.csv", index=False)
    prediction_frame(
        chapman_metadata,
        test_indices,
        test_labels,
        test_probabilities,
        thresholds,
        "record_id",
    ).to_csv(output_dir / "chapman_test_predictions.csv", index=False)
    prediction_frame(
        ptbxl_metadata,
        external_indices_output,
        external_labels,
        external_probabilities,
        thresholds,
        "ecg_id",
    ).to_csv(output_dir / "ptbxl_external_predictions.csv", index=False)

    split_manifest = chapman_metadata[["record_id", "record_base", "labels", *CLASS_NAMES]].copy()
    split_manifest["split"] = ""
    for split, indices in splits.items():
        split_manifest.loc[indices, "split"] = split
    split_manifest.to_csv(output_dir / "chapman_split_manifest.csv", index=False)

    report = {
        "experiment": "Six-label complete-exam rhythm benchmark",
        "intended_use": "Research benchmark only; not a diagnostic or clinical-use model.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Multi-label complete-exam rhythm classification",
        "class_names": list(CLASS_NAMES),
        "display_names": list(DISPLAY_NAMES),
        "label_mapping": {
            "chapman_snomed_ct": CHAPMAN_CODE_TO_CLASS,
            "ptbxl_scp_ecg": PTBXL_CODE_TO_CLASS,
        },
        "data_policy": {
            "primary_training_dataset": "Chapman-Shaoxing-Ningbo",
            "chapman_split": "80% train, 10% validation, 10% held-out test; stratified by label combination.",
            "external_dataset": "PTB-XL 1.0.3 official fold 10 only; never used for training or threshold selection.",
            "input": "12 leads, 10 seconds, resampled to 100 Hz",
        },
        "record_counts": {
            "chapman_eligible": int(len(chapman_metadata)),
            **{"chapman_{}".format(key): int(len(value)) for key, value in splits.items()},
            "ptbxl_external": int(len(ptbxl_metadata)),
        },
        "class_counts": {
            "chapman_all": {
                class_name: int(chapman_metadata[class_name].sum()) for class_name in CLASS_NAMES
            },
            "chapman_training": {
                class_name: int(chapman_metadata.iloc[splits["training"]][class_name].sum())
                for class_name in CLASS_NAMES
            },
            "ptbxl_external": {
                class_name: int(ptbxl_metadata[class_name].sum()) for class_name in CLASS_NAMES
            },
        },
        "configuration": asdict(config),
        "device": str(device),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
        "positive_weights": dict(zip(CLASS_NAMES, positive_weights.tolist())),
        "normalization": {
            "lead_order": list(LEADS),
            "means_mv": means.tolist(),
            "standard_deviations_mv": standard_deviations.tolist(),
        },
        "threshold_policy": "Per-class maximum F1 on Chapman validation data; fixed for both tests.",
        "thresholds": dict(zip(CLASS_NAMES, thresholds.tolist())),
        "best_validation_macro_average_precision_during_training": best_validation_map,
        "chapman_validation_metrics": validation_metrics,
        "chapman_test_metrics": test_metrics,
        "ptbxl_external_metrics": external_metrics,
        "limitations": [
            "This engineering taxonomy contains only six rhythm labels and is not the final clinical taxonomy.",
            "Chapman records without any target label are retained as all-negative background examples.",
            "Chapman provides one ECG per published subject record and does not expose a separate patient identifier.",
            "PTB-XL external performance may be affected by acquisition and annotation domain shift.",
            "The PTB-XL atrial-flutter support is small, so its external estimate is unstable.",
            "No specialist adjudication, probability calibration, or hardware-domain validation has been completed.",
        ],
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "model_name": "RhythmECGNet",
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "display_names": DISPLAY_NAMES,
            "thresholds": thresholds,
            "sampling_frequency_hz": config.sampling_frequency_hz,
            "sample_count": config.sample_count,
            "lead_order": LEADS,
            "normalization_means_mv": means,
            "normalization_standard_deviations_mv": standard_deviations,
            "configuration": asdict(config),
            "intended_use": report["intended_use"],
        },
        output_dir / "model.pt",
    )
    plot_evaluation(
        history_frame, test_metrics, external_metrics, output_dir / "evaluation.png"
    )

    print("Chapman held-out macro AP: {:.4f}".format(test_metrics["macro_average_precision"]))
    print("Chapman held-out macro AUROC: {:.4f}".format(test_metrics["macro_roc_auc"]))
    print("PTB-XL external macro AP: {:.4f}".format(external_metrics["macro_average_precision"]))
    print("PTB-XL external macro AUROC: {:.4f}".format(external_metrics["macro_roc_auc"]))
    print("Artifacts written to {}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
