"""Train a PTB-XL baseline for normal-versus-abnormal ECG screening.

This is an experimental research benchmark. It is not a diagnostic model and it
must not be used for clinical decisions.
"""

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
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
ABNORMAL_DIAGNOSTIC_CLASSES = frozenset(("MI", "STTC", "CD", "HYP"))
HEADER_GAIN_PATTERN = re.compile(
    r"^(?P<gain>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\((?P<baseline>[+-]?\d+)\)/(?P<unit>\S+)$"
)
DEFAULT_DATASET_ROOT = Path(
    "database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260830
    epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    num_workers: int = 0
    sampling_frequency_hz: int = 100
    sample_count: int = 1000


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_scp_codes(value: str) -> Dict[str, float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("scp_codes must contain a dictionary")
    return {str(code): float(likelihood) for code, likelihood in parsed.items()}


def load_diagnostic_class_map(path: Path) -> Dict[str, str]:
    statements = pd.read_csv(path, index_col=0)
    diagnostic = statements.loc[statements["diagnostic"] == 1.0, "diagnostic_class"].dropna()
    return {str(code): str(diagnostic_class) for code, diagnostic_class in diagnostic.items()}


def diagnostic_classes_for_codes(
    codes: Mapping[str, float], class_map: Mapping[str, str]
) -> frozenset[str]:
    return frozenset(class_map[code] for code in codes if code in class_map)


def anomaly_label(diagnostic_classes: Iterable[str]) -> Optional[int]:
    """Return 0 for NORM-only, 1 for any abnormal class, or None if unlabeled."""

    classes = frozenset(diagnostic_classes)
    if not classes:
        return None
    if classes == frozenset(("NORM",)):
        return 0
    if classes.intersection(ABNORMAL_DIAGNOSTIC_CLASSES):
        return 1
    return None


def load_labeled_metadata(dataset_root: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    metadata = pd.read_csv(dataset_root / "ptbxl_database.csv")
    class_map = load_diagnostic_class_map(dataset_root / "scp_statements.csv")

    labels: List[Optional[int]] = []
    class_sets: List[str] = []
    for value in metadata["scp_codes"]:
        classes = diagnostic_classes_for_codes(parse_scp_codes(value), class_map)
        labels.append(anomaly_label(classes))
        class_sets.append("|".join(sorted(classes)))

    metadata = metadata.assign(anomaly_label=labels, diagnostic_classes=class_sets)
    excluded_count = int(metadata["anomaly_label"].isna().sum())
    labeled = metadata.loc[metadata["anomaly_label"].notna()].copy()
    labeled["anomaly_label"] = labeled["anomaly_label"].astype(np.int64)
    labeled.reset_index(drop=True, inplace=True)

    counts = {
        "metadata_records": int(len(metadata)),
        "labeled_records": int(len(labeled)),
        "excluded_without_binary_diagnostic_target": excluded_count,
        "normal_records": int((labeled["anomaly_label"] == 0).sum()),
        "abnormal_records": int((labeled["anomaly_label"] == 1).sum()),
    }
    return labeled, counts


def load_low_resolution_record(record_base: Path) -> np.ndarray:
    header_path = record_base.with_suffix(".hea")
    data_path = record_base.with_suffix(".dat")
    lines = header_path.read_text(encoding="ascii").splitlines()
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError("Malformed WFDB header: {}".format(header_path))

    signal_count, sampling_frequency, sample_count = int(first[1]), float(first[2]), int(first[3])
    if (signal_count, sampling_frequency, sample_count) != (12, 100.0, 1000):
        raise ValueError(
            "Expected a 12-lead, 100 Hz, 1000-sample record in {}".format(header_path)
        )
    if len(lines) < 13:
        raise ValueError("Incomplete WFDB signal definitions: {}".format(header_path))

    gains = []
    baselines = []
    leads = []
    for line in lines[1:13]:
        fields = line.split()
        if len(fields) < 9 or fields[1] != "16":
            raise ValueError("Unsupported WFDB signal definition in {}".format(header_path))
        match = HEADER_GAIN_PATTERN.match(fields[2])
        if match is None or match.group("unit") != "mV":
            raise ValueError("Unsupported ADC gain or unit in {}".format(header_path))
        gains.append(float(match.group("gain")))
        baselines.append(float(match.group("baseline")))
        leads.append(fields[-1].upper())

    if tuple(leads) != LEADS:
        raise ValueError("Unexpected lead order in {}: {}".format(header_path, leads))

    digital = np.fromfile(data_path, dtype="<i2")
    if digital.size != 12 * 1000:
        raise ValueError("Unexpected sample count in {}".format(data_path))
    digital = digital.reshape(1000, 12).T.astype(np.float32)
    return (digital - np.asarray(baselines, dtype=np.float32)[:, None]) / np.asarray(
        gains, dtype=np.float32
    )[:, None]


def metadata_fingerprint(metadata: pd.DataFrame) -> str:
    payload = "\n".join(
        "{}|{}|{}".format(row.ecg_id, row.filename_lr, row.anomaly_label)
        for row in metadata.itertuples()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_or_load_cache(
    dataset_root: Path, metadata: pd.DataFrame, cache_path: Path, workers: int = 16
) -> np.memmap:
    manifest_path = cache_path.with_suffix(".json")
    expected_manifest = {
        "format_version": 1,
        "metadata_fingerprint": metadata_fingerprint(metadata),
        "shape": [len(metadata), 12, 1000],
        "dtype": "float16",
        "unit": "mV",
        "lead_order": list(LEADS),
    }

    cache_valid = False
    if cache_path.is_file() and manifest_path.is_file():
        try:
            cache_valid = json.loads(manifest_path.read_text(encoding="utf-8")) == expected_manifest
        except (OSError, ValueError):
            cache_valid = False

    if not cache_valid:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".building.npy")
        waveform_cache = np.lib.format.open_memmap(
            temporary_path, mode="w+", dtype=np.float16, shape=(len(metadata), 12, 1000)
        )
        record_bases = [dataset_root / filename for filename in metadata["filename_lr"]]
        chunk_size = 512
        progress = tqdm(total=len(record_bases), desc="Building waveform cache")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for start in range(0, len(record_bases), chunk_size):
                chunk_paths = record_bases[start : start + chunk_size]
                chunk = np.stack(list(executor.map(load_low_resolution_record, chunk_paths)))
                waveform_cache[start : start + len(chunk)] = chunk.astype(np.float16)
                progress.update(len(chunk))
        progress.close()
        waveform_cache.flush()
        del waveform_cache
        temporary_path.replace(cache_path)
        manifest_path.write_text(json.dumps(expected_manifest, indent=2) + "\n", encoding="utf-8")

    return np.load(cache_path, mmap_mode="r")


def compute_normalization(
    waveforms: np.memmap, training_indices: Sequence[int], chunk_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(12, dtype=np.float64)
    squared_sums = np.zeros(12, dtype=np.float64)
    value_count = 0
    indices = np.asarray(training_indices, dtype=np.int64)
    for start in tqdm(range(0, len(indices), chunk_size), desc="Computing normalization"):
        batch = np.asarray(waveforms[indices[start : start + chunk_size]], dtype=np.float32)
        sums += batch.sum(axis=(0, 2), dtype=np.float64)
        squared_sums += np.square(batch, dtype=np.float32).sum(axis=(0, 2), dtype=np.float64)
        value_count += batch.shape[0] * batch.shape[2]
    means = sums / value_count
    variances = np.maximum(squared_sums / value_count - np.square(means), 1e-12)
    return means.astype(np.float32), np.sqrt(variances).astype(np.float32)


class ECGDataset(Dataset):
    def __init__(
        self,
        waveforms: np.memmap,
        metadata: pd.DataFrame,
        indices: Sequence[int],
        means: np.ndarray,
        standard_deviations: np.ndarray,
    ) -> None:
        self.waveforms = waveforms
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=np.int64)
        self.means = means[:, None]
        self.standard_deviations = standard_deviations[:, None]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item])
        waveform = np.asarray(self.waveforms[index], dtype=np.float32)
        waveform = (waveform - self.means) / self.standard_deviations
        label = float(self.metadata.iloc[index]["anomaly_label"])
        ecg_id = int(self.metadata.iloc[index]["ecg_id"])
        return torch.from_numpy(waveform), torch.tensor(label, dtype=torch.float32), ecg_id


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                input_channels, output_channels, kernel_size=7, stride=stride, padding=3, bias=False
            ),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channels, output_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(output_channels),
        )
        self.skip: nn.Module
        if stride != 1 or input_channels != output_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(output_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class AnomalyECGNet(nn.Module):
    def __init__(self) -> None:
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
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs)).squeeze(1)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if mask.any():
            error += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(error if total else math.nan)


def select_balanced_accuracy_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in candidates:
        score = balanced_accuracy_score(labels, probabilities >= threshold)
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def calculate_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = true_negative / max(true_negative + false_positive, 1)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "brier_score": float(np.mean(np.square(probabilities - labels))),
        "expected_calibration_error_10_bins": expected_calibration_error(labels, probabilities),
        "confusion_matrix": matrix.tolist(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_function: nn.Module, device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses: List[float] = []
    all_labels: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    all_ecg_ids: List[np.ndarray] = []
    for waveforms, labels, ecg_ids in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        logits = model(waveforms)
        losses.append(float(loss_function(logits, labels).item()) * len(labels))
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
        all_ecg_ids.append(ecg_ids.numpy())
    labels_array = np.concatenate(all_labels)
    return (
        sum(losses) / len(labels_array),
        labels_array.astype(np.int64),
        np.concatenate(all_probabilities),
        np.concatenate(all_ecg_ids),
    )


def plot_results(
    history: pd.DataFrame,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
    test_metrics: Mapping[str, object],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(history["epoch"], history["training_loss"], label="Training")
    axes[0, 0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0, 0].set(title="Loss", xlabel="Epoch", ylabel="Binary cross-entropy")
    axes[0, 0].legend()

    false_positive_rate, true_positive_rate, _ = roc_curve(test_labels, test_probabilities)
    axes[0, 1].plot(false_positive_rate, true_positive_rate)
    axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0, 1].set(
        title="Test ROC (AUROC {:.3f})".format(test_metrics["roc_auc"]),
        xlabel="False-positive rate",
        ylabel="Sensitivity",
    )

    precision, recall, _ = precision_recall_curve(test_labels, test_probabilities)
    axes[1, 0].plot(recall, precision)
    axes[1, 0].set(
        title="Test precision-recall (AP {:.3f})".format(test_metrics["average_precision"]),
        xlabel="Recall",
        ylabel="Precision",
    )

    matrix = np.asarray(test_metrics["confusion_matrix"])
    image = axes[1, 1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1, 1].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[1, 1].set(
        title="Test confusion matrix",
        xlabel="Predicted label",
        ylabel="Reference label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"],
    )
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046)
    figure.suptitle("PTB-XL experimental anomaly baseline")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache", type=Path, default=Path(".cache/ptbxl_anomaly_waveforms.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/anomaly_baseline"))
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
    )
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")

    set_reproducible_seed(config.seed)
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, dataset_counts = load_labeled_metadata(dataset_root)
    fold_indices = {
        "training": metadata.index[metadata["strat_fold"].between(1, 8)].to_numpy(),
        "validation": metadata.index[metadata["strat_fold"] == 9].to_numpy(),
        "test": metadata.index[metadata["strat_fold"] == 10].to_numpy(),
    }
    waveforms = build_or_load_cache(
        dataset_root, metadata, args.cache.resolve(), workers=max(1, min(16, config.num_workers or 16))
    )
    means, standard_deviations = compute_normalization(waveforms, fold_indices["training"])

    datasets = {
        split: ECGDataset(waveforms, metadata, indices, means, standard_deviations)
        for split, indices in fold_indices.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "training"),
            num_workers=config.num_workers,
            pin_memory=False,
        )
        for split, dataset in datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyECGNet().to(device)
    training_labels = metadata.iloc[fold_indices["training"]]["anomaly_label"].to_numpy()
    positive_weight = float((training_labels == 0).sum() / (training_labels == 1).sum())
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    history: List[Dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_auc = -math.inf
    epochs_without_improvement = 0
    print("Device: {}".format(device))
    print("Split sizes: {}".format({key: len(value) for key, value in datasets.items()}))
    print("Trainable parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    for epoch in range(1, config.epochs + 1):
        model.train()
        training_loss_sum = 0.0
        training_count = 0
        progress = tqdm(loaders["training"], desc="Epoch {}/{}".format(epoch, config.epochs))
        for waveforms_batch, labels_batch, _ in progress:
            waveforms_batch = waveforms_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms_batch)
            loss = loss_function(logits, labels_batch)
            loss.backward()
            optimizer.step()
            training_loss_sum += float(loss.item()) * len(labels_batch)
            training_count += len(labels_batch)
            progress.set_postfix(loss="{:.4f}".format(training_loss_sum / training_count))

        training_loss = training_loss_sum / training_count
        validation_loss, validation_labels, validation_probabilities, _ = evaluate(
            model, loaders["validation"], loss_function, device
        )
        validation_auc = float(roc_auc_score(validation_labels, validation_probabilities))
        scheduler.step(validation_auc)
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "validation_roc_auc": validation_auc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            "Epoch {:02d}: train loss {:.4f}, validation loss {:.4f}, validation AUROC {:.4f}".format(
                epoch, training_loss, validation_loss, validation_auc
            )
        )

        if validation_auc > best_validation_auc + 1e-4:
            best_validation_auc = validation_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print("Early stopping after epoch {}".format(epoch))
                break

    model.load_state_dict(best_state)
    validation_loss, validation_labels, validation_probabilities, validation_ecg_ids = evaluate(
        model, loaders["validation"], loss_function, device
    )
    threshold = select_balanced_accuracy_threshold(validation_labels, validation_probabilities)
    test_loss, test_labels, test_probabilities, test_ecg_ids = evaluate(
        model, loaders["test"], loss_function, device
    )
    validation_metrics = calculate_metrics(validation_labels, validation_probabilities, threshold)
    validation_metrics["loss"] = validation_loss
    test_metrics = calculate_metrics(test_labels, test_probabilities, threshold)
    test_metrics["loss"] = test_loss

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(
        {
            "ecg_id": validation_ecg_ids,
            "reference_label": validation_labels,
            "abnormal_probability": validation_probabilities,
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(
        {
            "ecg_id": test_ecg_ids,
            "reference_label": test_labels,
            "abnormal_probability": test_probabilities,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    report = {
        "experiment": "PTB-XL normal-versus-abnormal experimental baseline",
        "intended_use": "Research benchmark only; not a diagnostic or clinical-use model.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_definition": {
            "normal": "The only diagnostic superclass is NORM.",
            "abnormal": "At least one diagnostic superclass is MI, STTC, CD, or HYP.",
            "excluded": "No binary target can be derived from diagnostic SCP-ECG statements.",
        },
        "split_policy": "PTB-XL strat_fold 1-8 train, 9 validation, 10 test.",
        "threshold_policy": "Maximum balanced accuracy on validation; fixed before test evaluation.",
        "dataset_counts": dataset_counts,
        "split_counts": {key: int(len(value)) for key, value in fold_indices.items()},
        "configuration": asdict(config),
        "device": str(device),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
        "normalization": {
            "lead_order": list(LEADS),
            "means_mv": means.tolist(),
            "standard_deviations_mv": standard_deviations.tolist(),
        },
        "best_validation_roc_auc_during_training": best_validation_auc,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "model_name": "AnomalyECGNet",
            "model_state_dict": model.state_dict(),
            "configuration": asdict(config),
            "lead_order": LEADS,
            "sampling_frequency_hz": config.sampling_frequency_hz,
            "sample_count": config.sample_count,
            "normalization_means_mv": means,
            "normalization_standard_deviations_mv": standard_deviations,
            "decision_threshold": threshold,
            "label_definition": report["label_definition"],
        },
        output_dir / "model.pt",
    )
    plot_results(
        history_frame,
        test_labels,
        test_probabilities,
        test_metrics,
        output_dir / "evaluation.png",
    )

    print("Selected validation threshold: {:.4f}".format(threshold))
    print("Test AUROC: {:.4f}".format(test_metrics["roc_auc"]))
    print("Test average precision: {:.4f}".format(test_metrics["average_precision"]))
    print("Test sensitivity: {:.4f}".format(test_metrics["sensitivity"]))
    print("Test specificity: {:.4f}".format(test_metrics["specificity"]))
    print("Artifacts written to {}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
