"""Train the advanced multi-dataset bipolar limb-lead rhythm classifier.

This script preserves PTB-XL fold 10 as the untouched final test. PTB-XL folds
1-8 join the Chapman training split, while PTB-XL fold 9 joins validation. The
pipeline is an engineering experiment and is not intended for clinical use.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from advanced_rhythm_components import (
    RHYTHM_FEATURE_NAMES,
    AdvancedRhythmNet,
    AsymmetricMultiLabelLoss,
    MaskedSignalAutoencoder,
    augment_normalized_waveforms,
    mask_waveform_regions,
)
from advanced_rhythm_data import (
    build_or_load_feature_cache,
    build_or_load_selected_waveform_cache,
    load_ptbxl_rhythm_metadata,
)
from analyze_anomaly_errors import apply_temperature, fit_temperature
from train_rhythm_classifier import (
    BIPOLAR_LEADS,
    CHAPMAN_CODE_TO_CLASS,
    CLASS_NAMES,
    DEFAULT_CHAPMAN_ROOT,
    DEFAULT_PTBXL_ROOT,
    DISPLAY_NAMES,
    LEADS,
    PTBXL_CODE_TO_CLASS,
    RhythmECGNet,
    build_or_load_chapman_index,
    make_chapman_splits,
    multilabel_metrics,
)


@dataclass(frozen=True)
class AdvancedTrainingConfig:
    seed: int = 20260831
    epochs: int = 20
    pretraining_epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    sampling_frequency_hz: int = 100
    waveform_workers: int = 12
    data_loader_workers: int = 0
    minimum_specificity: float = 0.95
    atrial_minimum_specificity: float = 0.98
    hierarchical_loss_weight: float = 0.20
    hard_negative_loss_weight: float = 0.30
    distillation_loss_weight: float = 0.15
    distillation_temperature: float = 2.0


@dataclass
class SourceBundle:
    name: str
    metadata: pd.DataFrame
    waveforms: np.memmap
    features: np.memmap
    labels: np.ndarray
    identifier_column: str
    teacher_probabilities: Optional[np.ndarray] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resume_signature(
    config: AdvancedTrainingConfig,
    input_leads: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "format_version": 1,
        "configuration": asdict(config),
        "input_leads": list(input_leads),
        "training_limit": args.training_limit,
        "validation_limit": args.validation_limit,
        "disable_distillation": bool(args.disable_distillation),
        "disable_balanced_sampling": bool(args.disable_balanced_sampling),
        "disable_augmentation": bool(args.disable_augmentation),
    }


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".building")
    torch.save(value, temporary_path)
    temporary_path.replace(path)


def _random_state(generator: Optional[torch.Generator]) -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "sampler_generator": generator.get_state() if generator is not None else None,
    }


def _restore_random_state(
    state: Mapping[str, object], generator: Optional[torch.Generator]
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if generator is not None and state.get("sampler_generator") is not None:
        generator.set_state(state["sampler_generator"])


def _indices_by_ptbxl_fold(metadata: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "training": np.flatnonzero(metadata["strat_fold"].between(1, 8).to_numpy()),
        "validation": np.flatnonzero((metadata["strat_fold"] == 9).to_numpy()),
        "test": np.flatnonzero((metadata["strat_fold"] == 10).to_numpy()),
    }


def _references(source_index: int, indices: Sequence[int]) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    return np.column_stack((np.full(len(indices), source_index, dtype=np.int64), indices))


def limit_references(
    references: np.ndarray, maximum_count: Optional[int], seed: int
) -> np.ndarray:
    """Apply a deterministic experiment-only record limit."""

    if maximum_count is None or maximum_count >= len(references):
        return references
    if maximum_count < 1:
        raise ValueError("Reference limit must be positive")
    generator = np.random.default_rng(seed)
    selected = np.sort(generator.choice(len(references), size=maximum_count, replace=False))
    return references[selected]


def compute_combined_normalization(
    sources: Sequence[SourceBundle], references: np.ndarray, chunk_size: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_count = sources[0].waveforms.shape[1]
    feature_count = sources[0].features.shape[1]
    waveform_sum = np.zeros(channel_count, dtype=np.float64)
    waveform_square_sum = np.zeros(channel_count, dtype=np.float64)
    waveform_count = 0
    feature_sum = np.zeros(feature_count, dtype=np.float64)
    feature_square_sum = np.zeros(feature_count, dtype=np.float64)
    feature_count_total = 0
    for source_index, source in enumerate(sources):
        rows = references[references[:, 0] == source_index, 1]
        for start in tqdm(
            range(0, len(rows), chunk_size),
            desc="Normalizing {}".format(source.name),
        ):
            selected = rows[start : start + chunk_size]
            waveform_batch = np.asarray(source.waveforms[selected], dtype=np.float32)
            feature_batch = np.asarray(source.features[selected], dtype=np.float32)
            waveform_sum += waveform_batch.sum(axis=(0, 2), dtype=np.float64)
            waveform_square_sum += np.square(waveform_batch).sum(axis=(0, 2), dtype=np.float64)
            waveform_count += waveform_batch.shape[0] * waveform_batch.shape[2]
            feature_sum += feature_batch.sum(axis=0, dtype=np.float64)
            feature_square_sum += np.square(feature_batch).sum(axis=0, dtype=np.float64)
            feature_count_total += len(feature_batch)
    waveform_means = waveform_sum / waveform_count
    waveform_variances = np.maximum(
        waveform_square_sum / waveform_count - np.square(waveform_means), 1e-12
    )
    feature_means = feature_sum / feature_count_total
    feature_variances = np.maximum(
        feature_square_sum / feature_count_total - np.square(feature_means), 1e-12
    )
    return (
        waveform_means.astype(np.float32),
        np.sqrt(waveform_variances).astype(np.float32),
        feature_means.astype(np.float32),
        np.sqrt(feature_variances).astype(np.float32),
    )


class CombinedRhythmDataset(Dataset):
    def __init__(
        self,
        sources: Sequence[SourceBundle],
        references: np.ndarray,
        waveform_means: np.ndarray,
        waveform_standard_deviations: np.ndarray,
        feature_means: np.ndarray,
        feature_standard_deviations: np.ndarray,
    ) -> None:
        self.sources = sources
        self.references = np.asarray(references, dtype=np.int64)
        self.waveform_means = waveform_means[:, None]
        self.waveform_standard_deviations = waveform_standard_deviations[:, None]
        self.feature_means = feature_means
        self.feature_standard_deviations = feature_standard_deviations

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, item: int):
        source_index, row_index = self.references[item]
        source = self.sources[int(source_index)]
        waveform = np.asarray(source.waveforms[int(row_index)], dtype=np.float32)
        features = np.asarray(source.features[int(row_index)], dtype=np.float32)
        waveform = (waveform - self.waveform_means) / self.waveform_standard_deviations
        features = (features - self.feature_means) / self.feature_standard_deviations
        if source.teacher_probabilities is None:
            teacher_probabilities = np.full(len(CLASS_NAMES), np.nan, dtype=np.float32)
        else:
            teacher_probabilities = source.teacher_probabilities[int(row_index)].astype(np.float32)
        return (
            torch.from_numpy(waveform),
            torch.from_numpy(features),
            torch.from_numpy(source.labels[int(row_index)]),
            torch.from_numpy(teacher_probabilities),
            int(source_index),
            int(row_index),
        )


def _balanced_sample_weights(sources: Sequence[SourceBundle], references: np.ndarray) -> np.ndarray:
    labels = np.stack(
        [sources[source_index].labels[row_index] for source_index, row_index in references]
    )
    prevalence = np.maximum(labels.mean(axis=0), 1e-4)
    class_weights = np.minimum(1.0 / np.sqrt(prevalence), 5.0)
    weights = 1.0 + (labels * class_weights[None, :]).sum(axis=1) / max(len(CLASS_NAMES), 1)
    weights[labels.sum(axis=1) == 0] *= 1.5
    return np.clip(weights, 0.5, 4.0).astype(np.float64)


def _teacher_cache_path(output_cache_root: Path, source_name: str, checkpoint: Path) -> Path:
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:12]
    return output_cache_root / "teacher_{}_{}.npy".format(source_name, digest)


@torch.no_grad()
def build_or_load_teacher_probabilities(
    source_name: str,
    full_waveforms: np.memmap,
    teacher_checkpoint_path: Path,
    cache_root: Path,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(teacher_checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["class_names"]) != CLASS_NAMES or tuple(checkpoint["lead_order"]) != LEADS:
        raise ValueError("Teacher checkpoint must use the approved classes and all 12 leads")
    cache_path = _teacher_cache_path(cache_root, source_name, teacher_checkpoint_path)
    manifest_path = cache_path.with_suffix(".json")
    expected_manifest = {
        "format_version": 1,
        "source": source_name,
        "record_count": len(full_waveforms),
        "teacher_sha256": hashlib.sha256(teacher_checkpoint_path.read_bytes()).hexdigest(),
        "class_names": list(CLASS_NAMES),
    }
    if cache_path.is_file() and manifest_path.is_file():
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")) == expected_manifest:
                return np.load(cache_path, mmap_mode="r")
        except (OSError, ValueError):
            pass
    teacher = RhythmECGNet(class_count=len(CLASS_NAMES), input_channel_count=len(LEADS)).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    means = np.asarray(checkpoint["normalization_means_mv"], dtype=np.float32)[:, None]
    deviations = np.asarray(
        checkpoint["normalization_standard_deviations_mv"], dtype=np.float32
    )[:, None]
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".building.npy")
    probabilities = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(len(full_waveforms), len(CLASS_NAMES))
    )
    for start in tqdm(range(0, len(full_waveforms), batch_size), desc="Distilling {} teacher".format(source_name)):
        batch = np.asarray(full_waveforms[start : start + batch_size], dtype=np.float32)
        normalized = (batch - means) / deviations
        logits = teacher(torch.from_numpy(normalized).to(device))
        probabilities[start : start + len(batch)] = torch.sigmoid(logits).cpu().numpy().astype(np.float16)
    probabilities.flush()
    del probabilities
    temporary.replace(cache_path)
    manifest_path.write_text(json.dumps(expected_manifest, indent=2) + "\n", encoding="utf-8")
    return np.load(cache_path, mmap_mode="r")


def constrained_threshold(
    labels: np.ndarray, probabilities: np.ndarray, minimum_specificity: float
) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    candidates = np.unique(np.concatenate(([0.0], thresholds, [1.0])))
    best_threshold = 0.5
    best_f1 = -1.0
    negatives = labels == 0
    for threshold in candidates:
        predictions = probabilities >= threshold
        specificity = float(np.mean(~predictions[negatives])) if negatives.any() else 1.0
        if specificity + 1e-12 < minimum_specificity:
            continue
        true_positive = int(np.sum(predictions & (labels == 1)))
        false_positive = int(np.sum(predictions & (labels == 0)))
        false_negative = int(np.sum((~predictions) & (labels == 1)))
        f1 = 2.0 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def select_constrained_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum_specificity: float,
    atrial_minimum_specificity: float,
) -> np.ndarray:
    return np.asarray(
        [
            constrained_threshold(
                labels[:, index],
                probabilities[:, index],
                atrial_minimum_specificity if index >= 4 else minimum_specificity,
            )
            for index in range(len(CLASS_NAMES))
        ],
        dtype=np.float64,
    )


def calculate_loss(
    model_outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    classification_loss: nn.Module,
    config: AdvancedTrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = model_outputs["class_logits"]
    base_loss = classification_loss(logits, labels)
    family_target = labels[:, 4:6].amax(dim=1)
    family_loss = F.binary_cross_entropy_with_logits(
        model_outputs["atrial_family_logit"], family_target
    )
    pure_atrial = labels[:, 4] != labels[:, 5]
    if pure_atrial.any():
        type_loss = F.binary_cross_entropy_with_logits(
            model_outputs["atrial_type_logit"][pure_atrial], labels[pure_atrial, 5]
        )
    else:
        type_loss = logits.sum() * 0.0
    all_negative = labels.sum(dim=1) == 0
    if all_negative.any():
        hard_logits = logits[all_negative].topk(k=min(2, logits.shape[1]), dim=1).values
        hard_negative_loss = F.binary_cross_entropy_with_logits(
            hard_logits, torch.zeros_like(hard_logits)
        )
    else:
        hard_negative_loss = logits.sum() * 0.0
    valid_teacher = torch.isfinite(teacher_probabilities).all(dim=1)
    if valid_teacher.any() and config.distillation_loss_weight > 0:
        temperature = config.distillation_temperature
        softened_teacher = torch.sigmoid(
            torch.logit(teacher_probabilities[valid_teacher].clamp(1e-5, 1.0 - 1e-5))
            / temperature
        )
        distillation_loss = F.binary_cross_entropy_with_logits(
            logits[valid_teacher] / temperature, softened_teacher
        ) * (temperature**2)
    else:
        distillation_loss = logits.sum() * 0.0
    total = base_loss
    total += config.hierarchical_loss_weight * (family_loss + type_loss)
    total += config.hard_negative_loss_weight * hard_negative_loss
    total += config.distillation_loss_weight * distillation_loss
    return total, {
        "classification": float(base_loss.detach()),
        "atrial_family": float(family_loss.detach()),
        "atrial_type": float(type_loss.detach()),
        "hard_negative": float(hard_negative_loss.detach()),
        "distillation": float(distillation_loss.detach()),
    }


@torch.no_grad()
def evaluate(
    model: AdvancedRhythmNet,
    loader: DataLoader,
    loss_function: nn.Module,
    config: AdvancedTrainingConfig,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    count = 0
    labels_output = []
    probabilities_output = []
    sources_output = []
    rows_output = []
    for waveforms, features, labels, teacher, source_indices, row_indices in loader:
        waveforms = waveforms.to(device)
        features = features.to(device)
        labels = labels.to(device)
        teacher = teacher.to(device)
        outputs = model.forward_all(waveforms, features)
        loss, _ = calculate_loss(outputs, labels, teacher, loss_function, config)
        loss_sum += float(loss.item()) * len(labels)
        count += len(labels)
        labels_output.append(labels.cpu().numpy())
        probabilities_output.append(torch.sigmoid(outputs["class_logits"]).cpu().numpy())
        sources_output.append(source_indices.numpy())
        rows_output.append(row_indices.numpy())
    return (
        loss_sum / count,
        np.concatenate(labels_output),
        np.concatenate(probabilities_output),
        np.concatenate(sources_output),
        np.concatenate(rows_output),
    )


def pretrain_encoder(
    model: AdvancedRhythmNet,
    loader: DataLoader,
    config: AdvancedTrainingConfig,
    device: torch.device,
) -> List[Dict[str, float]]:
    if config.pretraining_epochs <= 0:
        return []
    autoencoder = MaskedSignalAutoencoder(model.signal_encoder.network[0].in_channels).to(device)
    optimizer = torch.optim.AdamW(
        autoencoder.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = []
    for epoch in range(1, config.pretraining_epochs + 1):
        autoencoder.train()
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(loader, desc="Self-supervised epoch {}/{}".format(epoch, config.pretraining_epochs))
        for waveforms, _, _, _, _, _ in progress:
            waveforms = waveforms.to(device)
            corrupted = mask_waveform_regions(waveforms)
            corrupted = augment_normalized_waveforms(
                corrupted, config.sampling_frequency_hz, lead_mask_probability=0.10
            )
            optimizer.zero_grad(set_to_none=True)
            reconstructed = autoencoder(corrupted)
            loss = F.mse_loss(reconstructed, waveforms)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(waveforms)
            sample_count += len(waveforms)
            progress.set_postfix(loss="{:.4f}".format(loss_sum / sample_count))
        history.append({"epoch": epoch, "reconstruction_loss": loss_sum / sample_count})
    model.signal_encoder.load_state_dict(autoencoder.encoder.state_dict())
    return history


def _prediction_frame(
    source: SourceBundle,
    rows: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {source.identifier_column: source.metadata.iloc[rows][source.identifier_column].to_numpy()}
    )
    for index, class_name in enumerate(CLASS_NAMES):
        result[class_name + "_reference"] = labels[:, index].astype(np.int64)
        result[class_name + "_probability"] = probabilities[:, index]
        result[class_name + "_calibrated_probability"] = calibrated_probabilities[:, index]
        result[class_name + "_prediction"] = (
            calibrated_probabilities[:, index] >= thresholds[index]
        ).astype(np.int64)
    return result


def _plot_results(
    history: pd.DataFrame,
    chapman_metrics: Mapping[str, object],
    ptbxl_metrics: Mapping[str, object],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["epoch"], history["training_loss"], label="Training")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].set_title("Advanced training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    x = np.arange(len(CLASS_NAMES))
    width = 0.38
    axes[1].bar(
        x - width / 2,
        [entry["average_precision"] for entry in chapman_metrics["per_class"]],
        width,
        label="Chapman held-out",
    )
    axes[1].bar(
        x + width / 2,
        [entry["average_precision"] for entry in ptbxl_metrics["per_class"]],
        width,
        label="PTB-XL fold 10",
    )
    axes[1].set_xticks(x, DISPLAY_NAMES, rotation=35, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Average precision")
    axes[1].legend()
    figure.suptitle("Advanced bipolar rhythm classifier")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapman-root", type=Path, default=DEFAULT_CHAPMAN_ROOT)
    parser.add_argument("--ptbxl-root", type=Path, default=DEFAULT_PTBXL_ROOT)
    parser.add_argument("--index", type=Path, default=Path(".cache/chapman_rhythm_index_all_records.csv"))
    parser.add_argument("--cache-root", type=Path, default=Path(".cache/advanced_rhythm"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/advanced_bipolar_rhythm_classifier"))
    parser.add_argument("--input-leads", nargs="+", choices=BIPOLAR_LEADS, default=list(BIPOLAR_LEADS))
    parser.add_argument("--sampling-frequency", type=int, choices=(100, 250, 500), default=100)
    parser.add_argument("--epochs", type=int, default=AdvancedTrainingConfig.epochs)
    parser.add_argument("--pretraining-epochs", type=int, default=AdvancedTrainingConfig.pretraining_epochs)
    parser.add_argument("--batch-size", type=int, default=AdvancedTrainingConfig.batch_size)
    parser.add_argument("--patience", type=int, default=AdvancedTrainingConfig.patience)
    parser.add_argument("--learning-rate", type=float, default=AdvancedTrainingConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=AdvancedTrainingConfig.seed)
    parser.add_argument("--waveform-workers", type=int, default=AdvancedTrainingConfig.waveform_workers)
    parser.add_argument("--minimum-specificity", type=float, default=AdvancedTrainingConfig.minimum_specificity)
    parser.add_argument("--atrial-minimum-specificity", type=float, default=AdvancedTrainingConfig.atrial_minimum_specificity)
    parser.add_argument("--teacher-model", type=Path, default=Path("artifacts/rhythm_classifier/model.pt"))
    parser.add_argument("--training-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--disable-distillation", action="store_true")
    parser.add_argument("--disable-balanced-sampling", action="store_true")
    parser.add_argument("--disable-augmentation", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    input_leads = tuple(args.input_leads)
    if len(set(input_leads)) != len(input_leads):
        raise ValueError("Input leads must not contain duplicates")
    if input_leads not in (("I", "II"), BIPOLAR_LEADS):
        raise ValueError("The approved ablation inputs are I II or I II III")
    config = AdvancedTrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        pretraining_epochs=args.pretraining_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        sampling_frequency_hz=args.sampling_frequency,
        waveform_workers=args.waveform_workers,
        minimum_specificity=args.minimum_specificity,
        atrial_minimum_specificity=args.atrial_minimum_specificity,
        distillation_loss_weight=(
            0.0 if args.disable_distillation else AdvancedTrainingConfig.distillation_loss_weight
        ),
    )
    if config.epochs < 1 or config.batch_size < 1 or config.pretraining_epochs < 0:
        raise ValueError("Epoch and batch settings must be positive")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    cache_root = args.cache_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    chapman_root = args.chapman_root.resolve()
    ptbxl_root = args.ptbxl_root.resolve()

    chapman_metadata = build_or_load_chapman_index(
        chapman_root, args.index.resolve(), config.waveform_workers
    )
    ptbxl_metadata = load_ptbxl_rhythm_metadata(ptbxl_root)
    chapman_splits = make_chapman_splits(chapman_metadata, config.seed)
    ptbxl_splits = _indices_by_ptbxl_fold(ptbxl_metadata)
    lead_tag = "-".join(input_leads).lower()
    rate_tag = str(config.sampling_frequency_hz)
    chapman_waveform_path = cache_root / "chapman_{}_{}hz.npy".format(lead_tag, rate_tag)
    ptbxl_waveform_path = cache_root / "ptbxl_{}_{}hz.npy".format(lead_tag, rate_tag)
    chapman_waveforms = build_or_load_selected_waveform_cache(
        "chapman", chapman_root, chapman_metadata, chapman_waveform_path,
        input_leads, config.sampling_frequency_hz, config.waveform_workers,
    )
    ptbxl_waveforms = build_or_load_selected_waveform_cache(
        "ptbxl", ptbxl_root, ptbxl_metadata, ptbxl_waveform_path,
        input_leads, config.sampling_frequency_hz, config.waveform_workers,
    )
    chapman_features = build_or_load_feature_cache(
        chapman_waveforms, chapman_waveform_path,
        cache_root / "chapman_{}_{}hz_features.npy".format(lead_tag, rate_tag),
        input_leads, config.sampling_frequency_hz, config.waveform_workers,
    )
    ptbxl_features = build_or_load_feature_cache(
        ptbxl_waveforms, ptbxl_waveform_path,
        cache_root / "ptbxl_{}_{}hz_features.npy".format(lead_tag, rate_tag),
        input_leads, config.sampling_frequency_hz, config.waveform_workers,
    )
    sources = [
        SourceBundle(
            "chapman", chapman_metadata, chapman_waveforms, chapman_features,
            chapman_metadata.loc[:, CLASS_NAMES].to_numpy(dtype=np.float32), "record_id",
        ),
        SourceBundle(
            "ptbxl", ptbxl_metadata, ptbxl_waveforms, ptbxl_features,
            ptbxl_metadata.loc[:, CLASS_NAMES].to_numpy(dtype=np.float32), "ecg_id",
        ),
    ]

    teacher_path = args.teacher_model.resolve()
    if config.distillation_loss_weight > 0:
        if config.sampling_frequency_hz != 100:
            raise ValueError("The available 12-lead teacher supports only 100 Hz")
        if not teacher_path.is_file():
            raise FileNotFoundError("Teacher checkpoint not found: {}".format(teacher_path))
        chapman_full_path = Path(".cache/chapman_rhythm_waveforms_all_records.npy").resolve()
        chapman_full = np.load(chapman_full_path, mmap_mode="r")
        ptbxl_full_path = cache_root / "ptbxl_12lead_100hz.npy"
        ptbxl_full = build_or_load_selected_waveform_cache(
            "ptbxl", ptbxl_root, ptbxl_metadata, ptbxl_full_path,
            LEADS, 100, config.waveform_workers,
        )
        sources[0].teacher_probabilities = build_or_load_teacher_probabilities(
            "chapman", chapman_full, teacher_path, cache_root,
            config.batch_size, device,
        )
        sources[1].teacher_probabilities = build_or_load_teacher_probabilities(
            "ptbxl", ptbxl_full, teacher_path, cache_root,
            config.batch_size, device,
        )

    training_references = np.concatenate(
        (_references(0, chapman_splits["training"]), _references(1, ptbxl_splits["training"]))
    )
    validation_references = np.concatenate(
        (_references(0, chapman_splits["validation"]), _references(1, ptbxl_splits["validation"]))
    )
    training_references = limit_references(
        training_references, args.training_limit, config.seed
    )
    validation_references = limit_references(
        validation_references, args.validation_limit, config.seed + 1
    )
    chapman_test_references = _references(0, chapman_splits["test"])
    ptbxl_test_references = _references(1, ptbxl_splits["test"])
    normalization = compute_combined_normalization(sources, training_references)
    datasets = {
        "training": CombinedRhythmDataset(sources, training_references, *normalization),
        "validation": CombinedRhythmDataset(sources, validation_references, *normalization),
        "chapman_test": CombinedRhythmDataset(sources, chapman_test_references, *normalization),
        "ptbxl_test": CombinedRhythmDataset(sources, ptbxl_test_references, *normalization),
    }
    generator: Optional[torch.Generator] = None
    if args.disable_balanced_sampling:
        training_sampler = None
        training_shuffle = True
    else:
        weights = _balanced_sample_weights(sources, training_references)
        generator = torch.Generator().manual_seed(config.seed)
        training_sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=generator
        )
        training_shuffle = False
    loaders = {
        "training": DataLoader(
            datasets["training"], batch_size=config.batch_size,
            shuffle=training_shuffle, sampler=training_sampler,
            num_workers=config.data_loader_workers,
        ),
        "validation": DataLoader(datasets["validation"], batch_size=config.batch_size),
        "chapman_test": DataLoader(datasets["chapman_test"], batch_size=config.batch_size),
        "ptbxl_test": DataLoader(datasets["ptbxl_test"], batch_size=config.batch_size),
    }

    model = AdvancedRhythmNet(len(input_leads), len(CLASS_NAMES)).to(device)
    print("Device: {}".format(device))
    print("Input leads: {} at {} Hz".format(", ".join(input_leads), config.sampling_frequency_hz))
    print("Training records: {:,}".format(len(training_references)))
    print("Validation records: {:,}".format(len(validation_references)))
    print("Chapman held-out records: {:,}".format(len(chapman_test_references)))
    print("PTB-XL fold 10 records: {:,}".format(len(ptbxl_test_references)))
    print("Trainable parameters: {:,}".format(sum(parameter.numel() for parameter in model.parameters())))
    resume_path = output_dir / "training_resume.pt"
    signature = _resume_signature(config, input_leads, args)
    resume_checkpoint = None
    if resume_path.is_file():
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_checkpoint.get("signature") != signature:
            raise ValueError(
                "The existing resume checkpoint belongs to a different experiment; "
                "use a different output directory or remove it explicitly."
            )
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        pretraining_history = list(resume_checkpoint["pretraining_history"])
        print("Resuming from {}".format(resume_path))
    else:
        pretraining_history = pretrain_encoder(model, loaders["training"], config, device)
        _atomic_torch_save(
            {
                "phase": "pretrained",
                "signature": signature,
                "model_state_dict": model.state_dict(),
                "pretraining_history": pretraining_history,
                "random_state": _random_state(generator),
            },
            resume_path,
        )

    classification_loss = AsymmetricMultiLabelLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(loaders["training"]),
        pct_start=0.20,
    )
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_map = -math.inf
    epochs_without_improvement = 0
    start_epoch = 1
    if resume_checkpoint is not None:
        _restore_random_state(resume_checkpoint["random_state"], generator)
        if resume_checkpoint["phase"] == "supervised":
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            history = list(resume_checkpoint["history"])
            best_state = resume_checkpoint["best_model_state_dict"]
            best_validation_map = float(resume_checkpoint["best_validation_map"])
            epochs_without_improvement = int(
                resume_checkpoint["epochs_without_improvement"]
            )
            start_epoch = int(resume_checkpoint["completed_epoch"]) + 1
            print("Continuing at supervised epoch {}".format(start_epoch))
        elif resume_checkpoint["phase"] != "pretrained":
            raise ValueError("Unsupported resume-checkpoint phase")
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(loaders["training"], desc="Advanced epoch {}/{}".format(epoch, config.epochs))
        for waveforms, features, labels, teacher, _, _ in progress:
            waveforms = waveforms.to(device)
            features = features.to(device)
            labels = labels.to(device)
            teacher = teacher.to(device)
            if not args.disable_augmentation:
                waveforms = augment_normalized_waveforms(
                    waveforms, config.sampling_frequency_hz
                )
            optimizer.zero_grad(set_to_none=True)
            outputs = model.forward_all(waveforms, features)
            loss, _ = calculate_loss(outputs, labels, teacher, classification_loss, config)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.item()) * len(labels)
            sample_count += len(labels)
            progress.set_postfix(loss="{:.4f}".format(loss_sum / sample_count))
        training_loss = loss_sum / sample_count
        validation_loss, validation_labels, validation_probabilities, _, _ = evaluate(
            model, loaders["validation"], classification_loss, config, device
        )
        validation_map = float(
            np.mean(
                [
                    average_precision_score(
                        validation_labels[:, index], validation_probabilities[:, index]
                    )
                    for index in range(len(CLASS_NAMES))
                ]
            )
        )
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
            "Epoch {:02d}: training loss {:.4f}, validation loss {:.4f}, validation macro AP {:.4f}".format(
                epoch, training_loss, validation_loss, validation_map
            )
        )
        should_stop = False
        if validation_map > best_validation_map + 1e-4:
            best_validation_map = validation_map
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                should_stop = True
        _atomic_torch_save(
            {
                "phase": "supervised",
                "signature": signature,
                "completed_epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_model_state_dict": best_state,
                "best_validation_map": best_validation_map,
                "epochs_without_improvement": epochs_without_improvement,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "pretraining_history": pretraining_history,
                "random_state": _random_state(generator),
            },
            resume_path,
        )
        if should_stop:
            print("Early stopping after epoch {}".format(epoch))
            break
    model.load_state_dict(best_state)

    validation_result = evaluate(model, loaders["validation"], classification_loss, config, device)
    validation_labels = validation_result[1]
    validation_raw = validation_result[2]
    temperatures = np.asarray(
        [
            fit_temperature(validation_labels[:, index], validation_raw[:, index])
            for index in range(len(CLASS_NAMES))
        ],
        dtype=np.float64,
    )
    validation_calibrated = np.column_stack(
        [
            apply_temperature(validation_raw[:, index], temperatures[index])
            for index in range(len(CLASS_NAMES))
        ]
    )
    thresholds = select_constrained_thresholds(
        validation_labels,
        validation_calibrated,
        config.minimum_specificity,
        config.atrial_minimum_specificity,
    )

    evaluation_outputs = {}
    for split in ("validation", "chapman_test", "ptbxl_test"):
        loss, labels, raw, source_indices, rows = evaluate(
            model, loaders[split], classification_loss, config, device
        )
        calibrated = np.column_stack(
            [apply_temperature(raw[:, index], temperatures[index]) for index in range(len(CLASS_NAMES))]
        )
        metrics = multilabel_metrics(labels, calibrated, thresholds)
        metrics["loss"] = loss
        evaluation_outputs[split] = {
            "labels": labels,
            "raw": raw,
            "calibrated": calibrated,
            "source_indices": source_indices,
            "rows": rows,
            "metrics": metrics,
        }

    for split, source_index, filename in (
        ("chapman_test", 0, "chapman_test_predictions.csv"),
        ("ptbxl_test", 1, "ptbxl_fold10_predictions.csv"),
    ):
        values = evaluation_outputs[split]
        if not np.all(values["source_indices"] == source_index):
            raise RuntimeError("Unexpected source in {}".format(split))
        _prediction_frame(
            sources[source_index], values["rows"], values["labels"], values["raw"],
            values["calibrated"], thresholds,
        ).to_csv(output_dir / filename, index=False)

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(pretraining_history).to_csv(output_dir / "pretraining_history.csv", index=False)
    report = {
        "experiment": "Advanced multi-dataset bipolar rhythm classifier",
        "intended_use": "Research benchmark only; not a diagnostic or clinical-use model.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_leads": list(input_leads),
        "sampling_frequency_hz": config.sampling_frequency_hz,
        "class_names": list(CLASS_NAMES),
        "display_names": list(DISPLAY_NAMES),
        "configuration": asdict(config),
        "data_policy": {
            "chapman": "80% training, 10% validation, 10% held-out test using the fixed seed.",
            "ptbxl": "Official folds 1-8 training, fold 9 validation, fold 10 untouched final test.",
            "training": "Chapman training plus PTB-XL folds 1-8.",
            "validation": "Chapman validation plus PTB-XL fold 9.",
            "test": "Chapman held-out and PTB-XL fold 10 evaluated separately.",
            "training_limit": args.training_limit,
            "validation_limit": args.validation_limit,
        },
        "record_counts": {
            "training": int(len(training_references)),
            "validation": int(len(validation_references)),
            "chapman_test": int(len(chapman_test_references)),
            "ptbxl_fold10_test": int(len(ptbxl_test_references)),
        },
        "methods": {
            "loss": "Asymmetric focal classification loss with hard-negative penalty.",
            "hierarchy": "Auxiliary atrial-family and AF-versus-AFL heads.",
            "features": list(RHYTHM_FEATURE_NAMES),
            "self_supervised_pretraining": config.pretraining_epochs > 0,
            "teacher_distillation": config.distillation_loss_weight > 0,
            "balanced_sampling": not args.disable_balanced_sampling,
            "research_augmentation": not args.disable_augmentation,
            "probability_calibration": "Per-class temperature scaling fitted only on combined validation data.",
            "threshold_policy": "Maximum validation F1 subject to minimum specificity constraints.",
        },
        "normalization": {
            "waveform_means_mv": normalization[0].tolist(),
            "waveform_standard_deviations_mv": normalization[1].tolist(),
            "feature_means": normalization[2].tolist(),
            "feature_standard_deviations": normalization[3].tolist(),
        },
        "temperatures": dict(zip(CLASS_NAMES, temperatures.tolist())),
        "thresholds": dict(zip(CLASS_NAMES, thresholds.tolist())),
        "best_validation_macro_average_precision": best_validation_map,
        "validation_metrics": evaluation_outputs["validation"]["metrics"],
        "chapman_test_metrics": evaluation_outputs["chapman_test"]["metrics"],
        "ptbxl_fold10_metrics": evaluation_outputs["ptbxl_test"]["metrics"],
        "limitations": [
            "The six-label taxonomy is an engineering benchmark and is not specialist-approved for clinical use.",
            "Quality thresholds and augmentations are experimental and are not hardware requirements.",
            "The 12-lead teacher predates the multi-dataset experiment and may transfer its own errors.",
            "PTB-XL fold 10 contains only a small number of atrial-flutter records.",
            "No data acquired by the final PCB is available for hardware-domain validation.",
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
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "model_name": "AdvancedRhythmNet",
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "display_names": DISPLAY_NAMES,
            "lead_order": input_leads,
            "sampling_frequency_hz": config.sampling_frequency_hz,
            "sample_count": config.sampling_frequency_hz * 10,
            "feature_names": RHYTHM_FEATURE_NAMES,
            "waveform_means_mv": normalization[0],
            "waveform_standard_deviations_mv": normalization[1],
            "feature_means": normalization[2],
            "feature_standard_deviations": normalization[3],
            "temperatures": temperatures,
            "thresholds": thresholds,
            "configuration": asdict(config),
            "intended_use": report["intended_use"],
        },
        output_dir / "model.pt",
    )
    _plot_results(
        history_frame,
        evaluation_outputs["chapman_test"]["metrics"],
        evaluation_outputs["ptbxl_test"]["metrics"],
        output_dir / "evaluation.png",
    )
    if resume_path.is_file():
        resume_path.unlink()
    print("Chapman held-out macro AP: {:.4f}".format(
        evaluation_outputs["chapman_test"]["metrics"]["macro_average_precision"]
    ))
    print("PTB-XL fold 10 macro AP: {:.4f}".format(
        evaluation_outputs["ptbxl_test"]["metrics"]["macro_average_precision"]
    ))
    print("Artifacts written to {}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
