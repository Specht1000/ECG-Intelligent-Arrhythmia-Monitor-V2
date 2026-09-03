"""Reusable components for advanced reduced-lead ECG rhythm experiments.

All quality limits and augmentations in this module are experimental research
settings. They are not approved hardware specifications or clinical criteria.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import butter, find_peaks, sosfiltfilt
from torch import nn
from torch.nn import functional as F

from train_anomaly_baseline import ResidualBlock


RHYTHM_FEATURE_NAMES = (
    "heart_rate_bpm",
    "rr_mean_seconds",
    "rr_standard_deviation_seconds",
    "rr_rmssd_seconds",
    "rr_pnn50",
    "detected_qrs_count",
    "signal_robust_range_mv",
    "derivative_mad_mv",
    "baseline_power_ratio",
    "high_frequency_power_ratio",
    "spectral_entropy",
    "einthoven_rmse_mv",
)


@dataclass(frozen=True)
class ExperimentalQualityLimits:
    minimum_lead_standard_deviation_mv: float = 0.01
    maximum_robust_range_mv: float = 8.0
    maximum_derivative_mad_mv: float = 0.25
    maximum_baseline_power_ratio: float = 0.65
    maximum_high_frequency_power_ratio: float = 0.45
    maximum_einthoven_rmse_mv: float = 0.10
    minimum_qrs_count: int = 3
    maximum_qrs_count: int = 35


@dataclass(frozen=True)
class SignalQualityAssessment:
    acceptable: bool
    score: float
    reasons: Tuple[str, ...]
    metrics: Mapping[str, float]
    policy: str = "Experimental software research gate; not clinically approved."

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        result["metrics"] = dict(self.metrics)
        return result


@lru_cache(maxsize=32)
def _design_sos_filter(
    sampling_frequency_hz: int,
    low_hz: Optional[float] = None,
    high_hz: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Return reusable filter coefficients for one research configuration."""

    nyquist = sampling_frequency_hz / 2.0
    if low_hz is not None and high_hz is not None:
        upper = min(high_hz, nyquist * 0.90)
        lower = min(low_hz, upper * 0.50)
        return butter(
            2,
            (lower, upper),
            btype="bandpass",
            fs=sampling_frequency_hz,
            output="sos",
        )
    elif high_hz is not None:
        upper = min(high_hz, nyquist * 0.90)
        return butter(2, upper, btype="lowpass", fs=sampling_frequency_hz, output="sos")
    elif low_hz is not None:
        lower = min(low_hz, nyquist * 0.45)
        return butter(2, lower, btype="highpass", fs=sampling_frequency_hz, output="sos")
    return None


def _safe_sos_filter(
    values: np.ndarray,
    sampling_frequency_hz: int,
    low_hz: Optional[float] = None,
    high_hz: Optional[float] = None,
) -> np.ndarray:
    sos = _design_sos_filter(sampling_frequency_hz, low_hz, high_hz)
    if sos is None:
        return values.copy()
    try:
        return sosfiltfilt(sos, values).astype(np.float32)
    except ValueError:
        return values.astype(np.float32, copy=True)


def detect_qrs_peaks(signal_mv: np.ndarray, sampling_frequency_hz: int) -> np.ndarray:
    """Return experimental Pan-Tompkins-inspired QRS peak indices."""

    signal_mv = np.asarray(signal_mv, dtype=np.float32)
    filtered = _safe_sos_filter(signal_mv, sampling_frequency_hz, 5.0, 20.0)
    derivative = np.diff(filtered, prepend=filtered[0])
    energy = np.square(derivative, dtype=np.float32)
    window = max(1, int(round(0.12 * sampling_frequency_hz)))
    integrated = np.convolve(energy, np.ones(window, dtype=np.float32) / window, mode="same")
    median = float(np.median(integrated))
    high_quantile = float(np.quantile(integrated, 0.90))
    prominence = max((high_quantile - median) * 0.35, np.finfo(np.float32).eps)
    peaks, _ = find_peaks(
        integrated,
        distance=max(1, int(round(0.25 * sampling_frequency_hz))),
        prominence=prominence,
    )
    search_radius = max(1, int(round(0.08 * sampling_frequency_hz)))
    refined = []
    for peak in peaks:
        start = max(0, peak - search_radius)
        stop = min(len(filtered), peak + search_radius + 1)
        refined.append(start + int(np.argmax(np.abs(filtered[start:stop]))))
    return np.unique(np.asarray(refined, dtype=np.int64))


def extract_rhythm_features(
    waveform_mv: np.ndarray,
    sampling_frequency_hz: int,
    lead_names: Sequence[str],
) -> np.ndarray:
    """Extract deterministic rhythm and signal-shape features from one ECG."""

    waveform = np.asarray(waveform_mv, dtype=np.float32)
    if waveform.ndim != 2 or waveform.shape[0] != len(lead_names):
        raise ValueError("Waveform shape and lead names do not match")
    if waveform.shape[1] < sampling_frequency_hz:
        raise ValueError("At least one second of ECG is required")
    if not np.isfinite(waveform).all():
        return np.zeros(len(RHYTHM_FEATURE_NAMES), dtype=np.float32)

    lead_index = lead_names.index("II") if "II" in lead_names else 0
    signal_mv = waveform[lead_index]
    peaks = detect_qrs_peaks(signal_mv, sampling_frequency_hz)
    rr = np.diff(peaks).astype(np.float32) / float(sampling_frequency_hz)
    physiologic_rr = rr[(rr >= 0.25) & (rr <= 2.5)]
    if len(physiologic_rr):
        rr_mean = float(np.mean(physiologic_rr))
        heart_rate = 60.0 / max(rr_mean, 1e-6)
        rr_standard_deviation = float(np.std(physiologic_rr))
        rr_differences = np.diff(physiologic_rr)
        rr_rmssd = (
            float(np.sqrt(np.mean(np.square(rr_differences))))
            if len(rr_differences)
            else 0.0
        )
        rr_pnn50 = (
            float(np.mean(np.abs(rr_differences) > 0.05)) if len(rr_differences) else 0.0
        )
    else:
        heart_rate = rr_mean = rr_standard_deviation = rr_rmssd = rr_pnn50 = 0.0

    lower, upper = np.quantile(signal_mv, (0.01, 0.99))
    robust_range = float(upper - lower)
    derivative_mad = float(np.median(np.abs(np.diff(signal_mv))))
    centered = signal_mv - float(np.mean(signal_mv))
    total_power = float(np.mean(np.square(centered))) + 1e-12
    baseline = _safe_sos_filter(centered, sampling_frequency_hz, high_hz=0.7)
    baseline_ratio = float(np.mean(np.square(baseline)) / total_power)
    if sampling_frequency_hz > 45:
        high_frequency = _safe_sos_filter(centered, sampling_frequency_hz, low_hz=35.0)
        high_frequency_ratio = float(np.mean(np.square(high_frequency)) / total_power)
    else:
        high_frequency_ratio = 0.0
    spectrum = np.square(np.abs(np.fft.rfft(centered)).astype(np.float64))
    spectrum = spectrum[1:]
    distribution = spectrum / max(float(spectrum.sum()), 1e-12)
    spectral_entropy = float(
        -np.sum(distribution * np.log(distribution + 1e-12))
        / max(np.log(max(len(distribution), 2)), 1e-12)
    )
    if all(name in lead_names for name in ("I", "II", "III")):
        residual = waveform[lead_names.index("II")] - waveform[lead_names.index("I")]
        residual -= waveform[lead_names.index("III")]
        einthoven_rmse = float(np.sqrt(np.mean(np.square(residual))))
    else:
        einthoven_rmse = 0.0

    features = np.asarray(
        (
            heart_rate,
            rr_mean,
            rr_standard_deviation,
            rr_rmssd,
            rr_pnn50,
            float(len(peaks)),
            robust_range,
            derivative_mad,
            baseline_ratio,
            high_frequency_ratio,
            spectral_entropy,
            einthoven_rmse,
        ),
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def assess_signal_quality(
    waveform_mv: np.ndarray,
    sampling_frequency_hz: int,
    lead_names: Sequence[str],
    limits: ExperimentalQualityLimits = ExperimentalQualityLimits(),
) -> SignalQualityAssessment:
    """Apply an explicitly experimental signal-quality gate."""

    waveform = np.asarray(waveform_mv, dtype=np.float32)
    reasons = []
    if waveform.ndim != 2 or waveform.shape[0] != len(lead_names):
        raise ValueError("Waveform shape and lead names do not match")
    if not np.isfinite(waveform).all():
        return SignalQualityAssessment(
            acceptable=False,
            score=0.0,
            reasons=("non_finite_samples",),
            metrics={"non_finite_sample_count": float((~np.isfinite(waveform)).sum())},
        )
    features = extract_rhythm_features(waveform, sampling_frequency_hz, lead_names)
    values = dict(zip(RHYTHM_FEATURE_NAMES, features.astype(float)))
    minimum_standard_deviation = float(np.min(np.std(waveform, axis=1)))
    values["minimum_lead_standard_deviation_mv"] = minimum_standard_deviation
    if minimum_standard_deviation < limits.minimum_lead_standard_deviation_mv:
        reasons.append("flat_or_disconnected_lead")
    if values["signal_robust_range_mv"] > limits.maximum_robust_range_mv:
        reasons.append("excessive_amplitude")
    if values["derivative_mad_mv"] > limits.maximum_derivative_mad_mv:
        reasons.append("high_derivative_noise")
    if values["baseline_power_ratio"] > limits.maximum_baseline_power_ratio:
        reasons.append("baseline_wander")
    if values["high_frequency_power_ratio"] > limits.maximum_high_frequency_power_ratio:
        reasons.append("high_frequency_noise")
    qrs_count = int(round(values["detected_qrs_count"]))
    if qrs_count < limits.minimum_qrs_count or qrs_count > limits.maximum_qrs_count:
        reasons.append("implausible_qrs_count")
    if (
        all(name in lead_names for name in ("I", "II", "III"))
        and values["einthoven_rmse_mv"] > limits.maximum_einthoven_rmse_mv
    ):
        reasons.append("einthoven_inconsistency")
    score = max(0.0, 1.0 - len(set(reasons)) / 6.0)
    return SignalQualityAssessment(
        acceptable=not reasons,
        score=score,
        reasons=tuple(sorted(set(reasons))),
        metrics=values,
    )


class AsymmetricMultiLabelLoss(nn.Module):
    """Asymmetric focal loss with optional clipping of easy negatives."""

    def __init__(
        self,
        gamma_negative: float = 4.0,
        gamma_positive: float = 1.0,
        negative_clip: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.negative_clip = negative_clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        positive_probabilities = torch.sigmoid(logits)
        negative_probabilities = 1.0 - positive_probabilities
        if self.negative_clip > 0:
            negative_probabilities = (negative_probabilities + self.negative_clip).clamp(max=1.0)
        positive_loss = targets * torch.log(positive_probabilities.clamp_min(1e-8))
        negative_loss = (1.0 - targets) * torch.log(negative_probabilities.clamp_min(1e-8))
        weights = torch.pow(1.0 - positive_probabilities, self.gamma_positive) * targets
        weights += torch.pow(1.0 - negative_probabilities, self.gamma_negative) * (1.0 - targets)
        return -(weights * (positive_loss + negative_loss)).mean()


class AdvancedSignalEncoder(nn.Module):
    def __init__(self, input_channel_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channel_count, 24, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            ResidualBlock(24, 24),
            ResidualBlock(24, 48, stride=2),
            ResidualBlock(48, 96, stride=2),
            ResidualBlock(96, 128, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MaskedSignalAutoencoder(nn.Module):
    """Small denoising autoencoder used only to initialize the signal encoder."""

    def __init__(self, input_channel_count: int) -> None:
        super().__init__()
        self.encoder = AdvancedSignalEncoder(input_channel_count)
        self.decoder_projection = nn.Conv1d(128, input_channel_count, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        reconstructed = self.decoder_projection(encoded)
        return F.interpolate(reconstructed, size=inputs.shape[-1], mode="linear", align_corners=False)


class AdvancedRhythmNet(nn.Module):
    """Waveform/feature fusion model with auxiliary atrial hierarchy heads."""

    def __init__(
        self,
        input_channel_count: int,
        class_count: int,
        feature_count: int = len(RHYTHM_FEATURE_NAMES),
    ) -> None:
        super().__init__()
        self.signal_encoder = AdvancedSignalEncoder(input_channel_count)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_count, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(160, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
        )
        self.classifier = nn.Linear(160, class_count)
        self.atrial_family_head = nn.Linear(160, 1)
        self.atrial_type_head = nn.Linear(160, 1)

    def encode(self, waveforms: torch.Tensor, rhythm_features: torch.Tensor) -> torch.Tensor:
        signal_embedding = self.pool(self.signal_encoder(waveforms)).flatten(1)
        feature_embedding = self.feature_encoder(rhythm_features)
        return self.fusion(torch.cat((signal_embedding, feature_embedding), dim=1))

    def forward_all(
        self, waveforms: torch.Tensor, rhythm_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        embedding = self.encode(waveforms, rhythm_features)
        return {
            "class_logits": self.classifier(embedding),
            "atrial_family_logit": self.atrial_family_head(embedding).squeeze(1),
            "atrial_type_logit": self.atrial_type_head(embedding).squeeze(1),
        }

    def forward(self, waveforms: torch.Tensor, rhythm_features: torch.Tensor) -> torch.Tensor:
        return self.forward_all(waveforms, rhythm_features)["class_logits"]


def augment_normalized_waveforms(
    waveforms: torch.Tensor,
    sampling_frequency_hz: int,
    lead_mask_probability: float = 0.05,
) -> torch.Tensor:
    """Apply conservative research augmentations to a normalized training batch."""

    result = waveforms.clone()
    batch_size, channel_count, sample_count = result.shape
    device = result.device
    result *= torch.empty(batch_size, channel_count, 1, device=device).uniform_(0.90, 1.10)
    result += torch.randn_like(result) * torch.empty(
        batch_size, channel_count, 1, device=device
    ).uniform_(0.0, 0.025)
    time = torch.arange(sample_count, device=device, dtype=result.dtype)
    time = time / float(sampling_frequency_hz)
    baseline_frequency = torch.empty(batch_size, 1, 1, device=device).uniform_(0.15, 0.50)
    baseline_phase = torch.empty(batch_size, 1, 1, device=device).uniform_(0.0, 2.0 * np.pi)
    baseline_amplitude = torch.empty(batch_size, channel_count, 1, device=device).uniform_(0.0, 0.04)
    result += baseline_amplitude * torch.sin(
        2.0 * np.pi * baseline_frequency * time.view(1, 1, -1) + baseline_phase
    )
    if sampling_frequency_hz > 120:
        mains_frequency = torch.where(
            torch.rand(batch_size, 1, 1, device=device) < 0.5,
            torch.tensor(50.0, device=device),
            torch.tensor(60.0, device=device),
        )
        mains_amplitude = torch.empty(batch_size, channel_count, 1, device=device).uniform_(0.0, 0.015)
        result += mains_amplitude * torch.sin(
            2.0 * np.pi * mains_frequency * time.view(1, 1, -1) + baseline_phase
        )
    if lead_mask_probability > 0 and channel_count > 1:
        mask = torch.rand(batch_size, channel_count, 1, device=device) < lead_mask_probability
        all_masked = mask.all(dim=1, keepdim=True)
        mask[:, :1] &= ~all_masked
        result = result.masked_fill(mask, 0.0)
    return result


def mask_waveform_regions(
    waveforms: torch.Tensor,
    mask_fraction: float = 0.20,
    region_fraction: float = 0.08,
) -> torch.Tensor:
    """Mask contiguous time regions for denoising pre-training."""

    result = waveforms.clone()
    sample_count = result.shape[-1]
    region_length = max(1, int(round(sample_count * region_fraction)))
    region_count = max(1, int(round(mask_fraction / region_fraction)))
    for batch_index in range(result.shape[0]):
        for _ in range(region_count):
            start = int(torch.randint(0, max(1, sample_count - region_length + 1), (1,)).item())
            result[batch_index, :, start : start + region_length] = 0.0
    return result
