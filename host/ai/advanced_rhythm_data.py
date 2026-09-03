"""Data utilities for reproducible reduced-lead ECG experiments."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from tqdm import tqdm

from advanced_rhythm_components import RHYTHM_FEATURE_NAMES, extract_rhythm_features
from train_anomaly_baseline import HEADER_GAIN_PATTERN, parse_scp_codes
from train_rhythm_classifier import (
    CLASS_NAMES,
    LEADS,
    PTBXL_CODE_TO_CLASS,
    cache_fingerprint,
    label_vector,
)


def load_ptbxl_rhythm_metadata(dataset_root: Path) -> pd.DataFrame:
    """Load every PTB-XL record and the approved six-label rhythm targets."""

    metadata = pd.read_csv(dataset_root / "ptbxl_database.csv")
    vectors = []
    label_likelihoods = []
    for value in metadata["scp_codes"]:
        codes = parse_scp_codes(value)
        selected = {PTBXL_CODE_TO_CLASS[code] for code in codes if code in PTBXL_CODE_TO_CLASS}
        vectors.append(label_vector(selected))
        label_likelihoods.append(
            json.dumps(
                {
                    PTBXL_CODE_TO_CLASS[code]: float(likelihood)
                    for code, likelihood in codes.items()
                    if code in PTBXL_CODE_TO_CLASS
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    matrix = np.stack(vectors)
    for index, class_name in enumerate(CLASS_NAMES):
        metadata[class_name] = matrix[:, index].astype(np.int64)
    metadata["labels"] = [
        "|".join(class_name for class_name, selected in zip(CLASS_NAMES, vector) if selected)
        for vector in matrix
    ]
    metadata["target_label_likelihoods"] = label_likelihoods
    return metadata.reset_index(drop=True)


def _parse_wfdb_physical_record(record_base: Path) -> Tuple[np.ndarray, int, Tuple[str, ...]]:
    header_path = record_base.with_suffix(".hea")
    data_path = record_base.with_suffix(".dat")
    lines = header_path.read_text(encoding="ascii").splitlines()
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError("Malformed WFDB header: {}".format(header_path))
    signal_count = int(first[1])
    sampling_frequency_hz = int(round(float(first[2])))
    sample_count = int(first[3])
    if signal_count != 12 or sample_count != sampling_frequency_hz * 10:
        raise ValueError("Expected a 10-second, 12-lead record in {}".format(header_path))
    gains = []
    baselines = []
    leads = []
    for line in lines[1:13]:
        fields = line.split()
        if len(fields) < 9 or fields[1] != "16":
            raise ValueError("Unsupported WFDB signal definition in {}".format(header_path))
        match = HEADER_GAIN_PATTERN.match(fields[2])
        if match is None or match.group("unit") != "mV":
            raise ValueError("Unsupported gain or unit in {}".format(header_path))
        gains.append(float(match.group("gain")))
        baselines.append(float(match.group("baseline")))
        leads.append(fields[-1].upper())
    expected = tuple(name.upper() for name in LEADS)
    if tuple(leads) != expected:
        raise ValueError("Unexpected lead order in {}".format(header_path))
    digital = np.fromfile(data_path, dtype="<i2")
    if digital.size != signal_count * sample_count:
        raise ValueError("Unexpected sample count in {}".format(data_path))
    digital = digital.reshape(sample_count, signal_count).T.astype(np.float32)
    physical = (digital - np.asarray(baselines, dtype=np.float32)[:, None])
    physical /= np.asarray(gains, dtype=np.float32)[:, None]
    return physical, sampling_frequency_hz, tuple(LEADS)


def _resample_to_frequency(
    waveform: np.ndarray, source_frequency_hz: int, target_frequency_hz: int
) -> np.ndarray:
    if target_frequency_hz == source_frequency_hz:
        return waveform.astype(np.float32, copy=False)
    divisor = int(np.gcd(source_frequency_hz, target_frequency_hz))
    result = resample_poly(
        waveform,
        up=target_frequency_hz // divisor,
        down=source_frequency_hz // divisor,
        axis=1,
    )
    expected_samples = target_frequency_hz * 10
    if result.shape[1] != expected_samples:
        if result.shape[1] > expected_samples:
            result = result[:, :expected_samples]
        else:
            result = np.pad(result, ((0, 0), (0, expected_samples - result.shape[1])))
    return result.astype(np.float32)


def load_wfdb_selected_record(
    record_base: Path,
    lead_names: Sequence[str],
    target_sampling_frequency_hz: int,
) -> np.ndarray:
    """Load and resample selected leads from a compatible 12-lead WFDB record."""

    if record_base.suffix.lower() in (".hea", ".dat"):
        record_base = record_base.with_suffix("")
    waveform, source_frequency_hz, source_leads = _parse_wfdb_physical_record(record_base)
    indices = np.asarray([source_leads.index(name) for name in lead_names], dtype=np.int64)
    return _resample_to_frequency(
        waveform[indices], source_frequency_hz, target_sampling_frequency_hz
    )


def _load_chapman_selected(
    dataset_root: Path,
    row: Mapping[str, object],
    lead_indices: np.ndarray,
    sampling_frequency_hz: int,
) -> np.ndarray:
    matrix = loadmat(
        (dataset_root / str(row["record_base"])).with_suffix(".mat"),
        variable_names=("val",),
    )["val"].astype(np.float32)
    gains = np.asarray(json.loads(str(row["gains"])), dtype=np.float32)[:, None]
    baselines = np.asarray(json.loads(str(row["baselines"])), dtype=np.float32)[:, None]
    physical_mv = (matrix - baselines) / gains
    return _resample_to_frequency(physical_mv[lead_indices], 500, sampling_frequency_hz)


def _load_ptbxl_selected(
    dataset_root: Path,
    row: Mapping[str, object],
    lead_indices: np.ndarray,
    sampling_frequency_hz: int,
) -> np.ndarray:
    filename_column = "filename_lr" if sampling_frequency_hz == 100 else "filename_hr"
    waveform, source_frequency_hz, _ = _parse_wfdb_physical_record(
        dataset_root / str(row[filename_column])
    )
    return _resample_to_frequency(
        waveform[lead_indices], source_frequency_hz, sampling_frequency_hz
    )


def _selected_cache_manifest(
    source: str,
    metadata: pd.DataFrame,
    lead_names: Sequence[str],
    sampling_frequency_hz: int,
) -> Dict[str, object]:
    if source == "chapman":
        fingerprint = cache_fingerprint(metadata)
    elif source == "ptbxl":
        payload = "\n".join(
            "{}|{}|{}|{}".format(row.ecg_id, row.filename_lr, row.filename_hr, row.labels)
            for row in metadata.itertuples()
        )
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    else:
        raise ValueError("Unsupported ECG source: {}".format(source))
    return {
        "format_version": 1,
        "source": source,
        "metadata_fingerprint": fingerprint,
        "shape": [len(metadata), len(lead_names), sampling_frequency_hz * 10],
        "dtype": "float16",
        "unit": "mV",
        "lead_order": list(lead_names),
        "sampling_frequency_hz": sampling_frequency_hz,
    }


def build_or_load_selected_waveform_cache(
    source: str,
    dataset_root: Path,
    metadata: pd.DataFrame,
    cache_path: Path,
    lead_names: Sequence[str],
    sampling_frequency_hz: int,
    workers: int,
) -> np.memmap:
    """Build a compact selected-lead cache at 100, 250, or 500 Hz."""

    if sampling_frequency_hz not in (100, 250, 500):
        raise ValueError("Sampling frequency must be 100, 250, or 500 Hz")
    if len(set(lead_names)) != len(lead_names) or any(name not in LEADS for name in lead_names):
        raise ValueError("Unsupported or duplicate lead selection")
    manifest = _selected_cache_manifest(source, metadata, lead_names, sampling_frequency_hz)
    manifest_path = cache_path.with_suffix(".json")
    valid = False
    if cache_path.is_file() and manifest_path.is_file():
        try:
            valid = json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
        except (OSError, ValueError):
            valid = False
    if not valid:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".building.npy")
        cache = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float16,
            shape=tuple(manifest["shape"]),
        )
        rows = metadata.to_dict(orient="records")
        lead_indices = np.asarray([LEADS.index(name) for name in lead_names], dtype=np.int64)

        def load(row: Mapping[str, object]) -> np.ndarray:
            if source == "chapman":
                return _load_chapman_selected(
                    dataset_root, row, lead_indices, sampling_frequency_hz
                )
            return _load_ptbxl_selected(
                dataset_root, row, lead_indices, sampling_frequency_hz
            )

        chunk_size = 128
        progress = tqdm(total=len(rows), desc="Building {} {} Hz cache".format(source, sampling_frequency_hz))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for start in range(0, len(rows), chunk_size):
                chunk = list(executor.map(load, rows[start : start + chunk_size]))
                cache[start : start + len(chunk)] = np.stack(chunk).astype(np.float16)
                progress.update(len(chunk))
        progress.close()
        cache.flush()
        del cache
        temporary_path.replace(cache_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return np.load(cache_path, mmap_mode="r")


def feature_cache_fingerprint(waveform_manifest: Mapping[str, object]) -> str:
    payload = json.dumps(waveform_manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_or_load_feature_cache(
    waveforms: np.memmap,
    waveform_cache_path: Path,
    feature_cache_path: Path,
    lead_names: Sequence[str],
    sampling_frequency_hz: int,
    workers: int,
) -> np.memmap:
    """Precompute deterministic rhythm features for one waveform cache."""

    waveform_manifest = json.loads(
        waveform_cache_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    manifest = {
        "format_version": 1,
        "waveform_fingerprint": feature_cache_fingerprint(waveform_manifest),
        "shape": [len(waveforms), len(RHYTHM_FEATURE_NAMES)],
        "dtype": "float32",
        "feature_names": list(RHYTHM_FEATURE_NAMES),
    }
    manifest_path = feature_cache_path.with_suffix(".json")
    valid = False
    if feature_cache_path.is_file() and manifest_path.is_file():
        try:
            valid = json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
        except (OSError, ValueError):
            valid = False
    if not valid:
        feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = feature_cache_path.with_suffix(".building.npy")
        cache = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float32,
            shape=tuple(manifest["shape"]),
        )

        def extract(index: int) -> np.ndarray:
            return extract_rhythm_features(
                np.asarray(waveforms[index], dtype=np.float32),
                sampling_frequency_hz,
                lead_names,
            )

        chunk_size = 256
        progress = tqdm(total=len(waveforms), desc="Extracting deterministic rhythm features")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for start in range(0, len(waveforms), chunk_size):
                indices = range(start, min(len(waveforms), start + chunk_size))
                chunk = list(executor.map(extract, indices))
                cache[start : start + len(chunk)] = np.stack(chunk)
                progress.update(len(chunk))
        progress.close()
        cache.flush()
        del cache
        temporary_path.replace(feature_cache_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return np.load(feature_cache_path, mmap_mode="r")
