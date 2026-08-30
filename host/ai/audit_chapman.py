"""Audit the Chapman-Shaoxing-Ningbo 12-lead ECG dataset.

The audit reads WFDB header files only. It does not load waveform matrices, so it
can validate dataset structure and label coverage quickly and without ML packages.
"""

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EXPECTED_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


def load_condition_map(path: Path) -> Dict[str, Dict[str, str]]:
    """Load the dataset's SNOMED CT condition mapping keyed by code."""

    conditions = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row["Snomed_CT"].strip()
            conditions[code] = {
                "acronym": row["Acronym Name"].strip(),
                "name": row["Full Name"].strip(),
            }
    return conditions


def parse_header(path: Path) -> Dict[str, object]:
    """Parse structural and diagnostic metadata from one WFDB header."""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        raise ValueError("Empty WFDB header: {}".format(path))

    first = lines[0].split()
    if len(first) < 4:
        raise ValueError("Invalid WFDB first line in {}: {!r}".format(path, lines[0]))

    record_name = first[0]
    signal_count = int(first[1])
    sampling_frequency_hz = float(first[2].split("/")[0])
    sample_count = int(first[3])

    signal_lines = lines[1 : 1 + signal_count]
    leads = tuple(line.split()[-1] for line in signal_lines)

    metadata = {}
    for line in lines[1 + signal_count :]:
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            metadata[key.strip()] = value.strip()

    diagnoses = tuple(
        code.strip() for code in metadata.get("Dx", "").split(",") if code.strip()
    )

    return {
        "record_name": record_name,
        "signal_count": signal_count,
        "sampling_frequency_hz": sampling_frequency_hz,
        "sample_count": sample_count,
        "leads": leads,
        "diagnoses": diagnoses,
        "age": metadata.get("Age"),
        "sex": metadata.get("Sex"),
    }


def iter_headers(records_root: Path) -> Iterable[Path]:
    return records_root.rglob("*.hea")


def _parse_header_safely(path: Path) -> Tuple[Path, Optional[Dict[str, object]], Optional[str]]:
    try:
        return path, parse_header(path), None
    except (OSError, TypeError, ValueError) as error:
        return path, None, str(error)


def audit_dataset(dataset_root: Path, workers: int = 16) -> Dict[str, object]:
    """Return a JSON-serializable structural and label audit."""

    records_root = dataset_root / "WFDBRecords"
    condition_map_path = dataset_root / "ConditionNames_SNOMED-CT.csv"
    if not records_root.is_dir():
        raise FileNotFoundError("WFDBRecords directory not found: {}".format(records_root))
    if not condition_map_path.is_file():
        raise FileNotFoundError(
            "ConditionNames_SNOMED-CT.csv not found: {}".format(condition_map_path)
        )

    condition_map = load_condition_map(condition_map_path)
    label_counts = Counter()
    signal_count_counts = Counter()
    sampling_frequency_counts = Counter()
    sample_count_counts = Counter()
    lead_order_counts = Counter()
    diagnoses_per_record_counts = Counter()
    sex_counts = Counter()
    unmapped_codes = Counter()
    parse_errors: List[Dict[str, str]] = []
    structural_anomalies: List[Dict[str, object]] = []
    total_records = 0

    header_paths = list(iter_headers(records_root))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        parsed_headers = executor.map(_parse_header_safely, header_paths)
        for header_path, record, error in parsed_headers:
            if error is not None:
                parse_errors.append({"path": str(header_path), "error": error})
                continue

            total_records += 1
            signal_count_counts[str(record["signal_count"])] += 1
            sampling_frequency_counts[str(record["sampling_frequency_hz"])] += 1
            sample_count_counts[str(record["sample_count"])] += 1
            lead_order_counts["|".join(record["leads"])] += 1
            diagnoses_per_record_counts[str(len(record["diagnoses"]))] += 1
            sex_counts[str(record["sex"])] += 1

            if (
                record["signal_count"] != len(EXPECTED_LEADS)
                or record["sampling_frequency_hz"] != 500.0
                or record["sample_count"] != 5000
                or record["leads"] != EXPECTED_LEADS
            ):
                structural_anomalies.append(
                    {
                        "path": str(header_path),
                        "record_name": record["record_name"],
                        "signal_count": record["signal_count"],
                        "sampling_frequency_hz": record["sampling_frequency_hz"],
                        "sample_count": record["sample_count"],
                        "leads": list(record["leads"]),
                    }
                )

            for code in record["diagnoses"]:
                label_counts[code] += 1
                if code not in condition_map:
                    unmapped_codes[code] += 1

    labels = []
    for code, count in label_counts.most_common():
        condition = condition_map.get(code, {})
        labels.append(
            {
                "snomed_ct": code,
                "acronym": condition.get("acronym", "UNMAPPED"),
                "name": condition.get("name", "Not listed in local condition map"),
                "record_count": count,
                "record_prevalence": count / total_records if total_records else 0.0,
            }
        )

    expected_lead_key = "|".join(EXPECTED_LEADS)
    return {
        "dataset": "Chapman-Shaoxing-Ningbo 12-lead ECG",
        "dataset_root": str(dataset_root.resolve()),
        "header_file_count": len(header_paths),
        "total_records": total_records,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "structural_anomaly_count": len(structural_anomalies),
        "structural_anomalies": structural_anomalies,
        "signal_count_distribution": dict(signal_count_counts),
        "sampling_frequency_hz_distribution": dict(sampling_frequency_counts),
        "sample_count_distribution": dict(sample_count_counts),
        "expected_lead_order_records": lead_order_counts.get(expected_lead_key, 0),
        "lead_order_distribution": dict(lead_order_counts),
        "diagnoses_per_record_distribution": dict(diagnoses_per_record_counts),
        "sex_distribution": dict(sex_counts),
        "unique_diagnosis_codes": len(label_counts),
        "unmapped_diagnosis_codes": dict(unmapped_codes),
        "labels": labels,
    }


def print_summary(audit: Dict[str, object], top_labels: int) -> None:
    print("Dataset: {}".format(audit["dataset"]))
    print("Header files: {:,}".format(audit["header_file_count"]))
    print("Successfully parsed records: {:,}".format(audit["total_records"]))
    print("Header parse errors: {:,}".format(audit["parse_error_count"]))
    print("Structural anomalies: {:,}".format(audit["structural_anomaly_count"]))
    print("Signal counts: {}".format(audit["signal_count_distribution"]))
    print("Sampling frequencies: {}".format(audit["sampling_frequency_hz_distribution"]))
    print("Sample counts: {}".format(audit["sample_count_distribution"]))
    print(
        "Canonical 12-lead order: {:,}/{:,}".format(
            audit["expected_lead_order_records"], audit["total_records"]
        )
    )
    print("Unique diagnosis codes: {}".format(audit["unique_diagnosis_codes"]))
    print("Codes absent from local condition map: {}".format(audit["unmapped_diagnosis_codes"]))
    print("Top diagnosis labels:")
    for label in audit["labels"][:top_labels]:
        print(
            "  {acronym:>6}  {record_count:>6,}  {record_prevalence:>7.2%}  "
            "{snomed_ct}  {name}".format(**label)
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Path to the extracted dataset")
    parser.add_argument(
        "--output-json", type=Path, help="Optional path for the full JSON audit"
    )
    parser.add_argument(
        "--top-labels", type=int, default=20, help="Number of labels shown in the console"
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Concurrent header readers"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    audit = audit_dataset(args.dataset_root, workers=args.workers)
    print_summary(audit, args.top_labels)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print("Audit written to {}".format(args.output_json))

    return 0 if not audit["parse_error_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
