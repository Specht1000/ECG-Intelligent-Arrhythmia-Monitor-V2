"""Audit PTB-XL metadata and 500 Hz WFDB headers without loading waveforms."""

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from audit_chapman import EXPECTED_LEADS, parse_header


PTBXL_LEAD_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


def normalize_leads(leads: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(PTBXL_LEAD_ALIASES.get(lead, lead) for lead in leads)


def parse_scp_codes(value: str) -> Dict[str, float]:
    """Parse the Python-dictionary representation used by PTB-XL metadata."""

    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("scp_codes must contain a dictionary")

    result = {}
    for code, likelihood in parsed.items():
        if not isinstance(code, str):
            raise ValueError("SCP-ECG code must be a string: {!r}".format(code))
        result[code] = float(likelihood)
    return result


def load_scp_statements(path: Path) -> Dict[str, Dict[str, str]]:
    """Load SCP-ECG statement metadata keyed by statement code."""

    statements = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            padded = row + [""] * (len(header) - len(row))
            statements[padded[0].strip()] = {
                key: value.strip() for key, value in zip(header[1:], padded[1:])
            }
    return statements


def _parse_header_safely(path: Path):
    try:
        return path, parse_header(path), None
    except (OSError, TypeError, ValueError) as error:
        return path, None, str(error)


def audit_dataset(dataset_root: Path, workers: int = 16) -> Dict[str, object]:
    metadata_path = dataset_root / "ptbxl_database.csv"
    statements_path = dataset_root / "scp_statements.csv"
    records_root = dataset_root / "records500"

    for required_path in (metadata_path, statements_path, records_root):
        if not required_path.exists():
            raise FileNotFoundError("Required PTB-XL path not found: {}".format(required_path))

    statements = load_scp_statements(statements_path)
    label_counts = Counter()
    labels_per_record_counts = Counter()
    fold_counts = Counter()
    patient_record_counts = Counter()
    patient_folds = defaultdict(set)
    human_validation_counts = Counter()
    missing_metadata_files = []
    invalid_metadata_rows = []
    metadata_row_count = 0

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            metadata_row_count += 1
            try:
                ecg_id = row["ecg_id"].strip()
                patient_id = row["patient_id"].strip()
                fold = row["strat_fold"].strip()
                labels = parse_scp_codes(row["scp_codes"])
                relative_record = Path(row["filename_hr"].strip())
            except (KeyError, SyntaxError, TypeError, ValueError) as error:
                invalid_metadata_rows.append(
                    {"row_number": metadata_row_count + 1, "error": str(error)}
                )
                continue

            labels_per_record_counts[str(len(labels))] += 1
            label_counts.update(labels.keys())
            fold_counts[fold] += 1
            patient_record_counts[patient_id] += 1
            patient_folds[patient_id].add(fold)
            human_validation_counts[row["validated_by_human"].strip()] += 1

            record_base = dataset_root / relative_record
            missing_extensions = [
                extension
                for extension in (".hea", ".dat")
                if not record_base.with_suffix(extension).is_file()
            ]
            if missing_extensions:
                missing_metadata_files.append(
                    {
                        "ecg_id": ecg_id,
                        "record": str(relative_record),
                        "missing_extensions": missing_extensions,
                    }
                )

    patients_in_multiple_folds = {
        patient_id: sorted(folds)
        for patient_id, folds in patient_folds.items()
        if len(folds) > 1
    }

    header_paths = list(records_root.rglob("*.hea"))
    signal_count_counts = Counter()
    sampling_frequency_counts = Counter()
    sample_count_counts = Counter()
    normalized_lead_order_counts = Counter()
    parse_errors = []
    structural_anomalies = []
    parsed_header_count = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for path, record, error in executor.map(_parse_header_safely, header_paths):
            if error is not None:
                parse_errors.append({"path": str(path), "error": error})
                continue

            parsed_header_count += 1
            normalized_lead_order = normalize_leads(record["leads"])
            signal_count_counts[str(record["signal_count"])] += 1
            sampling_frequency_counts[str(record["sampling_frequency_hz"])] += 1
            sample_count_counts[str(record["sample_count"])] += 1
            normalized_lead_order_counts["|".join(normalized_lead_order)] += 1

            if (
                record["signal_count"] != len(EXPECTED_LEADS)
                or record["sampling_frequency_hz"] != 500.0
                or record["sample_count"] != 5000
                or normalized_lead_order != EXPECTED_LEADS
            ):
                structural_anomalies.append(
                    {
                        "path": str(path),
                        "record_name": record["record_name"],
                        "signal_count": record["signal_count"],
                        "sampling_frequency_hz": record["sampling_frequency_hz"],
                        "sample_count": record["sample_count"],
                        "leads": list(record["leads"]),
                        "normalized_leads": list(normalized_lead_order),
                    }
                )

    labels = []
    for code, count in label_counts.most_common():
        statement = statements.get(code, {})
        labels.append(
            {
                "scp_code": code,
                "description": statement.get("description", "Not found in scp_statements.csv"),
                "record_count": count,
                "record_prevalence": count / metadata_row_count if metadata_row_count else 0.0,
                "diagnostic": statement.get("diagnostic", ""),
                "form": statement.get("form", ""),
                "rhythm": statement.get("rhythm", ""),
                "diagnostic_class": statement.get("diagnostic_class", ""),
                "diagnostic_subclass": statement.get("diagnostic_subclass", ""),
            }
        )

    repeated_patient_count = sum(1 for count in patient_record_counts.values() if count > 1)
    expected_lead_key = "|".join(EXPECTED_LEADS)
    return {
        "dataset": "PTB-XL 1.0.3",
        "dataset_root": str(dataset_root.resolve()),
        "metadata_row_count": metadata_row_count,
        "invalid_metadata_row_count": len(invalid_metadata_rows),
        "invalid_metadata_rows": invalid_metadata_rows,
        "unique_patient_count": len(patient_record_counts),
        "patients_with_multiple_records": repeated_patient_count,
        "patients_in_multiple_folds": patients_in_multiple_folds,
        "fold_distribution": dict(sorted(fold_counts.items())),
        "human_validation_distribution": dict(human_validation_counts),
        "labels_per_record_distribution": dict(labels_per_record_counts),
        "unique_scp_code_count": len(label_counts),
        "codes_missing_statement_metadata": sorted(set(label_counts) - set(statements)),
        "labels": labels,
        "metadata_records_with_missing_files": missing_metadata_files,
        "header_file_count": len(header_paths),
        "parsed_header_count": parsed_header_count,
        "header_parse_errors": parse_errors,
        "structural_anomalies": structural_anomalies,
        "signal_count_distribution": dict(signal_count_counts),
        "sampling_frequency_hz_distribution": dict(sampling_frequency_counts),
        "sample_count_distribution": dict(sample_count_counts),
        "canonical_12_lead_records": normalized_lead_order_counts.get(expected_lead_key, 0),
        "normalized_lead_order_distribution": dict(normalized_lead_order_counts),
    }


def print_summary(audit: Mapping[str, object], top_labels: int) -> None:
    print("Dataset: {}".format(audit["dataset"]))
    print("Metadata rows: {:,}".format(audit["metadata_row_count"]))
    print("Unique patients: {:,}".format(audit["unique_patient_count"]))
    print("Patients with multiple records: {:,}".format(audit["patients_with_multiple_records"]))
    print("Patients crossing folds: {}".format(len(audit["patients_in_multiple_folds"])))
    print("Fold distribution: {}".format(audit["fold_distribution"]))
    print("500 Hz header files: {:,}".format(audit["header_file_count"]))
    print("Parsed headers: {:,}".format(audit["parsed_header_count"]))
    print("Header parse errors: {}".format(len(audit["header_parse_errors"])))
    print("Structural anomalies: {}".format(len(audit["structural_anomalies"])))
    print("Missing waveform file pairs: {}".format(len(audit["metadata_records_with_missing_files"])))
    print("Canonical 12-lead records: {:,}".format(audit["canonical_12_lead_records"]))
    print("Unique SCP-ECG codes: {}".format(audit["unique_scp_code_count"]))
    print("Codes absent from statement map: {}".format(audit["codes_missing_statement_metadata"]))
    print("Top labels:")
    for label in audit["labels"][:top_labels]:
        print(
            "  {scp_code:>8}  {record_count:>6,}  {record_prevalence:>7.2%}  "
            "{description}".format(**label)
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Path to PTB-XL 1.0.3")
    parser.add_argument("--output-json", type=Path, help="Optional JSON report path")
    parser.add_argument("--top-labels", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
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

    has_errors = any(
        (
            audit["invalid_metadata_row_count"],
            audit["patients_in_multiple_folds"],
            audit["metadata_records_with_missing_files"],
            audit["header_parse_errors"],
            audit["structural_anomalies"],
        )
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
