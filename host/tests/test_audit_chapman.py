import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.audit_chapman import EXPECTED_LEADS, load_condition_map, parse_header  # noqa: E402


class ChapmanHeaderParserTests(unittest.TestCase):
    def test_parses_structure_and_metadata(self):
        header = "\n".join(
            ["TEST001 12 500 5000"]
            + [
                "TEST001.mat 16+24 1000/mV 16 0 0 0 0 {}".format(lead)
                for lead in EXPECTED_LEADS
            ]
            + ["#Age: 67", "#Sex: Female", "#Dx: 426783006,164889003"]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TEST001.hea"
            path.write_text(header, encoding="utf-8")
            result = parse_header(path)

        self.assertEqual(result["record_name"], "TEST001")
        self.assertEqual(result["signal_count"], 12)
        self.assertEqual(result["sampling_frequency_hz"], 500.0)
        self.assertEqual(result["sample_count"], 5000)
        self.assertEqual(result["leads"], EXPECTED_LEADS)
        self.assertEqual(result["diagnoses"], ("426783006", "164889003"))
        self.assertEqual(result["age"], "67")
        self.assertEqual(result["sex"], "Female")

    def test_rejects_invalid_first_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.hea"
            path.write_text("INVALID\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                parse_header(path)


class ConditionMapParserTests(unittest.TestCase):
    def test_loads_snomed_code_mapping(self):
        content = (
            "Acronym Name,Full Name,Snomed_CT\n"
            "AFIB,Atrial Fibrillation,164889003\n"
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditions.csv"
            path.write_text(content, encoding="utf-8")
            result = load_condition_map(path)

        self.assertEqual(result["164889003"]["acronym"], "AFIB")
        self.assertEqual(result["164889003"]["name"], "Atrial Fibrillation")


if __name__ == "__main__":
    unittest.main()
