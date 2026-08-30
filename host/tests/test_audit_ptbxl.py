import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai"))

from audit_ptbxl import load_scp_statements, normalize_leads, parse_scp_codes  # noqa: E402


class ScpCodesParserTests(unittest.TestCase):
    def test_parses_ptbxl_dictionary(self):
        result = parse_scp_codes("{'NORM': 100.0, 'SR': 0.0}")

        self.assertEqual(result, {"NORM": 100.0, "SR": 0.0})

    def test_rejects_non_dictionary_value(self):
        with self.assertRaises(ValueError):
            parse_scp_codes("['NORM']")


class LeadNormalizationTests(unittest.TestCase):
    def test_normalizes_augmented_limb_lead_names(self):
        result = normalize_leads(("I", "II", "III", "AVR", "AVL", "AVF", "V1"))

        self.assertEqual(result, ("I", "II", "III", "aVR", "aVL", "aVF", "V1"))


class ScpStatementParserTests(unittest.TestCase):
    def test_loads_unnamed_code_column(self):
        content = (
            ",description,diagnostic,form,rhythm\n"
            "NORM,normal ECG,1.0,,\n"
            "SR,sinus rhythm,,,1.0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scp_statements.csv"
            path.write_text(content, encoding="utf-8")
            statements = load_scp_statements(path)

        self.assertEqual(statements["NORM"]["description"], "normal ECG")
        self.assertEqual(statements["SR"]["rhythm"], "1.0")


if __name__ == "__main__":
    unittest.main()
