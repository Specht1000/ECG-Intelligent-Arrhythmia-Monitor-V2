import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


AI_DIR = Path(__file__).resolve().parents[1] / "ai"
sys.path.insert(0, str(AI_DIR))

from train_rhythm_classifier import (  # noqa: E402
    CLASS_NAMES,
    RhythmECGNet,
    _parse_chapman_header,
    label_vector,
    select_f1_threshold,
)
from predict_rhythm import format_predictions  # noqa: E402


class RhythmLabelTests(unittest.TestCase):
    def test_label_vector_follows_canonical_class_order(self):
        vector = label_vector({"sinus_rhythm", "atrial_fibrillation"})
        expected = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(vector, expected)

    def test_threshold_separates_simple_validation_predictions(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
        threshold = select_f1_threshold(labels, probabilities)
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.8)


class ChapmanHeaderTests(unittest.TestCase):
    def test_parses_target_label_gain_and_lead_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / "JS99999.hea"
            signal_lines = []
            leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
            for lead in leads:
                signal_lines.append("JS99999.mat 16+24 1000/mV 16 0 0 0 0 {}".format(lead))
            header.write_text(
                "JS99999 12 500 5000\n"
                + "\n".join(signal_lines)
                + "\n#Age: 60\n#Sex: Female\n#Dx: 426783006,164889003\n",
                encoding="utf-8",
            )
            parsed = _parse_chapman_header(header, root)
            self.assertEqual(parsed["labels"], "atrial_fibrillation|sinus_rhythm")
            self.assertEqual(parsed["record_base"], "JS99999")

    def test_retains_record_without_target_label_as_background_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / "JS99998.hea"
            leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
            signal_lines = [
                "JS99998.mat 16+24 1000/mV 16 0 0 0 0 {}".format(lead)
                for lead in leads
            ]
            header.write_text(
                "JS99998 12 500 5000\n"
                + "\n".join(signal_lines)
                + "\n#Age: 60\n#Sex: Male\n#Dx: 284470004\n",
                encoding="utf-8",
            )
            parsed = _parse_chapman_header(header, root)
            self.assertEqual(parsed["labels"], "")
            self.assertEqual(sum(parsed[name] for name in CLASS_NAMES), 0)


class RhythmModelTests(unittest.TestCase):
    def test_model_produces_one_logit_per_class(self):
        model = RhythmECGNet().eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 12, 1000))
        self.assertEqual(tuple(output.shape), (2, len(CLASS_NAMES)))


class RhythmInferenceTests(unittest.TestCase):
    def test_formats_independent_multi_label_decisions(self):
        predictions = format_predictions(
            ("class_a", "class_b"),
            ("Class A", "Class B"),
            np.asarray([0.7, 0.4]),
            np.asarray([0.6, 0.5]),
        )
        self.assertTrue(predictions[0]["positive"])
        self.assertFalse(predictions[1]["positive"])


if __name__ == "__main__":
    unittest.main()
