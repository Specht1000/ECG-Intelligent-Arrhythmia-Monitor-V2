import sys
import unittest
from pathlib import Path

import numpy as np
import torch


AI_DIR = Path(__file__).resolve().parents[1] / "ai"
sys.path.insert(0, str(AI_DIR))

from train_anomaly_baseline import (  # noqa: E402
    AnomalyECGNet,
    anomaly_label,
    expected_calibration_error,
    select_balanced_accuracy_threshold,
)


class AnomalyLabelTests(unittest.TestCase):
    def test_norm_only_is_normal(self):
        self.assertEqual(anomaly_label({"NORM"}), 0)

    def test_any_supported_abnormal_class_is_abnormal(self):
        self.assertEqual(anomaly_label({"NORM", "CD"}), 1)
        self.assertEqual(anomaly_label({"MI"}), 1)

    def test_missing_diagnostic_class_is_excluded(self):
        self.assertIsNone(anomaly_label(set()))


class MetricTests(unittest.TestCase):
    def test_threshold_is_selected_from_validation_predictions(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.2, 0.7, 0.9])
        threshold = select_balanced_accuracy_threshold(labels, probabilities)
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.7)

    def test_perfect_calibration_has_zero_error(self):
        labels = np.asarray([0, 1])
        probabilities = np.asarray([0.0, 1.0])
        self.assertAlmostEqual(expected_calibration_error(labels, probabilities), 0.0)


class ModelTests(unittest.TestCase):
    def test_model_accepts_twelve_lead_ten_second_input(self):
        model = AnomalyECGNet().eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 12, 1000))
        self.assertEqual(tuple(output.shape), (2,))


if __name__ == "__main__":
    unittest.main()
