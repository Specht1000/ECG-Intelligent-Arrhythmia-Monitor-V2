import sys
import unittest
from pathlib import Path

import numpy as np


AI_DIR = Path(__file__).resolve().parents[1] / "ai"
sys.path.insert(0, str(AI_DIR))

from analyze_anomaly_errors import apply_temperature, fit_temperature  # noqa: E402
from analyze_rhythm_errors import expected_calibration_error  # noqa: E402


class RhythmCalibrationTests(unittest.TestCase):
    def test_perfect_binary_probabilities_have_zero_ece(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(expected_calibration_error(labels, probabilities), 0.0)

    def test_temperature_is_fitted_from_probabilities(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        probabilities = np.asarray([0.10, 0.20, 0.40, 0.60, 0.80, 0.90])
        temperature = fit_temperature(labels, probabilities)
        calibrated = apply_temperature(probabilities, temperature)
        self.assertGreater(temperature, 0.0)
        self.assertEqual(calibrated.shape, probabilities.shape)
        self.assertTrue(np.all(np.diff(calibrated) > 0))


if __name__ == "__main__":
    unittest.main()
