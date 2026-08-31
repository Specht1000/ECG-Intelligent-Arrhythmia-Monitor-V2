import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


AI_DIR = Path(__file__).resolve().parents[1] / "ai"
sys.path.insert(0, str(AI_DIR))

from analyze_anomaly_errors import (  # noqa: E402
    age_group,
    apply_temperature,
    fit_temperature,
    has_annotation,
    has_signal_quality_annotation,
)


class AgeGroupingTests(unittest.TestCase):
    def test_groups_regular_ages(self):
        self.assertEqual(age_group(39), "0-39")
        self.assertEqual(age_group(40), "40-59")
        self.assertEqual(age_group(79), "60-79")
        self.assertEqual(age_group(80), "80+ (includes privacy-coded 90+)")

    def test_privacy_coded_age_is_grouped_as_ninety_plus(self):
        self.assertEqual(age_group(300), "80+ (includes privacy-coded 90+)")


class SignalQualityTests(unittest.TestCase):
    def test_empty_annotations_are_not_quality_issues(self):
        self.assertFalse(has_annotation(np.nan))
        self.assertFalse(has_annotation(" , "))

    def test_any_quality_annotation_marks_the_record(self):
        row = pd.Series(
            {
                "baseline_drift": np.nan,
                "static_noise": ", V1,",
                "burst_noise": np.nan,
                "electrodes_problems": np.nan,
            }
        )
        self.assertTrue(has_signal_quality_annotation(row))


class TemperatureScalingTests(unittest.TestCase):
    def test_temperature_preserves_probability_order(self):
        probabilities = np.asarray([0.1, 0.4, 0.8])
        calibrated = apply_temperature(probabilities, 2.0)
        self.assertTrue(np.all(np.diff(calibrated) > 0))

    def test_fitted_temperature_is_positive(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.2, 0.3, 0.7, 0.8])
        self.assertGreater(fit_temperature(labels, probabilities), 0)


if __name__ == "__main__":
    unittest.main()
