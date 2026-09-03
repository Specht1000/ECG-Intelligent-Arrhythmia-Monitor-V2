import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai"))

from advanced_rhythm_components import (  # noqa: E402
    RHYTHM_FEATURE_NAMES,
    AdvancedRhythmNet,
    AsymmetricMultiLabelLoss,
    assess_signal_quality,
    extract_rhythm_features,
)
from train_advanced_rhythm_classifier import constrained_threshold  # noqa: E402
from train_rhythm_classifier import CLASS_NAMES  # noqa: E402
from predict_advanced_rhythm import _load_numpy_record  # noqa: E402
from analyze_advanced_rhythm_errors import (  # noqa: E402
    _no_target_false_positive_rate,
    summarize_predictions,
)


class RhythmFeatureTests(unittest.TestCase):
    def test_extracts_finite_feature_vector(self):
        sampling_frequency = 100
        waveform = np.zeros((3, 1000), dtype=np.float32)
        for sample in range(50, 1000, 100):
            waveform[:, sample] = (1.0, 1.2, 0.2)
        features = extract_rhythm_features(
            waveform, sampling_frequency, ("I", "II", "III")
        )
        self.assertEqual(features.shape, (len(RHYTHM_FEATURE_NAMES),))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(features[RHYTHM_FEATURE_NAMES.index("detected_qrs_count")], 5)
        self.assertAlmostEqual(
            features[RHYTHM_FEATURE_NAMES.index("einthoven_rmse_mv")], 0.0, places=6
        )

    def test_flat_signal_is_rejected_by_research_quality_gate(self):
        assessment = assess_signal_quality(
            np.zeros((2, 1000), dtype=np.float32), 100, ("I", "II")
        )
        self.assertFalse(assessment.acceptable)
        self.assertIn("flat_or_disconnected_lead", assessment.reasons)


class AdvancedLossTests(unittest.TestCase):
    def test_correct_logits_have_lower_asymmetric_loss(self):
        loss = AsymmetricMultiLabelLoss()
        targets = torch.tensor([[1.0, 0.0]])
        correct = loss(torch.tensor([[5.0, -5.0]]), targets)
        incorrect = loss(torch.tensor([[-5.0, 5.0]]), targets)
        self.assertLess(float(correct), float(incorrect))


class AdvancedModelTests(unittest.TestCase):
    def test_model_combines_three_leads_and_features(self):
        model = AdvancedRhythmNet(3, len(CLASS_NAMES)).eval()
        with torch.no_grad():
            output = model(
                torch.zeros(2, 3, 1000),
                torch.zeros(2, len(RHYTHM_FEATURE_NAMES)),
            )
        self.assertEqual(tuple(output.shape), (2, len(CLASS_NAMES)))

    def test_model_accepts_two_independent_leads(self):
        model = AdvancedRhythmNet(2, len(CLASS_NAMES)).eval()
        with torch.no_grad():
            output = model(
                torch.zeros(1, 2, 2500),
                torch.zeros(1, len(RHYTHM_FEATURE_NAMES)),
            )
        self.assertEqual(tuple(output.shape), (1, len(CLASS_NAMES)))


class ConstrainedThresholdTests(unittest.TestCase):
    def test_selected_threshold_respects_minimum_specificity(self):
        labels = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
        probabilities = np.asarray([0.1, 0.2, 0.4, 0.8, 0.7, 0.9])
        threshold = constrained_threshold(labels, probabilities, minimum_specificity=0.75)
        predictions = probabilities >= threshold
        specificity = np.mean(~predictions[labels == 0])
        self.assertGreaterEqual(specificity, 0.75)


class AdvancedInputTests(unittest.TestCase):
    def test_two_independent_leads_reconstruct_lead_iii(self):
        waveform = np.stack(
            (
                np.linspace(-1.0, 1.0, 1000, dtype=np.float32),
                np.linspace(1.0, 3.0, 1000, dtype=np.float32),
            )
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "independent_leads.npy"
            np.save(path, waveform)
            loaded = _load_numpy_record(path, ("I", "II", "III"), 1000)
        self.assertEqual(loaded.shape, (3, 1000))
        np.testing.assert_allclose(loaded[2], loaded[1] - loaded[0])


class AdvancedErrorAnalysisTests(unittest.TestCase):
    def test_uses_calibrated_probabilities_and_counts_no_target_errors(self):
        frame = pd.DataFrame({"ecg_id": [1, 2]})
        for class_name in CLASS_NAMES:
            frame[class_name + "_reference"] = [0, 1]
            frame[class_name + "_probability"] = [0.9, 0.1]
            frame[class_name + "_calibrated_probability"] = [0.1, 0.9]
            frame[class_name + "_prediction"] = [0, 1]
        summaries, errors = summarize_predictions(frame, "synthetic")
        self.assertTrue((summaries["average_precision"] == 1.0).all())
        self.assertTrue(errors.empty)
        no_target_count, false_positive_rate = _no_target_false_positive_rate(frame)
        self.assertEqual(no_target_count, 1)
        self.assertEqual(false_positive_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
