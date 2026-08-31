import sys
import unittest
from pathlib import Path

import torch
from torch import nn


AI_DIR = Path(__file__).resolve().parents[1] / "ai"
sys.path.insert(0, str(AI_DIR))

from explain_rhythm_prediction import integrated_gradients, select_class_index  # noqa: E402


class SumModel(nn.Module):
    def forward(self, inputs):
        total = inputs.flatten(start_dim=1).sum(dim=1)
        return torch.stack((total, -total), dim=1)


class IntegratedGradientsTests(unittest.TestCase):
    def test_linear_model_attributions_satisfy_completeness(self):
        model = SumModel()
        inputs = torch.arange(24, dtype=torch.float32).reshape(1, 2, 12) / 10.0
        attributions = integrated_gradients(model, inputs, class_index=0, steps=4)
        self.assertEqual(tuple(attributions.shape), (2, 12))
        self.assertAlmostEqual(
            float(attributions.sum()),
            float(model(inputs)[0, 0] - model(torch.zeros_like(inputs))[0, 0]),
            places=4,
        )

    def test_class_selection_accepts_name_or_uses_maximum(self):
        probabilities = torch.tensor([0.2, 0.8]).numpy()
        self.assertEqual(select_class_index(("a", "b"), probabilities, "a"), 0)
        self.assertEqual(select_class_index(("a", "b"), probabilities, None), 1)


if __name__ == "__main__":
    unittest.main()
