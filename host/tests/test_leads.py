import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_v2.leads import (  # noqa: E402
    BIPOLAR_LIMB_LEADS,
    STANDARD_12_LEADS,
    derive_limb_leads,
    reconstruct_bipolar_limb_leads,
    reconstruct_12_leads,
)


class LimbLeadDerivationTests(unittest.TestCase):
    def test_known_values(self):
        lead_iii, avr, avl, avf = derive_limb_leads(1.0, 2.0)

        self.assertEqual(lead_iii, 1.0)
        self.assertEqual(avr, -1.5)
        self.assertEqual(avl, 0.0)
        self.assertEqual(avf, 1.5)

    def test_einthoven_identity(self):
        lead_i = -127.25
        lead_ii = 431.75
        lead_iii, _, _, _ = derive_limb_leads(lead_i, lead_ii)

        self.assertAlmostEqual(lead_i + lead_iii, lead_ii)

    def test_augmented_leads_sum_to_zero(self):
        _, avr, avl, avf = derive_limb_leads(380.0, -75.0)

        self.assertAlmostEqual(avr + avl + avf, 0.0)

    def test_non_numeric_input_is_rejected(self):
        with self.assertRaises(TypeError):
            derive_limb_leads("1.0", 2.0)


class BipolarLeadReconstructionTests(unittest.TestCase):
    def test_returns_only_bipolar_leads_in_canonical_order(self):
        result = reconstruct_bipolar_limb_leads({"I": 100.0, "II": 250.0})
        self.assertEqual(tuple(result), BIPOLAR_LIMB_LEADS)
        self.assertEqual(result["III"], 150.0)

    def test_requires_both_independent_leads(self):
        with self.assertRaisesRegex(ValueError, "II"):
            reconstruct_bipolar_limb_leads({"I": 100.0})


class TwelveLeadReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "I": 100.0,
            "II": 250.0,
            "V1": -10.0,
            "V2": -20.0,
            "V3": 30.0,
            "V4": 40.0,
            "V5": 50.0,
            "V6": 60.0,
        }

    def test_canonical_order_and_derived_values(self):
        result = reconstruct_12_leads(self.base)

        self.assertEqual(tuple(result.keys()), STANDARD_12_LEADS)
        self.assertEqual(result["III"], 150.0)
        self.assertEqual(result["aVR"], -175.0)
        self.assertEqual(result["aVL"], -25.0)
        self.assertEqual(result["aVF"], 200.0)

    def test_precordial_leads_are_preserved(self):
        result = reconstruct_12_leads(self.base)

        for name in ("V1", "V2", "V3", "V4", "V5", "V6"):
            self.assertEqual(result[name], self.base[name])

    def test_missing_inputs_are_reported_together(self):
        incomplete = dict(self.base)
        del incomplete["II"]
        del incomplete["V4"]

        with self.assertRaisesRegex(ValueError, "II, V4"):
            reconstruct_12_leads(incomplete)

    def test_extra_metadata_is_ignored(self):
        sample_with_metadata = dict(self.base, sample_index=42, timestamp_us=168000)

        result = reconstruct_12_leads(sample_with_metadata)

        self.assertNotIn("sample_index", result)
        self.assertEqual(len(result), 12)


if __name__ == "__main__":
    unittest.main()
