import unittest
from study.dataset_export import (GAP_VERIFICATION_SAMPLE, _gap_stratum,
                                  select_gap_verification_sample)


def _rows(count, condition="stable_natural", direction="participant_to_assistant",
          duration=800.0, prefix="P01001"):
    return [{"session_id": f"{prefix}_R01_S02_A01", "gap_id": f"gap_{i:04d}",
             "condition": condition, "direction": direction,
             "gap_duration_ms": duration} for i in range(count)]


class GapSampleTests(unittest.TestCase):
    def test_sample_is_the_requested_size_and_deterministic(self):
        rows = _rows(900) + _rows(900, condition="vc_activation", prefix="P01002")
        first = select_gap_verification_sample(rows)
        self.assertEqual(len(first), GAP_VERIFICATION_SAMPLE)
        self.assertEqual(set(first), set(select_gap_verification_sample(rows)))

    def test_every_stratum_is_represented(self):
        rows = _rows(500)
        rows += _rows(4, duration=100.0, prefix="P01002")   # rare stratum
        sample = select_gap_verification_sample(rows, size=50)
        self.assertIn("stable_natural|participant_to_assistant|under_200ms",
                      set(sample.values()))

    def test_allocation_follows_stratum_size(self):
        rows = _rows(800) + _rows(200, duration=100.0, prefix="P01002")
        sample = select_gap_verification_sample(rows, size=100)
        counts = {}
        for stratum in sample.values():
            counts[stratum] = counts.get(stratum, 0) + 1
        big = counts["stable_natural|participant_to_assistant|over_600ms"]
        small = counts["stable_natural|participant_to_assistant|under_200ms"]
        self.assertEqual(big + small, 100)
        self.assertAlmostEqual(big / 100, 0.8, delta=0.03)

    def test_a_smaller_population_than_the_target_is_taken_whole(self):
        rows = _rows(12)
        self.assertEqual(len(select_gap_verification_sample(rows)), 12)

    def test_stratum_separates_condition_direction_and_band(self):
        self.assertEqual(
            _gap_stratum({"condition": "vc_activation",
                          "direction": "assistant_to_participant",
                          "gap_duration_ms": 350.0}),
            "vc_activation|assistant_to_participant|200_600ms")


if __name__ == "__main__":
    unittest.main()
