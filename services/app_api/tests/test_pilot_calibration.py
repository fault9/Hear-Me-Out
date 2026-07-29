import sys
import unittest
from pathlib import Path


VC_QUALITY_DIR = Path(__file__).resolve().parents[2] / "vc_quality"
if str(VC_QUALITY_DIR) not in sys.path:
    sys.path.insert(0, str(VC_QUALITY_DIR))

from pilot_calibration import parse_profiles, select_vc_utterances  # noqa: E402


class PilotCalibrationTests(unittest.TestCase):
    def test_profile_matrix_expands_each_stream_window_across_gates(self):
        profiles = parse_profiles(
            "observed,offline,stream120,stream200,stream320",
            "0.006,0.008",
        )
        self.assertEqual(
            [profile["id"] for profile in profiles],
            [
                "observed",
                "offline",
                "stream120_g0p006",
                "stream120_g0p008",
                "stream200_g0p006",
                "stream200_g0p008",
                "stream320_g0p006",
                "stream320_g0p008",
            ],
        )

    def test_fixed_latency_profiles_cross_smoothing_and_input_level(self):
        profiles = parse_profiles(
            "observed,stream120",
            "0.008",
            "20:100,40:80,60:60",
            "raw,normalized",
        )

        self.assertEqual(
            [profile["id"] for profile in profiles],
            [
                "observed",
                "stream120_g0p008",
                "stream120_g0p008_norm",
                "stream120_g0p008_s40_f80",
                "stream120_g0p008_s40_f80_norm",
                "stream120_g0p008_s60_f60",
                "stream120_g0p008_s60_f60_norm",
            ],
        )
        for profile in profiles[1:]:
            self.assertEqual(profile["smooth_ms"] + profile["future_ms"], 120)

    def test_fixed_latency_profiles_reject_larger_lookahead(self):
        with self.assertRaisesRegex(ValueError, "must remain 120 ms"):
            parse_profiles("stream120", "0.008", "40:100", "raw")

    def test_vc_utterances_use_guard_padding_and_route_mapping(self):
        events = [
            {"event": "route_activated", "event_sequence": 1,
             "to_mode": "natural", "input_sample": 0},
            {"event": "transmitted_window", "event_sequence": 2,
             "route_mode": "natural", "input_start_sample": 0,
             "input_end_sample": 16000, "transmitted_start_sample": 0,
             "transmitted_end_sample": 16000},
            {"event": "route_activated", "event_sequence": 3,
             "to_mode": "vc", "input_sample": 16000},
            {"event": "transmitted_window", "event_sequence": 4,
             "route_mode": "vc", "input_start_sample": 16000,
             "input_end_sample": 48000, "transmitted_start_sample": 32000,
             "transmitted_end_sample": 64000},
            {"event": "participant_speech_start", "event_sequence": 5,
             "input_sample": 23000},
            {"event": "participant_speech_end", "event_sequence": 6,
             "input_sample": 31000},
            {"event": "stream_stop", "event_sequence": 7,
             "input_samples": 48000},
        ]

        utterances = select_vc_utterances(
            events, guard_s=0.5, padding_s=0.2, min_utterance_s=0.4,
        )

        self.assertEqual(len(utterances), 1)
        row = utterances[0]
        # The VC region is guarded to [24000, 40000]. Speech crossing its
        # leading guard is clipped before padding and duration filtering.
        self.assertEqual(row["speech_start_sample"], 24000)
        self.assertEqual(row["speech_end_sample"], 31000)
        self.assertEqual(row["source_start_sample"], 24000)
        self.assertEqual(row["source_end_sample"], 34200)
        # The transmitted route starts one second later, and the route-local
        # mapping retains each interval's relative position within that span.
        self.assertEqual(row["observed_start_sample"], 40000)
        self.assertEqual(row["observed_end_sample"], 50200)

    def test_natural_route_speech_is_not_selected(self):
        events = [
            {"event": "route_activated", "event_sequence": 1,
             "to_mode": "natural", "input_sample": 0},
            {"event": "transmitted_window", "event_sequence": 2,
             "route_mode": "natural", "input_start_sample": 0,
             "input_end_sample": 32000, "transmitted_start_sample": 0,
             "transmitted_end_sample": 32000},
            {"event": "participant_speech_start", "event_sequence": 3,
             "input_sample": 8000},
            {"event": "participant_speech_end", "event_sequence": 4,
             "input_sample": 24000},
            {"event": "stream_stop", "event_sequence": 5,
             "input_samples": 32000},
        ]

        self.assertEqual(
            select_vc_utterances(
                events, guard_s=0.5, padding_s=0.2, min_utterance_s=0.4,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
