import sys
import tempfile
import unittest
from pathlib import Path


VC_QUALITY_DIR = Path(__file__).resolve().parents[2] / "vc_quality"
if str(VC_QUALITY_DIR) not in sys.path:
    sys.path.insert(0, str(VC_QUALITY_DIR))

from pilot_calibration import diagnostic_profiles, select_vc_utterances  # noqa: E402
from target_screening import _candidate_wavs  # noqa: E402


class PilotCalibrationTests(unittest.TestCase):
    def test_diagnostics_use_only_the_frozen_production_stream(self):
        profiles = diagnostic_profiles()
        self.assertEqual(
            [profile["id"] for profile in profiles],
            ["observed", "offline", "production_stream"],
        )
        self.assertEqual(profiles[-1], {"id": "production_stream", "mode": "streaming"})

    def test_target_screening_discovers_only_wavs_recursively(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "speaker"
            nested.mkdir()
            (root / "a.wav").touch()
            (nested / "b.WAV").touch()
            (nested / "notes.txt").touch()

            self.assertEqual(
                [path.name for path in _candidate_wavs(root)],
                ["a.wav", "b.WAV"],
            )

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
