from __future__ import annotations

import unittest

from study.turn_taking import (build_positive_response_gaps,
                               build_turn_episodes,
                               turn_events_from_timing)


class TurnEpisodeTests(unittest.TestCase):
    def test_participant_onset_during_assistant_is_barge_in(self):
        episodes = build_turn_episodes(
            [{"start_ms": 500, "end_ms": 1300}],
            [{"start_ms": 0, "end_ms": 900}],
        )

        self.assertEqual(len(episodes), 1)
        event = episodes[0]
        self.assertEqual(event["initiator"], "participant")
        self.assertTrue(event["overlap_200ms_candidate"])
        self.assertTrue(event["participant_barge_in_candidate"])
        self.assertFalse(event["assistant_premature_onset_candidate"])
        self.assertEqual(event["assistant_stop_latency_ms"], 400)
        self.assertIsNone(event["participant_stop_latency_ms"])

    def test_assistant_onset_during_participant_is_premature_onset(self):
        episodes = build_turn_episodes(
            [{"start_ms": 0, "end_ms": 900}],
            [{"start_ms": 500, "end_ms": 1300}],
        )

        event = episodes[0]
        self.assertEqual(event["initiator"], "assistant")
        self.assertFalse(event["participant_barge_in_candidate"])
        self.assertTrue(event["assistant_premature_onset_candidate"])
        self.assertEqual(event["participant_stop_latency_ms"], 400)
        self.assertIsNone(event["assistant_stop_latency_ms"])

    def test_equal_onsets_are_not_assigned_a_direction(self):
        event = build_turn_episodes(
            [{"start_ms": 100, "end_ms": 500}],
            [{"start_ms": 100, "end_ms": 600}],
        )[0]

        self.assertEqual(event["initiator"], "simultaneous")
        self.assertFalse(event["participant_barge_in_candidate"])
        self.assertFalse(event["assistant_premature_onset_candidate"])

    def test_short_directional_overlap_remains_barge_candidate(self):
        event = build_turn_episodes(
            [{"start_ms": 850, "end_ms": 1100}],
            [{"start_ms": 0, "end_ms": 900}],
        )[0]

        self.assertEqual(event["overlap_duration_ms"], 50)
        self.assertFalse(event["overlap_200ms_candidate"])
        self.assertTrue(event["participant_barge_in_candidate"])


class PositiveGapTests(unittest.TestCase):
    def test_positive_gaps_are_directional(self):
        gaps = build_positive_response_gaps(
            [
                {"start_ms": 0, "end_ms": 500},
                {"start_ms": 2000, "end_ms": 2500},
            ],
            [{"start_ms": 800, "end_ms": 1500}],
        )

        self.assertEqual(
            [row["direction"] for row in gaps],
            ["participant_to_assistant", "assistant_to_participant"],
        )
        self.assertEqual([row["gap_duration_ms"] for row in gaps], [300, 500])

    def test_latest_same_speaker_interval_prevents_false_response_gap(self):
        gaps = build_positive_response_gaps(
            [
                {"start_ms": 0, "end_ms": 400},
                {"start_ms": 600, "end_ms": 900},
            ],
            [{"start_ms": 1200, "end_ms": 1500}],
        )

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["from_interval"], 1)
        self.assertEqual(gaps[0]["gap_duration_ms"], 300)

    def test_tied_preceding_speakers_are_omitted(self):
        gaps = build_positive_response_gaps(
            [{"start_ms": 0, "end_ms": 500}],
            [
                {"start_ms": 100, "end_ms": 500},
                {"start_ms": 800, "end_ms": 1000},
            ],
        )

        self.assertEqual(gaps, [])

    def test_ongoing_speech_prevents_false_positive_gap(self):
        gaps = build_positive_response_gaps(
            [
                {"start_ms": 0, "end_ms": 400},
                {"start_ms": 900, "end_ms": 1300},
            ],
            [
                {"start_ms": 500, "end_ms": 1100},
            ],
        )

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["direction"], "participant_to_assistant")
        self.assertEqual(gaps[0]["gap_duration_ms"], 100)


class CompatibilityTests(unittest.TestCase):
    def test_v4_intervals_are_reconstructed_directionally(self):
        episodes, gaps = turn_events_from_timing({
            "schema": "hmo.timing-analysis.v4",
            "participant_intervals": [
                {"start_ms": 500, "end_ms": 1300},
                {"start_ms": 2000, "end_ms": 2400},
            ],
            "assistant_intervals": [
                {"start_ms": 0, "end_ms": 900},
                {"start_ms": 1600, "end_ms": 1800},
            ],
        })

        self.assertEqual(len(episodes), 1)
        self.assertTrue(episodes[0]["participant_barge_in_candidate"])
        self.assertEqual(
            [row["direction"] for row in gaps],
            ["participant_to_assistant", "assistant_to_participant"],
        )

    def test_legacy_rows_without_intervals_remain_exportable(self):
        episodes, gaps = turn_events_from_timing({
            "schema": "hmo.timing-analysis.v4",
            "participant_intervals": [],
            "assistant_intervals": [],
            "overlaps": [{
                "start_ms": 100, "end_ms": 500, "duration_ms": 400,
            }],
            "barge_ins": [{
                "start_ms": 900, "end_ms": 1100, "stop_latency_ms": 200,
            }],
        })

        self.assertEqual(len(episodes), 2)
        self.assertEqual(sum(row["overlap_200ms_candidate"] for row in episodes), 1)
        self.assertEqual(
            sum(row["participant_barge_in_candidate"] for row in episodes), 1)
        self.assertEqual(gaps, [])

    def test_matching_legacy_overlap_and_barge_rows_are_linked_once(self):
        episodes, _ = turn_events_from_timing({
            "participant_intervals": [],
            "assistant_intervals": [],
            "overlaps": [{
                "participant_interval": 2,
                "assistant_interval": 4,
                "start_ms": 100,
                "end_ms": 500,
                "duration_ms": 400,
            }],
            "barge_ins": [{
                "participant_interval": 2,
                "assistant_interval": 4,
                "participant_onset_ms": 100,
                "assistant_stop_ms": 500,
                "stop_latency_ms": 400,
            }],
        })

        self.assertEqual(len(episodes), 1)
        self.assertTrue(episodes[0]["overlap_200ms_candidate"])
        self.assertTrue(episodes[0]["participant_barge_in_candidate"])
        self.assertEqual(episodes[0]["participant_onset_ms"], 100)
        self.assertEqual(episodes[0]["assistant_offset_ms"], 500)


if __name__ == "__main__":
    unittest.main()
