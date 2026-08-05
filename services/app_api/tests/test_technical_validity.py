from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from study.artifacts import file_record
from study.technical_validity import (BOUNDARY_CONFIRMATION_STATUS,
                                      evaluate_technical_validity)


class TechnicalValidityTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, dict]:
        session_dir = root / "sessions" / "attempt"
        session_dir.mkdir(parents=True)
        artifact_names = [
            "participant", "participant_raw", "model", "merged", "client_timeline", "target",
            "proxy_received.wav", "participant_proxy.wav", "personaplex_input.opus",
            "proxy_timeline",
        ]
        artifacts = {}
        for name in artifact_names:
            filename = name if "." in name else f"{name}.wav"
            path = session_dir / filename
            path.write_bytes(f"artifact:{name}".encode())
            artifacts[name] = file_record(path, relative_to=root)

        events = [
            {"event": "stream_start", "event_sequence": 1},
            {"event": "route_activated", "event_sequence": 2,
             "from_mode": None, "to_mode": "vc", "input_sample": 0,
             "requested_start_s": 0},
            {"event": "input_chunk", "event_sequence": 3,
             "chunk_sequence": 1, "input_start_sample": 0, "input_end_sample": 4096},
            {"event": "xvc_inference_batch", "event_sequence": 4,
             "input_start_sample": 0, "input_end_sample": 4096,
             "inference_windows": 1},
            {"event": "transmitted_route_activated", "event_sequence": 5,
             "from_mode": None, "to_mode": "vc", "transmitted_sample": 0},
            {"event": "stream_stop", "event_sequence": 6,
             "input_samples": 4096, "transmitted_samples": 1920},
        ]
        event_path = session_dir / "events.jsonl"
        event_path.write_text("".join(json.dumps(row) + "\n" for row in events))
        artifacts["events"] = file_record(event_path, relative_to=root)

        session = {
            "session_id": "S1",
            "ended_at": 10.0,
            "end_reason": "goal_reached",
            "schedule": [{"mode": "vc", "start_s": 0, "end_s": None}],
            "config_snapshot": {
                "engine": "xvc",
                "study": {"settings": {"technical_validity": {
                    "max_route_activation_lag_ms": 250,
                    "max_capture_gap_total_ms": 50,
                    "max_capture_gap_ms": 10,
                }}},
            },
            "artifact_manifest": {
                "artifacts": artifacts,
                "software": {
                    "hmo_commit": "hmo-test",
                    "xvc_commit": "xvc-test",
                    "personaplex_version": "pp-test",
                },
            },
        }
        timing = {
            "status": "estimated_pending_validation",
            "integrity": {
                "estimated_dropped_samples": 0,
                "capture_gaps": {
                    "gap_count": 0,
                    "total_gap_ms": 0,
                    "max_gap_ms": 0,
                    "browser_reported_estimated_dropped_samples": 0,
                },
                "crosswalk_complete": True,
                "valid_for_timing": True,
                "playback": {"queue_underrun_count": 0,
                             "queue_underrun_total_ms": 0,
                             "queue_underrun_max_ms": 0},
            },
        }
        return session, timing

    def test_complete_xvc_session_is_valid_but_timing_awaits_human_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._fixture(Path(temp))

            result = evaluate_technical_validity(session, Path(temp), timing)

        self.assertEqual(result["status"], "valid")
        self.assertTrue(result["valid_for_condition_analysis"])
        self.assertTrue(result["valid_for_timing_reconstruction"])
        self.assertFalse(result["valid_for_confirmatory_timing_analysis"])
        self.assertFalse(result["valid_for_manual_turn_verification"])
        self.assertEqual(result["failures"], [])

    def test_confirmed_boundary_audit_unlocks_manual_turn_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._fixture(Path(temp))
            timing["status"] = BOUNDARY_CONFIRMATION_STATUS

            result = evaluate_technical_validity(session, Path(temp), timing)

        self.assertTrue(result["valid_for_manual_turn_verification"])
        self.assertTrue(result["valid_for_confirmatory_timing_analysis"])
        self.assertEqual(
            result["speech_boundary_validation_status"],
            BOUNDARY_CONFIRMATION_STATUS,
        )

    def test_matched_analysis_checkpoint_is_a_separate_post_checkpoint_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session, timing = self._fixture(root)
            session["config_snapshot"]["study"]["settings"][
                "analysis_checkpoint_s"
            ] = 0.1
            event_path = root / session["artifact_manifest"]["artifacts"]["events"]["path"]
            rows = [json.loads(line) for line in event_path.read_text().splitlines()]
            rows.insert(1, {
                "event": "analysis_checkpoint_requested",
                "event_sequence": 2,
                "requested_start_s": 0.1,
                "requested_input_sample": 1600,
            })
            rows.insert(-1, {
                "event": "analysis_checkpoint_reached",
                "event_sequence": 7,
                "requested_start_s": 0.1,
                "requested_input_sample": 1600,
                "input_sample": 1664,
                "checkpoint_lag_ms": 4.0,
                "route_mode": "vc",
            })
            for sequence, row in enumerate(rows, start=1):
                row["event_sequence"] = sequence
            event_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            session["artifact_manifest"]["artifacts"]["events"] = file_record(
                event_path, relative_to=root)

            result = evaluate_technical_validity(session, root, timing)

        self.assertTrue(result["valid_for_condition_analysis"])
        self.assertTrue(result["valid_for_post_checkpoint_analysis"])
        self.assertEqual(result["analysis_checkpoint_s"], 0.1)
        self.assertTrue(result["checks"]["analysis_checkpoint_reached"]["passed"])

    def test_short_session_remains_valid_but_has_no_post_checkpoint_window(self):
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._fixture(Path(temp))
            session["config_snapshot"]["study"]["settings"][
                "analysis_checkpoint_s"
            ] = 45

            result = evaluate_technical_validity(session, Path(temp), timing)

        self.assertTrue(result["valid_for_condition_analysis"])
        self.assertFalse(result["valid_for_post_checkpoint_analysis"])
        self.assertFalse(result["checks"]["analysis_checkpoint_reached"]["passed"])
        self.assertNotIn(
            "analysis_checkpoint_reached",
            {warning["code"] for warning in result["warnings"]},
        )

    def test_inference_failure_invalidates_session_and_capture_gap_is_scoped_to_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session, timing = self._fixture(root)
            event_path = root / session["artifact_manifest"]["artifacts"]["events"]["path"]
            rows = [json.loads(line) for line in event_path.read_text().splitlines()]
            rows.insert(-1, {"event": "inference_failure", "event_sequence": 6})
            rows[-1]["event_sequence"] = 7
            event_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            session["artifact_manifest"]["artifacts"]["events"] = file_record(
                event_path, relative_to=root)
            timing["integrity"]["estimated_dropped_samples"] = 960
            timing["integrity"]["capture_gaps"] = {
                "gap_count": 2,
                "total_gap_ms": 60,
                "max_gap_ms": 12,
                "browser_reported_estimated_dropped_samples": 4096,
            }

            result = evaluate_technical_validity(session, root, timing)

        failure_codes = {failure["code"] for failure in result["failures"]}
        warning_codes = {warning["code"] for warning in result["warnings"]}
        self.assertEqual(result["status"], "invalid")
        self.assertIn("xvc_inference_failures", failure_codes)
        self.assertNotIn("microphone_capture_drops", failure_codes)
        self.assertIn("microphone_capture_drops", warning_codes)

    def test_large_capture_gap_preserves_condition_data_but_blocks_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._fixture(Path(temp))
            timing["integrity"]["capture_gaps"] = {
                "gap_count": 2,
                "total_gap_ms": 60,
                "max_gap_ms": 12,
                "browser_reported_estimated_dropped_samples": 4096,
            }

            result = evaluate_technical_validity(session, Path(temp), timing)

        self.assertEqual(result["status"], "valid")
        self.assertTrue(result["valid_for_condition_analysis"])
        self.assertFalse(result["valid_for_timing_reconstruction"])
        self.assertIn("microphone_capture_drops",
                      {warning["code"] for warning in result["warnings"]})

    def test_small_sample_boundary_gaps_pass_despite_legacy_overestimate(self):
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._fixture(Path(temp))
            timing["integrity"]["estimated_dropped_samples"] = 640
            timing["integrity"]["capture_gaps"] = {
                "gap_count": 5,
                "total_gap_ms": 40,
                "max_gap_ms": 8,
                "browser_reported_estimated_dropped_samples": 3200,
            }

            result = evaluate_technical_validity(session, Path(temp), timing)

        self.assertEqual(result["status"], "valid")
        self.assertTrue(result["valid_for_condition_analysis"])
        self.assertTrue(result["valid_for_timing_reconstruction"])

    def test_rerunnable_posthoc_failure_is_a_warning_not_session_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._fixture(Path(temp))

            result = evaluate_technical_validity(
                session, Path(temp), timing,
                {"preprocessing": "temporary Whisper model error"},
            )

        self.assertEqual(result["status"], "valid")
        self.assertTrue(result["valid_for_condition_analysis"])
        self.assertIn("preprocessing_stage",
                      {warning["code"] for warning in result["warnings"]})


if __name__ == "__main__":
    unittest.main()
