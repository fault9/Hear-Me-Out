from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from study.artifacts import file_record
from study.technical_validity import evaluate_technical_validity


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
                    "max_estimated_dropped_samples": 0,
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
        self.assertEqual(result["failures"], [])

    def test_inference_failure_and_dropped_samples_invalidate_session(self):
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
            timing["integrity"]["estimated_dropped_samples"] = 80

            result = evaluate_technical_validity(session, root, timing)

        codes = {failure["code"] for failure in result["failures"]}
        self.assertEqual(result["status"], "invalid")
        self.assertIn("xvc_inference_failures", codes)
        self.assertIn("microphone_capture_drops", codes)

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
