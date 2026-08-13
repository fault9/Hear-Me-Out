from __future__ import annotations

import json
import tempfile
import unittest
import wave
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

    def test_new_pending_artifact_finalization_blocks_analysis_without_deleting_data(self):
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._fixture(Path(temp))
            session["artifact_manifest"]["finalization"] = {
                "schema": "hmo.session-finalization.v1",
                "status": "pending",
            }

            result = evaluate_technical_validity(session, Path(temp), timing)

        self.assertFalse(result["valid_for_condition_analysis"])
        self.assertIn(
            "artifact_finalization",
            {failure["code"] for failure in result["failures"]},
        )

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

    def _with_recording(self, root, session, held_s, assistant_end_s):
        """A participant WAV holding `held_s` of audio, against an assistant
        track still speaking at `assistant_end_s`."""
        import wave
        path = root / "participant_raw.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * int(16000 * held_s))
        session["files"] = {"participant_raw": path.name}
        return [{"start_ms": 0.0, "end_ms": assistant_end_s * 1000.0}]

    def test_uncorroborated_technical_end_reason_can_be_reclassified(self):
        # The button is the participant's report. Corroborated by the record
        # it excludes; uncorroborated and reclassified with recorded evidence,
        # it downgrades to a warning and the session stands.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session, timing = self._fixture(root)
            session["study_id"] = 1
            session["end_reason"] = "technical_problem"

            unlisted = evaluate_technical_validity(session, root, timing)

            review = root / "review" / "study1"
            review.mkdir(parents=True)
            (review / "end_reason_reclassifications.json").write_text(json.dumps({
                session["session_id"]: {
                    "recorded_at": "2026-08-13",
                    "evidence": "full-length session, no failure codes, "
                                "questionnaire contradicts the end reason",
                }}))
            listed = evaluate_technical_validity(session, root, timing)

        self.assertEqual(unlisted["status"], "invalid")
        self.assertIn("end_reason",
                      {f["code"] for f in unlisted["failures"]})
        self.assertEqual(listed["status"], "valid")
        self.assertIn("end_reason_reclassified",
                      {w["code"] for w in listed["warnings"]})

    def test_a_recording_missing_most_of_its_time_base_is_invalid(self):
        # P01010 held 106 s of audio for a 148 s conversation: the file plays
        # fast and nothing derived from it lines up with the assistant track.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session, timing = self._fixture(root)
            timing["assistant_intervals"] = self._with_recording(
                root, session, held_s=10.0, assistant_end_s=14.5)

            result = evaluate_technical_validity(session, root, timing)

        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["valid_for_condition_analysis"])
        self.assertIn("capture_time_base",
                      {failure["code"] for failure in result["failures"]})

    def test_a_recording_that_covers_its_session_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session, timing = self._fixture(root)
            # 8% short - real, but recoverable and not this check's business.
            timing["assistant_intervals"] = self._with_recording(
                root, session, held_s=10.0, assistant_end_s=10.8)

            result = evaluate_technical_validity(session, root, timing)

        self.assertNotIn("capture_time_base",
                         {failure["code"] for failure in result["failures"]})

    def test_long_single_capture_gap_preserves_condition_data_but_blocks_timing(self):
        # Half a second of missing microphone audio can hide a whole turn, so
        # no amount of listening recovers it.
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._fixture(Path(temp))
            timing["integrity"]["capture_gaps"] = {
                "gap_count": 2,
                "total_gap_ms": 512,
                "max_gap_ms": 504,
                "browser_reported_estimated_dropped_samples": 4096,
            }

            result = evaluate_technical_validity(session, Path(temp), timing)

        self.assertEqual(result["status"], "valid")
        self.assertTrue(result["valid_for_condition_analysis"])
        self.assertFalse(result["valid_for_timing_reconstruction"])
        self.assertIn("microphone_capture_drops",
                      {warning["code"] for warning in result["warnings"]})

    def test_accumulated_small_gaps_keep_manual_verification_but_not_confirmatory(self):
        # Many single blocks lost across a session leave silence in the right
        # place; a listener still hears where the turn starts, but the
        # automatic measurement keeps its frozen budget.
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._fixture(Path(temp))
            timing["status"] = BOUNDARY_CONFIRMATION_STATUS
            timing["integrity"]["capture_gaps"] = {
                "gap_count": 145,
                "total_gap_ms": 1160,
                "max_gap_ms": 8,
                "browser_reported_estimated_dropped_samples": 18560,
            }

            result = evaluate_technical_validity(session, Path(temp), timing)

        self.assertEqual(result["status"], "valid")
        self.assertTrue(result["valid_for_timing_reconstruction"])
        self.assertTrue(result["valid_for_manual_turn_verification"])
        self.assertFalse(result["valid_for_confirmatory_timing_analysis"])
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

    def _failed_delivery_fixture(self, root: Path, *, contiguous: bool):
        """No proxy artifacts, a truncated event stream, no crosswalk."""
        session, timing = self._fixture(root)
        artifacts = session["artifact_manifest"]["artifacts"]
        for name in ("proxy_received.wav", "participant_proxy.wav",
                     "personaplex_input.opus", "proxy_timeline"):
            artifacts.pop(name)
        integrity = timing["integrity"]
        integrity["crosswalk_complete"] = False
        integrity["valid_for_timing"] = False

        session_dir = root / "sessions" / "attempt"
        windows = [(0, 2048, 1), (2048, 4096, 2)] if contiguous else [(0, 2048, 1),
                                                                      (3072, 4096, 2)]
        events = [
            {"event": "stream_start", "event_sequence": 1},
            {"event": "route_activated", "event_sequence": 2,
             "from_mode": None, "to_mode": "vc", "input_sample": 0,
             "requested_start_s": 0},
            {"event": "input_chunk", "event_sequence": 3,
             "chunk_sequence": 1, "input_start_sample": 0, "input_end_sample": 2048},
            {"event": "input_chunk", "event_sequence": 4,
             "chunk_sequence": 2, "input_start_sample": 2048, "input_end_sample": 4096},
            {"event": "xvc_inference_batch", "event_sequence": 5,
             "input_start_sample": 0, "input_end_sample": 4096,
             "inference_windows": 1},
            {"event": "transmitted_route_activated", "event_sequence": 6,
             "from_mode": None, "to_mode": "vc", "transmitted_sample": 0},
        ] + [
            {"event": "transmitted_window", "event_sequence": 7 + index,
             "input_start_sample": start, "input_end_sample": end,
             "output_sequence": sequence}
            for index, (start, end, sequence) in enumerate(windows)
        ]
        event_path = session_dir / "events.jsonl"
        event_path.write_text("".join(json.dumps(row) + "\n" for row in events))
        artifacts["events"] = file_record(event_path, relative_to=root)

        with wave.open(str(session_dir / "participant.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 4096)
        artifacts["participant"] = file_record(
            session_dir / "participant.wav", relative_to=root)
        return session, timing

    def test_verified_continuity_downgrades_failed_delivery_to_warnings(self):
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._failed_delivery_fixture(
                Path(temp), contiguous=True)

            result = evaluate_technical_validity(session, Path(temp), timing)

        self.assertEqual(result["status"], "valid")
        self.assertTrue(result["valid_for_condition_analysis"])
        self.assertEqual(result["failures"], [])
        warned = {warning["code"] for warning in result["warnings"]}
        self.assertIn("proxy_artifacts_present", warned)
        self.assertIn("stream_stop", warned)
        self.assertIn("browser_proxy_crosswalk", warned)
        self.assertEqual(result["delivery_continuity"]["verdict"], "pass")
        # The prespecified synchronization gate still excludes timing measures.
        self.assertFalse(result["valid_for_timing_reconstruction"])
        self.assertFalse(result["valid_for_confirmatory_timing_analysis"])

    def test_verified_continuity_recovers_the_post_checkpoint_window(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session, timing = self._failed_delivery_fixture(root, contiguous=True)
            session["config_snapshot"]["study"]["settings"][
                "analysis_checkpoint_s"] = 0.1  # 1600 samples, inside the capture
            events = [json.loads(line) for line
                      in (root / "sessions" / "attempt" / "events.jsonl")
                      .read_text().splitlines()]
            next_sequence = max(row.get("event_sequence") or 0
                                for row in events) + 1
            events += [
                {"event": "analysis_checkpoint_requested",
                 "event_sequence": next_sequence,
                 "requested_input_sample": 1600},
                {"event": "analysis_checkpoint_reached",
                 "event_sequence": next_sequence + 1,
                 "requested_input_sample": 1600, "input_sample": 2048,
                 "route_mode": "vc"},
            ]
            (root / "sessions" / "attempt" / "events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events))
            session["artifact_manifest"]["artifacts"]["events"] = file_record(
                root / "sessions" / "attempt" / "events.jsonl", relative_to=root)

            result = evaluate_technical_validity(session, root, timing)

        # No stream_stop survived, so the capture's extent comes from its last
        # certified chunk instead of being read as "never reached the window".
        self.assertTrue(result["valid_for_condition_analysis"])
        self.assertTrue(result["valid_for_post_checkpoint_analysis"])

    def test_unverified_continuity_keeps_failed_delivery_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._failed_delivery_fixture(
                Path(temp), contiguous=False)

            result = evaluate_technical_validity(session, Path(temp), timing)

        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["valid_for_condition_analysis"])
        failed = {failure["code"] for failure in result["failures"]}
        self.assertIn("proxy_artifacts_present", failed)
        self.assertEqual(result["delivery_continuity"]["verdict"], "fail")

    def test_intact_delivery_keeps_missing_artifacts_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            session, timing = self._fixture(Path(temp))
            session["artifact_manifest"]["artifacts"].pop("participant_proxy.wav")

            result = evaluate_technical_validity(session, Path(temp), timing)

        self.assertEqual(result["status"], "invalid")
        self.assertIn("proxy_artifacts_present",
                      {failure["code"] for failure in result["failures"]})


if __name__ == "__main__":
    unittest.main()
