import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from study.analysis_worker import _analysis_candidates, _latest_analysis_result
from study.session_scope import annotate_analysis_scope, annotate_analysis_scopes


class AnalysisWorkerTests(unittest.TestCase):
    def test_practice_is_excluded_even_from_forced_analysis(self):
        files = {"participant": "participant.wav"}
        practice = {"session_id": "practice", "voice_condition": "practice", "ended_at": 1,
                    "files": files}
        analytical = {"session_id": "analytical", "voice_condition": "stable_natural", "ended_at": 1,
                      "files": files}

        selected = _analysis_candidates([practice, analytical], force=True)

        self.assertEqual([session["session_id"] for session in selected],
                         ["analytical"])

    def test_admin_scope_marks_practice_as_ineligible(self):
        practice = annotate_analysis_scope({
            "voice_condition": "stable_natural",
            "config_snapshot": {"scenario": {"scenario_card": {
                "study_role": "practice",
            }}},
        })
        analytical = annotate_analysis_scope({"voice_condition": "vc_activation"})

        self.assertEqual(practice["study_role"], "practice")
        self.assertFalse(practice["analysis_eligible"])
        self.assertTrue(analytical["analysis_eligible"])

    def test_missing_audio_still_enters_queue_for_validity_report(self):
        selected = _analysis_candidates([{
            "session_id": "missing-audio",
            "voice_condition": "stable_natural",
            "ended_at": 1,
            "files": {},
        }], force=False)

        self.assertEqual([session["session_id"] for session in selected],
                         ["missing-audio"])

    def test_live_session_is_not_analyzed_while_capture_is_still_mutating(self):
        selected = _analysis_candidates([{
            "session_id": "live",
            "voice_condition": "stable_natural",
            "ended_at": None,
            "files": {"participant": "participant.wav"},
        }], force=True)

        self.assertEqual(selected, [])

    def test_current_dialogue_artifact_does_not_reenter_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = {
                "timing_latest": ("timing.json", "hmo.timing-analysis.v4", {}),
                "dialogue_transcript_latest": (
                    "dialogue.json", "hmo.dialogue-transcript.v1", {}),
                "technical_validity_latest": (
                    "validity.json", "hmo.technical-validity.v3", {"status": "valid"}),
            }
            manifest = {"analysis": {}}
            for key, (name, schema, extra) in artifacts.items():
                (root / name).write_text(json.dumps({"schema": schema, **extra}))
                manifest["analysis"][key] = {"path": name}
            session = {
                "session_id": "complete",
                "voice_condition": "stable_natural",
                "ended_at": 1,
                "files": {"participant": "participant.wav"},
                "metrics": {},
                "transcript": {
                    "schema": "hmo.study-transcript.v2",
                    "participant_segments": [],
                },
                "artifact_manifest": manifest,
            }

            with patch("study.analysis_worker.STUDY_DATA_DIR", root):
                selected = _analysis_candidates([session], force=False)

            self.assertEqual(selected, [])

    def test_loaded_analysis_retains_its_manifest_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "timing.json").write_text(json.dumps({
                "schema": "hmo.timing-analysis.v4",
            }))
            session = {"artifact_manifest": {"analysis": {
                "timing_latest": {
                    "path": "timing.json", "sha256": "abc", "size_bytes": 10,
                },
            }}}

            with patch("study.analysis_worker.STUDY_DATA_DIR", root):
                result = _latest_analysis_result(session, "timing_latest")

            self.assertEqual(result["result_artifact"]["path"], "timing.json")
            self.assertEqual(result["result_artifact"]["sha256"], "abc")

    def test_latest_valid_attempt_in_latest_submitted_run_is_included(self):
        def session(session_id, run_id, run_attempt, scenario_attempt, valid):
            return {
                "session_id": session_id,
                "participant_id": "P1",
                "run_id": run_id,
                "run_attempt": run_attempt,
                "scenario_order": 2,
                "scenario_attempt": scenario_attempt,
                "voice_condition": "stable_natural",
                "artifact_manifest": {"analysis": {
                    "technical_validity_summary": {
                        "status": "valid" if valid else "invalid",
                        "valid_for_condition_analysis": valid,
                        "failures": [] if valid else [{"code": "input_route_sequence"}],
                        "warnings": [],
                    },
                }},
            }

        annotated = annotate_analysis_scopes([
            session("old-run", 1, 1, 1, True),
            session("valid-attempt", 2, 2, 1, True),
            session("failed-retry", 2, 2, 2, False),
        ], [
            {"id": 1, "participant_id": "P1", "attempt": 1, "status": "submitted"},
            {"id": 2, "participant_id": "P1", "attempt": 2, "status": "submitted"},
        ])
        by_id = {row["session_id"]: row for row in annotated}

        self.assertTrue(by_id["valid-attempt"]["analysis_included"])
        self.assertTrue(by_id["valid-attempt"]["canonical_attempt"])
        self.assertFalse(by_id["failed-retry"]["analysis_included"])
        self.assertIn("technical:input_route_sequence",
                      by_id["failed-retry"]["analysis_exclusion_reasons"])
        self.assertEqual(by_id["old-run"]["analysis_exclusion_reasons"],
                         ["superseded_run"])


if __name__ == "__main__":
    unittest.main()
