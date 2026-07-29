import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from study import vc_quality_worker


class _CompletedProcess:
    returncode = 0

    def poll(self):
        return self.returncode


class VCQualityWorkerTests(unittest.TestCase):
    def test_score_batch_preserves_job_order_and_relative_paths(self):
        jobs = [
            {"converted": "b/converted.wav", "target": "b/target.wav",
             "source": "b/source.wav"},
            {"converted": "a/converted.wav", "target": "a/target.wav",
             "source": "a/source.wav"},
        ]

        def fake_popen(command, **_kwargs):
            manifest = Path(command[command.index("--manifest") + 1])
            output = Path(command[command.index("--out") + 1])
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            # Deliberately reverse output to exercise _job_index ordering.
            output.write_text("".join(
                json.dumps({**row, "wer": index / 10, "sim": 0.8,
                            "utmos": 3.5}) + "\n"
                for index, row in reversed(list(enumerate(rows)))
            ))
            return _CompletedProcess()

        with tempfile.TemporaryDirectory() as data_root:
            with patch.object(vc_quality_worker, "STUDY_DATA_DIR", Path(data_root)), \
                    patch.object(vc_quality_worker.subprocess, "Popen", fake_popen):
                results, diagnostics = vc_quality_worker._score_batch(jobs)

        self.assertEqual([result["wer"] for result in results], [0.0, 0.1])
        self.assertEqual(results[0]["converted_path"], "b/converted.wav")
        self.assertEqual(results[1]["target_path"], "a/target.wav")
        self.assertEqual(diagnostics, {"stdout": "", "stderr": ""})

    def test_missing_required_metrics_make_result_partial(self):
        status, unavailable = vc_quality_worker._completion([{
            "_region": 1,
            "wer": None,
            "wer_error": "ASR failed for: converted, reference",
            "sim": None,
            "sim_error": "SV checkpoint not found",
            "utmos": 2.4,
        }])

        self.assertEqual(status, "partial")
        self.assertEqual(
            [(item["region"], item["metric"]) for item in unavailable],
            [(1, "wer"), (1, "sim")],
        )

    def test_old_complete_profile_is_rescored(self):
        self.assertTrue(vc_quality_worker._needs_scoring({
            "vc_quality_status": "complete",
            "vc_quality": {"metric_profile": "xvc_objective_v1"},
        }))
        self.assertFalse(vc_quality_worker._needs_scoring({
            "vc_quality_status": "complete",
            "vc_quality": {"metric_profile": vc_quality_worker.METRIC_PROFILE},
        }))

    def test_natural_only_sessions_do_not_require_vc_quality(self):
        self.assertFalse(vc_quality_worker._session_has_vc_route({
            "voice_condition": "stable_natural",
            "schedule": [{"mode": "natural", "start_s": 0, "end_s": None}],
        }))
        self.assertTrue(vc_quality_worker._session_has_vc_route({
            "voice_condition": "vc_activation",
            "schedule": [
                {"mode": "natural", "start_s": 0, "end_s": 45},
                {"mode": "vc", "start_s": 45, "end_s": None},
            ],
        }))

    def test_practice_is_excluded_even_from_forced_scoring(self):
        files = {"participant": "participant.wav", "participant_raw": "raw.wav"}
        practice = {"session_id": "practice", "voice_condition": "practice", "ended_at": 1,
                    "files": files}
        analytical = {"session_id": "analytical", "voice_condition": "stable_natural", "ended_at": 1,
                      "files": files}

        selected = vc_quality_worker._scoring_candidates(
            [practice, analytical], force=True)

        self.assertEqual([session["session_id"] for session in selected],
                         ["analytical"])


if __name__ == "__main__":
    unittest.main()
