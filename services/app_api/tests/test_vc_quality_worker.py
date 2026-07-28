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
                results = vc_quality_worker._score_batch(jobs)

        self.assertEqual([result["wer"] for result in results], [0.0, 0.1])
        self.assertEqual(results[0]["converted_path"], "b/converted.wav")
        self.assertEqual(results[1]["target_path"], "a/target.wav")


if __name__ == "__main__":
    unittest.main()
