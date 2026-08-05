from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.artifacts import file_record  # noqa: E402
from study.turn_reconstruction import (ReconstructionError, combine_verified,
                                       reconstruct)  # noqa: E402
from study.turn_verification import finalize  # noqa: E402


class TurnReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.session_dir = self.root / "media" / "sessions" / "study_1" / "P1"
        self.session_dir.mkdir(parents=True)
        self.raw = self.session_dir / "participant_raw.wav"
        self.transmitted = self.session_dir / "participant.wav"
        self.model = self.session_dir / "model.wav"
        self._wav(self.raw, 16000, 3.0)
        self._wav(self.transmitted, 16000, 2.0)
        self._wav(self.model, 24000, 4.0)
        self.client_path = self.session_dir / "client_timeline.json"
        self.events_path = self.session_dir / "events.jsonl"
        self.timing_path = self.session_dir / "analysis" / "timing.json"
        self.timing_path.parent.mkdir()
        self._write_client()
        self._write_events()
        self._write_timing()
        self.session = {
            "session_id": "P1_R01_S03_A01",
            "participant_id": "P1",
            "study_id": 1,
            "voice_condition": "stable_converted",
            "ended_at": 10.0,
            "files": {
                "participant": str(self.transmitted.relative_to(self.root)),
                "participant_raw": str(self.raw.relative_to(self.root)),
                "model": str(self.model.relative_to(self.root)),
            },
            "config_snapshot": {"study": {"settings": {
                "technical_validity": {
                    "max_capture_gap_total_ms": 50,
                    "max_capture_gap_ms": 10,
                },
            }}},
            "artifact_manifest": {
                "artifacts": {
                    "participant": file_record(
                        self.transmitted, relative_to=self.root),
                },
                "analysis": {
                    "timing_latest": file_record(
                        self.timing_path, relative_to=self.root),
                },
            },
        }

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _wav(path: Path, rate: int, duration: float) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(b"\x00\x00" * round(rate * duration))

    def _write_client(self) -> None:
        client = {
            "capture": {
                "sample_rate_hz": 16000,
                "chunks": [
                    {"chunk_sequence": sequence,
                     "timeline_start_ms": (sequence - 1) * 1000,
                     "capture_start_sample": (sequence - 1) * 16000,
                     "sample_count": 16000}
                    for sequence in (1, 2, 3)
                ],
            },
            "playback": {
                "assistant_packets": [
                    {"packet_sequence": sequence,
                     "timeline_start_ms": (sequence - 2) * 1000,
                     "timeline_end_ms": (sequence - 1) * 1000,
                     "decoded_samples": 24000,
                     "sample_rate_hz": 24000}
                    for sequence in (2, 3, 4, 5)
                ],
            },
        }
        self.client_path.write_text(json.dumps(client))

    def _events(self) -> list[dict]:
        return [
            {"event": "input_chunk", "chunk_sequence": 1,
             "browser_chunk_sequence": 1},
            {"event": "input_chunk", "chunk_sequence": 2,
             "browser_chunk_sequence": 2},
            {"event": "transmitted_window", "input_start_sample": 0,
             "input_end_sample": 16000, "output_sequence": 1},
            {"event": "transmitted_window", "input_start_sample": 16000,
             "input_end_sample": 32000, "output_sequence": 2},
            {"event": "route_activated", "to_mode": "vc", "input_sample": 0},
            {"event": "xvc_inference_batch"},
            *[
                {"event": "personaplex_output_packet", "tag": 1,
                 "packet_sequence": sequence}
                for sequence in (1, 2, 3, 4)
            ],
        ]

    def _write_events(self, rows: list[dict] | None = None) -> None:
        self.events_path.write_text("".join(
            json.dumps(row) + "\n" for row in (rows or self._events())))

    def _write_timing(self) -> None:
        timing = {
            "schema": "hmo.timing-analysis.v5",
            "status": "confirmed_for_candidate_nomination",
            "participant_intervals": [
                {"start_ms": 500, "end_ms": 1500, "detector": "rms"},
                {"start_ms": 2200, "end_ms": 2500, "detector": "rms"},
            ],
            "assistant_intervals": [
                {"start_ms": 0, "end_ms": 1000, "detector": "packet_rms"},
                {"start_ms": 1600, "end_ms": 1900, "detector": "packet_rms"},
                {"start_ms": 2500, "end_ms": 2900, "detector": "packet_rms"},
            ],
            "integrity": {
                "crosswalk_complete": False,
                "capture_gaps": {
                    "gap_count": 1, "total_gap_samples": 128,
                    "total_gap_ms": 8.0, "max_gap_samples": 128,
                    "max_gap_ms": 8.0,
                },
                "playback": {"queue_underrun_count": 1,
                             "queue_underrun_total_ms": 8.0},
            },
            "sources": {
                "participant_raw": file_record(self.raw, relative_to=self.root),
                "model": file_record(self.model, relative_to=self.root),
                "client_timeline": file_record(
                    self.client_path, relative_to=self.root),
                "proxy_events": file_record(
                    self.events_path, relative_to=self.root),
            },
        }
        self.timing_path.write_text(json.dumps(timing))

    @staticmethod
    def _edit(path: Path, callback) -> None:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        for row in rows:
            callback(row)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _complete_review(self, review: Path) -> None:
        def event(row):
            row["verified_overlap"] = "yes"
            row["verified_participant_barge_in"] = "yes"
            row["successful_assistant_yielding"] = "yes"
            row["verifier_initials"] = "AB"

        def gap(row):
            row["verified_positive_gap"] = "yes"
            row["verifier_initials"] = "AB"

        def session(row):
            row["full_session_reviewed"] = "yes"
            row["additional_event_count"] = "0"
            row["verifier_initials"] = "AB"

        self._edit(review / "turn_verification_queue.csv", event)
        self._edit(review / "turn_gap_verification_queue.csv", gap)
        self._edit(review / "turn_session_review_queue.csv", session)
        result = finalize(review)
        self.assertEqual(result["status"], "complete")

    def test_trailing_loss_builds_portable_review_bundle(self):
        review = self.root / "review"
        result = reconstruct(self.session, self.root, review)

        self.assertEqual(result["status"], "awaiting_manual_verification")
        self.assertEqual(result["candidate_events"], 1)
        self.assertEqual(result["candidate_gaps"], 1)
        self.assertEqual(result["certified_end_ms"], 2000.0)
        self.assertEqual(result["excluded_tail_ms"], 900.0)
        self.assertTrue(
            (review / "audio" / "participant_browser_clock.wav").is_file())
        artifacts = list((self.session_dir / "analysis" / "turn_reconstruction")
                         .glob("*/turn_reconstruction.json"))
        self.assertEqual(len(artifacts), 1)
        artifact = json.loads(artifacts[0].read_text())
        self.assertEqual(
            artifact["status"], "eligible_for_amended_turn_sensitivity")
        self.assertTrue(artifact["certification"][
            "certified_prefix_crosswalk_complete"])

        self._complete_review(review)
        with (review / "turn_session_summary_verified.csv").open() as handle:
            summary = list(csv.DictReader(handle))
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["participant_barge_in_count"], "1")

    def test_internal_capture_loss_is_rejected(self):
        rows = self._events()
        rows[1]["browser_chunk_sequence"] = 3
        self._write_events(rows)
        self._write_timing()
        self.session["artifact_manifest"]["analysis"]["timing_latest"] = \
            file_record(self.timing_path, relative_to=self.root)

        with self.assertRaisesRegex(ReconstructionError, "internal common-prefix"):
            reconstruct(self.session, self.root, self.root / "invalid-review")

    def test_combiner_labels_reconstructed_session_as_sensitivity(self):
        review = self.root / "review"
        reconstruct(self.session, self.root, review)
        self._complete_review(review)

        primary = self.root / "primary"
        primary.mkdir()
        (primary / "turn_verification_manifest.json").write_text(json.dumps({
            "status": "complete", "sessions_reviewed": 1,
        }))
        empty_tables = [
            "turn_events_adjudicated.csv", "turn_events_verified.csv",
            "turn_gaps_adjudicated.csv", "turn_gaps_verified.csv",
        ]
        for name in empty_tables:
            self._write_table(primary / name, ["session_id"], [])
        self._write_table(
            primary / "turn_session_summary_verified.csv",
            ["session_id", "participant_id", "condition"],
            [{"session_id": "PRIMARY", "participant_id": "P0",
              "condition": "stable_natural"}],
        )

        combined = self.root / "combined"
        result = combine_verified(primary, review, combined)
        self.assertEqual(result["status"], "complete")
        with (combined / "turn_session_summary_verified.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        scopes = {row["session_id"]: row["analysis_scope"] for row in rows}
        self.assertEqual(scopes["PRIMARY"], "primary_prespecified")
        self.assertEqual(
            scopes[self.session["session_id"]],
            "reconstructed_common_prefix_sensitivity")

    @staticmethod
    def _write_table(path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
