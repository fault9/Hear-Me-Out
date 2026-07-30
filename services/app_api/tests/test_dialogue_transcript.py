import json
import tempfile
import unittest
import wave
from pathlib import Path

from study.dialogue_transcript import (DIALOGUE_TRANSCRIPT_SCHEMA,
                                       prepare_dialogue_transcript)


class DialogueTranscriptTests(unittest.TestCase):
    def _write_wav(self, path: Path, duration_s: float = 5.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * round(duration_s * 16000))

    def test_uses_timing_boundaries_and_preserves_route_switches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("sessions/test/participant_raw.wav")
            raw_path = root / relative
            self._write_wav(raw_path)
            (raw_path.parent / "client_timeline.json").write_text(json.dumps({
                "capture": {"chunks": [{
                    "chunk_sequence": 1,
                    "capture_start_sample": 0,
                    "sample_count": 2048,
                    "timeline_start_ms": 100.0,
                }]},
            }))
            session = {
                "session_id": "P1_R01_S03_A01",
                "voice_condition": "vc_activation",
                "schedule": [
                    {"mode": "natural", "start_s": 0, "end_s": 1.5},
                    {"mode": "vc", "start_s": 1.5, "end_s": None},
                ],
                "files": {"participant_raw": str(relative)},
                "transcript": {"model": [
                    {"text": "Lead in.", "start": 1.85, "end": 1.95,
                     "speaker": "personaplex"},
                    {"text": "Assistant reply.", "start": 2.1, "end": 2.4,
                     "speaker": "personaplex"},
                ]},
            }
            timing = {
                "schema": "hmo.timing-analysis.v4",
                "status": "estimated_pending_validation",
                "participant_intervals": [
                    {"start_ms": 1100.0, "end_ms": 1900.0, "detector": "rms"},
                    {"start_ms": 3000.0, "end_ms": 3600.0, "detector": "rms"},
                ],
                "assistant_intervals": [{
                    "start_ms": 2000.0, "end_ms": 2500.0,
                    "detector": "decoded_packet_rms_audio_context",
                    "first_packet_sequence": 4, "last_packet_sequence": 8,
                }],
                "route_switches": [{
                    "from_mode": "natural", "to_mode": "vc",
                    "participant_timeline_ms": 1500.0,
                }],
                "overlaps": [], "barge_ins": [],
                "integrity": {"participant_capture_latency_correction_ms": 0.0},
                "result_artifact": {"path": "analysis/timing/timing.json"},
            }
            calls = []

            def transcribe(path: str) -> dict:
                calls.append(path)
                return {"text": f"Participant utterance {len(calls)}.",
                        "segments": [], "status": "complete", "error": None}

            result = prepare_dialogue_transcript(
                session, root, "analysis-1", timing, transcribe)

            self.assertEqual(result["schema"], DIALOGUE_TRANSCRIPT_SCHEMA)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(calls), 2)
            self.assertEqual([row["speaker"] for row in result["utterances"]],
                             ["participant", "assistant", "participant"])
            first = result["utterances"][0]
            self.assertEqual((first["start_ms"], first["end_ms"]), (1100.0, 1900.0))
            self.assertEqual(first["voice_mode"], "mixed")
            self.assertEqual([row["mode"] for row in first["route_segments"]],
                             ["natural", "vc"])
            self.assertEqual(first["text_provenance"]["wav_start_ms"], 800.0)
            assistant = result["utterances"][1]
            self.assertEqual(assistant["text"], "Lead in. Assistant reply.")
            self.assertEqual(
                assistant["text_provenance"]["fragments"][0]["assignment_method"],
                "nearest_interval_edge_within_2000ms",
            )
            self.assertEqual(
                assistant["text_provenance"]["fragments"][1]["assignment_method"],
                "maximum_temporal_overlap",
            )
            artifact = root / result["result_artifact"]["path"]
            self.assertTrue(artifact.exists())
            persisted = json.loads(artifact.read_text())
            self.assertNotIn("result_artifact", persisted)


if __name__ == "__main__":
    unittest.main()
