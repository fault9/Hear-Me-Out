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
            model_relative = Path("sessions/test/model.wav")
            self._write_wav(root / model_relative)
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
                "files": {"participant_raw": str(relative),
                          "model": str(model_relative)},
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
                speaker = "Assistant" if "assistant" in path else "Participant"
                return {"text": f"{speaker} utterance {len(calls)}.",
                        "segments": [], "status": "complete", "error": None}

            result = prepare_dialogue_transcript(
                session, root, "analysis-1", timing, transcribe)

            self.assertEqual(result["schema"], DIALOGUE_TRANSCRIPT_SCHEMA)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(calls), 3)  # two participant + one assistant
            self.assertEqual([row["speaker"] for row in result["utterances"]],
                             ["participant", "assistant", "participant"])
            first = result["utterances"][0]
            self.assertEqual((first["start_ms"], first["end_ms"]), (1100.0, 1900.0))
            self.assertEqual(first["voice_mode"], "mixed")
            self.assertEqual([row["mode"] for row in first["route_segments"]],
                             ["natural", "vc"])
            self.assertEqual(first["text_provenance"]["wav_start_ms"], 800.0)
            # Assistant text now comes from ASR of its own audio, so it is
            # anchored to the interval rather than to the model's estimated
            # fragment times. The model's own text is kept for comparison.
            assistant = result["utterances"][1]
            self.assertTrue(assistant["text"].startswith("Assistant utterance"))
            provenance = assistant["text_provenance"]
            self.assertEqual(provenance["source"], "model.wav")
            self.assertEqual(provenance["browser_clock_origin_ms"], 0.0)
            self.assertEqual(provenance["model_text"], "Lead in. Assistant reply.")
            self.assertEqual(provenance["model_fragment_count"], 2)
            artifact = root / result["result_artifact"]["path"]
            self.assertTrue(artifact.exists())
            persisted = json.loads(artifact.read_text())
            self.assertNotIn("result_artifact", persisted)


    def test_assistant_text_follows_the_audio_not_the_model_clock(self):
        """The model merges text across its own pauses and back-computes a
        start from word count, so a fragment can sit seconds from the turn it
        belongs to. Placement must come from the audio."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("sessions/drift/participant_raw.wav")
            model_relative = Path("sessions/drift/model.wav")
            self._write_wav(root / relative, 40.0)
            self._write_wav(root / model_relative, 40.0)
            (root / relative).parent.joinpath("client_timeline.json").write_text(
                json.dumps({"capture": {"chunks": [
                    {"chunk_sequence": 1, "capture_start_sample": 0,
                     "sample_count": 2048, "timeline_start_ms": 0.0}]}}))
            session = {
                "session_id": "P1_R01_S02_A01",
                "voice_condition": "stable_natural",
                "schedule": [{"mode": "natural", "start_s": 0, "end_s": None}],
                "files": {"participant_raw": str(relative),
                          "model": str(model_relative)},
                # One merged fragment, timed where neither turn actually is.
                "transcript": {"model": [
                    {"text": "One moment please while I Okay, I see your case.",
                     "start": 29.04, "end": 32.74, "speaker": "personaplex"},
                ]},
            }
            timing = {
                "schema": "hmo.timing-analysis.v4",
                "status": "estimated_pending_validation",
                "participant_intervals": [
                    {"start_ms": 22800.0, "end_ms": 23800.0, "detector": "rms"}],
                "assistant_intervals": [
                    {"start_ms": 22900.0, "end_ms": 24000.0, "detector": "rms"},
                    {"start_ms": 32200.0, "end_ms": 38100.0, "detector": "rms"},
                ],
                "route_switches": [], "overlaps": [], "barge_ins": [],
                "integrity": {"participant_capture_latency_correction_ms": 0.0},
                "result_artifact": {"path": "analysis/timing/timing.json"},
            }

            def transcribe(path: str) -> dict:
                name = Path(path).stem
                return {"text": f"spoken at {name}", "segments": [],
                        "status": "complete", "error": None}

            result = prepare_dialogue_transcript(
                session, root, "analysis-2", timing, transcribe)

            assistants = [u for u in result["utterances"] if u["speaker"] == "assistant"]
            self.assertEqual(len(assistants), 2)
            # Both turns get their own audio's text: neither is left empty and
            # neither absorbs the other's words.
            self.assertTrue(all(u["text"] for u in assistants))
            self.assertNotEqual(assistants[0]["text"], assistants[1]["text"])
            self.assertEqual(result["summary"]["assistant_intervals_without_text"], 0)
            # The merged model fragment is still recorded, on one turn only.
            carried = [u["text_provenance"]["model_fragment_count"] for u in assistants]
            self.assertEqual(sorted(carried), [0, 1])


if __name__ == "__main__":
    unittest.main()
