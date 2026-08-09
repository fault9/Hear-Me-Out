import json
import tempfile
import unittest
import wave
from pathlib import Path

from study.dialogue_transcript import (DIALOGUE_TRANSCRIPT_SCHEMA,
                                       capture_scale, capture_time_map,
                                       prepare_dialogue_transcript)


class DialogueTranscriptTests(unittest.TestCase):
    @staticmethod
    def _no_slices(_path: str, windows) -> list[list[dict]]:
        """Whatever these tests leave empty, they are not testing the reread."""
        return [[] for _ in windows]

    def _write_wav(self, path: Path, duration_s: float = 5.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * round(duration_s * 16000))

    def test_short_capture_is_stretched_and_a_covering_one_is_not(self):
        """A recording that holds fewer seconds than it spans places every
        later participant event early; one that covers the session must be
        left exactly alone."""
        path = Path(tempfile.mkdtemp()) / "participant_raw.wav"
        self._write_wav(path, duration_s=10.0)
        self.assertAlmostEqual(
            capture_scale(path, [{"end_ms": 14000.0}]), 1.4, places=3)
        self.assertEqual(capture_scale(path, [{"end_ms": 10100.0}]), 1.0)
        self.assertEqual(capture_scale(path, []), 1.0)

    def test_file_position_is_read_off_the_chunk_that_holds_it(self):
        """The capture timeline says where each chunk's samples landed and
        when they arrived, so the mapping is looked up rather than modelled."""
        session_dir = Path(tempfile.mkdtemp())
        # 128 ms of audio delivered every 192 ms: the graph runs behind.
        (session_dir / "client_timeline.json").write_text(json.dumps({
            "capture": {"sample_rate_hz": 16000, "chunks": [
                {"chunk_sequence": i + 1, "capture_start_sample": i * 2048,
                 "sample_count": 2048, "timeline_start_ms": i * 192.0}
                for i in range(10)]}}))
        mapped = capture_time_map(session_dir, 1.5, 0.0)
        self.assertAlmostEqual(mapped(0.0), 0.0, places=6)
        self.assertAlmostEqual(mapped(128.0), 192.0, places=6)
        self.assertAlmostEqual(mapped(1152.0), 1728.0, places=6)
        self.assertAlmostEqual(mapped(160.0), 224.0, places=6)
        self.assertEqual(capture_time_map(session_dir, 1.0, 0.0)(7777.0), 7777.0)

    def test_unreadable_capture_timeline_falls_back_to_a_stretch(self):
        empty = Path(tempfile.mkdtemp())
        self.assertAlmostEqual(capture_time_map(empty, 1.5, 0.0)(1000.0), 1500.0,
                               places=6)

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
                return {"status": "complete", "error": None, "words": [
                    {"word": "Building", "start": 1.2, "end": 1.5},
                    {"word": "sixteen.", "start": 1.6, "end": 1.85},
                    {"word": "Blue", "start": 3.1, "end": 3.3},
                    {"word": "door.", "start": 3.35, "end": 3.55},
                ]}

            result = prepare_dialogue_transcript(
                session, root, "analysis-1", timing, transcribe)

            self.assertEqual(result["schema"], DIALOGUE_TRANSCRIPT_SCHEMA)
            self.assertEqual(result["status"], "complete")
            # One pass over the capture, not one per interval.
            self.assertEqual(len(calls), 1)
            self.assertEqual([row["speaker"] for row in result["utterances"]],
                             ["participant", "assistant"])
            # Both fragments are one spoken turn: nothing separates them.
            self.assertEqual(result["utterances"][1]["text"],
                             "Lead in. Assistant reply.")
            self.assertEqual(
                result["utterances"][1]["text_provenance"]["fragment_count"], 2)
            # The participant's two intervals are 1.1 s apart, so they are one
            # turn carrying both intervals' words.
            first = result["utterances"][0]
            self.assertEqual((first["start_ms"], first["end_ms"]), (1100.0, 3600.0))
            self.assertEqual(first["timing"]["intervals"], [0, 1])
            self.assertEqual(first["text"], "Building sixteen. Blue door.")
            self.assertEqual(first["voice_mode"], "mixed")
            self.assertEqual([row["mode"] for row in first["route_segments"]],
                             ["natural", "vc"])
            artifact = root / result["result_artifact"]["path"]
            self.assertTrue(artifact.exists())
            persisted = json.loads(artifact.read_text())
            self.assertNotIn("result_artifact", persisted)


    def test_model_text_is_carried_verbatim_and_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("sessions/paused/participant_raw.wav")
            self._write_wav(root / relative, 40.0)
            (root / relative).parent.joinpath("client_timeline.json").write_text(
                json.dumps({"capture": {"chunks": [
                    {"chunk_sequence": 1, "capture_start_sample": 0,
                     "sample_count": 2048, "timeline_start_ms": 0.0}]}}))
            session = {
                "session_id": "P1_R01_S02_A01",
                "voice_condition": "stable_natural",
                "schedule": [{"mode": "natural", "start_s": 0, "end_s": None}],
                "files": {"participant_raw": str(relative)},
                # The model pauses mid-utterance, so one fragment spans two of
                # its speech intervals and the next is timed before it.
                "transcript": {"model": [
                    {"text": "One moment please while I Okay, I see your case.",
                     "start": 29.04, "end": 32.74, "speaker": "personaplex"},
                    {"text": "Your claim was declined.", "start": 23.10,
                     "end": 38.90, "speaker": "personaplex"},
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

            def transcribe(_path: str) -> dict:
                return {"text": "Participant utterance.", "segments": [],
                        "status": "complete", "error": None}

            result = prepare_dialogue_transcript(
                session, root, "analysis-2", timing, transcribe, self._no_slices)

            # Nothing separates the two fragments, so they are one turn, and
            # every word the model emitted survives exactly once.
            assistants = [row for row in result["utterances"]
                          if row["speaker"] == "assistant"]
            self.assertEqual([row["text"] for row in assistants], [
                "One moment please while I Okay, I see your case."
                " Your claim was declined.",
            ])
            self.assertEqual(result["summary"]["unassigned_model_fragments"], 0)
            self.assertEqual(result["summary"]["model_fragments"], 2)

    def test_participant_speech_closes_an_assistant_turn(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("sessions/split/participant_raw.wav")
            self._write_wav(root / relative, 40.0)
            (root / relative).parent.joinpath("client_timeline.json").write_text(
                json.dumps({"capture": {"chunks": [
                    {"chunk_sequence": 1, "capture_start_sample": 0,
                     "sample_count": 2048, "timeline_start_ms": 0.0}]}}))
            session = {
                "session_id": "P1_R01_S02_A01",
                "voice_condition": "stable_natural",
                "schedule": [{"mode": "natural", "start_s": 0, "end_s": None}],
                "files": {"participant_raw": str(relative)},
                "transcript": {"model": [
                    {"text": "Got it.", "start": 10.1, "end": 10.9,
                     "speaker": "personaplex"},
                    {"text": "Let me check.", "start": 10.9, "end": 12.9,
                     "speaker": "personaplex"},
                    {"text": "Your case is about headphones.", "start": 30.0,
                     "end": 32.0, "speaker": "personaplex"},
                ]},
            }
            timing = {
                "schema": "hmo.timing-analysis.v4",
                "status": "estimated_pending_validation",
                "participant_intervals": [
                    {"start_ms": 20000.0, "end_ms": 22000.0, "detector": "rms"}],
                "assistant_intervals": [
                    {"start_ms": 10100.0, "end_ms": 12900.0, "detector": "rms"},
                    {"start_ms": 30000.0, "end_ms": 32000.0, "detector": "rms"}],
                "route_switches": [], "overlaps": [], "barge_ins": [],
                "integrity": {"participant_capture_latency_correction_ms": 0.0},
                "result_artifact": {"path": "analysis/timing/timing.json"},
            }

            def transcribe(_path: str) -> dict:
                return {"text": "Participant utterance.", "segments": [],
                        "status": "complete", "error": None}

            result = prepare_dialogue_transcript(
                session, root, "analysis-3", timing, transcribe, self._no_slices)

            self.assertEqual([row["speaker"] for row in result["utterances"]],
                             ["assistant", "participant", "assistant"])
            self.assertEqual([row["id"] for row in result["utterances"]],
                             ["assistant_001", "participant_001", "assistant_002"])
            self.assertEqual(result["utterances"][0]["text"],
                             "Got it. Let me check.")

    def test_an_answer_orders_before_the_reply_it_prompted(self):
        """P01001_R01_S02_A01: the model asks at 69-73s, the participant
        answers "Yes." at 74.9s, the model replies with text arriving at 76.9s
        whose back-computed start is 74.7s. Sorting on that start put the
        answer after the reply."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("sessions/order/participant_raw.wav")
            self._write_wav(root / relative, 90.0)
            (root / relative).parent.joinpath("client_timeline.json").write_text(
                json.dumps({"capture": {"chunks": [
                    {"chunk_sequence": 1, "capture_start_sample": 0,
                     "sample_count": 2048, "timeline_start_ms": 0.0}]}}))
            session = {
                "session_id": "P1_R01_S02_A01",
                "voice_condition": "stable_natural",
                "schedule": [{"mode": "natural", "start_s": 0, "end_s": None}],
                "files": {"participant_raw": str(relative)},
                "transcript": {"model": [
                    {"text": "Is that what you saying?", "start": 69.4,
                     "end": 73.4, "speaker": "personaplex"},
                    {"text": "Alright, now what did you do?", "start": 74.7,
                     "end": 76.9, "speaker": "personaplex"},
                ]},
            }
            timing = {
                "schema": "hmo.timing-analysis.v4",
                "status": "estimated_pending_validation",
                "participant_intervals": [
                    {"start_ms": 74900.0, "end_ms": 75200.0, "detector": "rms"}],
                # The model is audible 69.4-73.4 and again from 75.6.
                "assistant_intervals": [
                    {"start_ms": 69400.0, "end_ms": 73400.0, "detector": "rms"},
                    {"start_ms": 75600.0, "end_ms": 76900.0, "detector": "rms"},
                ],
                "route_switches": [], "overlaps": [], "barge_ins": [],
                "integrity": {"participant_capture_latency_correction_ms": 0.0},
                "result_artifact": {"path": "analysis/timing/timing.json"},
            }

            def transcribe(_path: str) -> dict:
                return {"text": "Yes.", "segments": [],
                        "status": "complete", "error": None}

            result = prepare_dialogue_transcript(
                session, root, "analysis-5", timing, transcribe)

            self.assertEqual([row["speaker"] for row in result["utterances"]],
                             ["assistant", "participant", "assistant"])
            reply = result["utterances"][2]
            self.assertEqual(reply["start_ms"], 75600.0)
            self.assertEqual(reply["timing"]["start_anchor"], "audible_run_onset")

    def test_words_land_on_their_own_interval_and_blips_stay_empty(self):
        """A sub-speech blip gets no words, so it carries no text. Slicing it
        and transcribing the slice is what produced "thank you." on 127 of the
        577 participant intervals under 600 ms."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("sessions/words/participant_raw.wav")
            self._write_wav(root / relative, 20.0)
            (root / relative).parent.joinpath("client_timeline.json").write_text(
                json.dumps({"capture": {"chunks": [
                    {"chunk_sequence": 1, "capture_start_sample": 0,
                     "sample_count": 2048, "timeline_start_ms": 0.0}]}}))
            session = {
                "session_id": "P1_R01_S02_A01",
                "voice_condition": "stable_natural",
                "schedule": [{"mode": "natural", "start_s": 0, "end_s": None}],
                "files": {"participant_raw": str(relative)},
                "transcript": {"model": [
                    {"text": "Go ahead.", "start": 8.0, "end": 9.0,
                     "speaker": "personaplex"}]},
            }
            timing = {
                "schema": "hmo.timing-analysis.v4",
                "status": "estimated_pending_validation",
                "participant_intervals": [
                    {"start_ms": 1000.0, "end_ms": 3000.0, "detector": "rms"},
                    # 300 ms of breath during the model's turn.
                    {"start_ms": 8200.0, "end_ms": 8500.0, "detector": "rms"},
                    {"start_ms": 12000.0, "end_ms": 14000.0, "detector": "rms"},
                ],
                "assistant_intervals": [
                    {"start_ms": 8000.0, "end_ms": 9000.0, "detector": "rms"}],
                "route_switches": [], "overlaps": [], "barge_ins": [],
                "integrity": {"participant_capture_latency_correction_ms": 0.0},
                "result_artifact": {"path": "analysis/timing/timing.json"},
            }

            def words(_path: str) -> dict:
                return {"status": "complete", "error": None, "words": [
                    # Starts before the detected onset and runs into it: the
                    # interval owns it, or content words are lost at edges.
                    {"word": "One,", "start": 0.85, "end": 1.15},
                    {"word": "Building", "start": 1.2, "end": 1.7},
                    {"word": "sixteen.", "start": 1.8, "end": 2.4},
                    {"word": "Blue", "start": 12.3, "end": 12.7},
                    {"word": "door.", "start": 12.8, "end": 13.2},
                ]}

            result = prepare_dialogue_transcript(
                session, root, "analysis-7", timing, words)

            spoken = [row for row in result["utterances"]
                      if row["speaker"] == "participant"]
            self.assertEqual([row["text"] for row in spoken],
                             ["One, Building sixteen.", "", "Blue door."])
            self.assertEqual(spoken[1]["text_provenance"]["word_count"], 0)
            self.assertEqual(spoken[0]["text_provenance"]["method"],
                             "whisper_word_timestamps_whole_file")
            # Interval boundaries are the validated instrument and stay put.
            self.assertEqual((spoken[0]["start_ms"], spoken[0]["end_ms"]),
                             (1000.0, 3000.0))
            self.assertEqual(spoken[0]["text_provenance"]["speech_start_ms"], 850.0)

    def test_two_turns_never_share_an_onset(self):
        """Text arriving in the silence after a run used to take that run's
        start, which an earlier fragment already held: 338 of 1211 assistant
        turns shared or went backwards, ordering arbitrarily against the
        participant."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("sessions/onset/participant_raw.wav")
            self._write_wav(root / relative, 45.0)
            (root / relative).parent.joinpath("client_timeline.json").write_text(
                json.dumps({"capture": {"chunks": [
                    {"chunk_sequence": 1, "capture_start_sample": 0,
                     "sample_count": 2048, "timeline_start_ms": 0.0}]}}))
            session = {
                "session_id": "P1_R01_S02_A01",
                "voice_condition": "stable_natural",
                "schedule": [{"mode": "natural", "start_s": 0, "end_s": None}],
                "files": {"participant_raw": str(relative)},
                # The second fragment's text arrives after the model has
                # already fallen silent.
                "transcript": {"model": [
                    {"text": "Your order was delivered to building 18.",
                     "start": 19.6, "end": 27.3, "speaker": "personaplex"},
                    {"text": "I see.", "start": 25.0, "end": 30.4,
                     "speaker": "personaplex"},
                ]},
            }
            timing = {
                "schema": "hmo.timing-analysis.v4",
                "status": "estimated_pending_validation",
                "participant_intervals": [
                    {"start_ms": 40000.0, "end_ms": 41000.0, "detector": "rms"}],
                "assistant_intervals": [
                    {"start_ms": 19600.0, "end_ms": 28100.0, "detector": "rms"}],
                "route_switches": [], "overlaps": [], "barge_ins": [],
                "integrity": {"participant_capture_latency_correction_ms": 0.0},
                "result_artifact": {"path": "analysis/timing/timing.json"},
            }

            def transcribe(_path: str) -> dict:
                return {"text": "Participant utterance.", "segments": [],
                        "status": "complete", "error": None}

            result = prepare_dialogue_transcript(
                session, root, "analysis-6", timing, transcribe, self._no_slices)

            assistants = [row for row in result["utterances"]
                          if row["speaker"] == "assistant"]
            self.assertEqual(len(assistants), 2)
            self.assertEqual(assistants[0]["start_ms"], 19600.0)
            # Anchored to where the model was last audible, not to the run's
            # start, and never before the previous turn ended.
            self.assertEqual(assistants[1]["start_ms"], 28100.0)
            self.assertGreater(assistants[1]["start_ms"], assistants[0]["end_ms"] - 1)

    def test_the_assistant_falling_silent_ends_its_turn(self):
        """A stall ("one moment please" ... 40s ... a new turn) is two turns
        even though the participant never speaks."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("sessions/stall/participant_raw.wav")
            self._write_wav(root / relative, 60.0)
            (root / relative).parent.joinpath("client_timeline.json").write_text(
                json.dumps({"capture": {"chunks": [
                    {"chunk_sequence": 1, "capture_start_sample": 0,
                     "sample_count": 2048, "timeline_start_ms": 0.0}]}}))
            session = {
                "session_id": "P1_R01_S02_A01",
                "voice_condition": "stable_natural",
                "schedule": [{"mode": "natural", "start_s": 0, "end_s": None}],
                "files": {"participant_raw": str(relative)},
                "transcript": {"model": [
                    {"text": "One moment please.", "start": 10.0, "end": 11.0,
                     "speaker": "personaplex"},
                    {"text": "Yes, I found your assignment.", "start": 49.0,
                     "end": 51.0, "speaker": "personaplex"},
                ]},
            }
            timing = {
                "schema": "hmo.timing-analysis.v4",
                "status": "estimated_pending_validation",
                "participant_intervals": [
                    {"start_ms": 55000.0, "end_ms": 56000.0, "detector": "rms"}],
                # The assistant is silent between 11s and 49s.
                "assistant_intervals": [
                    {"start_ms": 10000.0, "end_ms": 11000.0, "detector": "rms"},
                    {"start_ms": 49000.0, "end_ms": 51000.0, "detector": "rms"},
                ],
                "route_switches": [], "overlaps": [], "barge_ins": [],
                "integrity": {"participant_capture_latency_correction_ms": 0.0},
                "result_artifact": {"path": "analysis/timing/timing.json"},
            }

            def transcribe(_path: str) -> dict:
                return {"text": "Participant utterance.", "segments": [],
                        "status": "complete", "error": None}

            result = prepare_dialogue_transcript(
                session, root, "analysis-4", timing, transcribe, self._no_slices)

            assistants = [row for row in result["utterances"]
                          if row["speaker"] == "assistant"]
            self.assertEqual([row["text"] for row in assistants],
                             ["One moment please.",
                              "Yes, I found your assignment."])

    def test_a_long_empty_unit_is_read_again_and_a_short_one_is_not(self):
        """The whole-file pass returns nothing for a reference number spoken
        flatly after a pause. Re-reading the unit recovers it; doing the same
        to a sub-second unit only invents stock politeness."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("sessions/test/participant_raw.wav")
            raw_path = root / relative
            self._write_wav(raw_path, duration_s=10.0)
            (raw_path.parent / "client_timeline.json").write_text(json.dumps({
                "capture": {"sample_rate_hz": 16000, "chunks": [{
                    "chunk_sequence": 1, "capture_start_sample": 0,
                    "sample_count": 2048, "timeline_start_ms": 0.0}]}}))
            session = {
                "session_id": "P1_R01_S02_A01",
                "voice_condition": "natural",
                "files": {"participant_raw": str(relative)},
                "transcript": {"model": [
                    {"text": "Thank you.", "start": 4.0, "end": 4.5,
                     "speaker": "personaplex"}]},
            }
            timing = {
                "schema": "hmo.timing-analysis.v4",
                "status": "estimated_pending_validation",
                "participant_intervals": [
                    {"start_ms": 2000.0, "end_ms": 3200.0, "detector": "rms"},
                    {"start_ms": 6000.0, "end_ms": 6400.0, "detector": "rms"},
                ],
                "assistant_intervals": [
                    {"start_ms": 4000.0, "end_ms": 4500.0, "detector": "rms"}],
                "route_switches": [], "overlaps": [], "barge_ins": [],
                "integrity": {"participant_capture_latency_correction_ms": 0.0},
                "result_artifact": {"path": "analysis/timing/timing.json"},
            }
            asked = []

            def transcribe(_path: str) -> dict:
                return {"status": "complete", "error": None, "words": []}

            def slice_words(_path: str, windows) -> list[list[dict]]:
                asked.extend(windows)
                return [[{"word": "One", "start": 2.1, "end": 2.4},
                         {"word": "two.", "start": 2.5, "end": 2.9}]]

            result = prepare_dialogue_transcript(
                session, root, "analysis-7", timing, transcribe, slice_words)

            participants = [row for row in result["utterances"]
                            if row["speaker"] == "participant"]
            self.assertEqual(asked, [(2.0, 3.2)])
            self.assertEqual(participants[0]["text"], "One two.")
            self.assertEqual(participants[0]["text_provenance"]["method"],
                             "whisper_word_timestamps_unit_slice")
            self.assertEqual(participants[0]["text_provenance"]["word_count"], 2)
            self.assertEqual(
                participants[0]["text_provenance"]["speech_start_ms"], 2100.0)
            self.assertEqual(participants[1]["text"], "")
            self.assertEqual(participants[1]["text_provenance"]["method"],
                             "whisper_word_timestamps_whole_file")


if __name__ == "__main__":
    unittest.main()
