"""Tests for the soundboard-audit post-processor over a synthetic run."""

from __future__ import annotations

import io
import json
import struct
import sys
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.audit_postprocess import process_run  # noqa: E402


def tiny_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack("<h", 0) * 1600)  # 0.1 s silence
    return buffer.getvalue()


MANIFEST = {
    "schema": "hmo.soundboard-audit-manifest.v2",
    "manifest_sha256": "f" * 64,
    "presentations": [
        {"index": 1, "slot_id": "nat1", "label": "item1 [nat]",
         "manipulation": "unconverted", "engine": None,
         "clip_sha256": "aa11", "raw_sha256": "raw-item1",
         "clip_duration_ms": 1500, "presentation_mode": "after_silence"},
        {"index": 2, "slot_id": "vc1", "label": "item1 [xvc]",
         "manipulation": "vc", "engine": "xvc",
         "clip_sha256": "bb22", "raw_sha256": "raw-item1",
         "clip_duration_ms": 1500, "presentation_mode": "after_silence"},
        {"index": 3, "slot_id": "vc2", "label": "item2 [xvc only]",
         "manipulation": "vc", "engine": "xvc",
         "clip_sha256": "cc33", "raw_sha256": "raw-item2",
         "clip_duration_ms": 1200, "presentation_mode": "during_pp_speech"},
    ],
}

RECORDS = [
    {"index": 1, "slot_id": "nat1", "label": "item1 [nat]", "status": "ok",
     "sent_clip_sha256": "aa11",             # delivery verified
     "t_play_start_ms": 1000.0, "t_play_end_ms": 2500.0,
     "response_latency_ms": 640.0, "pp_yield_latency_ms": None,
     "pp_speech_events": [
         {"type": "pp_speech_start", "timestampMs": 200.0},
         {"type": "pp_speech_end", "timestampMs": 800.0},
         {"type": "pp_speech_start", "timestampMs": 3140.0},
         {"type": "pp_speech_end", "timestampMs": 6000.0}]},
    {"index": 2, "slot_id": "vc1", "label": "item1 [xvc]", "status": "ok",
     "sent_clip_sha256": "DIFFERENT",        # delivery NOT verified
     "t_play_start_ms": 1000.0, "t_play_end_ms": 2500.0,
     "response_latency_ms": 900.0, "pp_yield_latency_ms": None,
     "pp_speech_events": [
         # PP talks INTO the clip window: overlap 1200-2500 + onset at 2000.
         {"type": "pp_speech_start", "timestampMs": 700.0},
         {"type": "pp_speech_end", "timestampMs": 1200.0},
         {"type": "pp_speech_start", "timestampMs": 2000.0},
         {"type": "pp_speech_end", "timestampMs": 5200.0}]},
    {"index": 3, "slot_id": "vc2", "label": "item2 [xvc only]",
     "status": "no_response", "sent_clip_sha256": "cc33",
     "t_play_start_ms": 900.0, "t_play_end_ms": 2100.0,
     "response_latency_ms": None, "pp_yield_latency_ms": 350.0,
     "pp_speech_events": []},
]


class AuditPostprocessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "20260802T120000Z_ffffffff"
        self.run_dir.mkdir(parents=True)
        wav = tiny_wav_bytes()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("manifest.json", json.dumps(MANIFEST))
            archive.writestr("run_log.json", json.dumps({"records": RECORDS}))
            for record in RECORDS:
                archive.writestr(
                    f"runs/{record['index']:03d}_{record['slot_id']}/personaplex.wav",
                    wav)
        (self.run_dir / "results.zip").write_bytes(buffer.getvalue())
        self.calls: list[str] = []

    def tearDown(self):
        self._tmp.cleanup()

    def _transcriber(self, path: str) -> dict:
        self.calls.append(path)
        return {"text": f"response {len(self.calls)}", "status": "complete"}

    def test_process_run_end_to_end(self):
        summary = process_run(self.run_dir, transcriber=self._transcriber)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["counts"],
                         {"presentations": 3, "ok": 2, "delivery_verified": 2})
        # Pairing: item1 has unconverted+vc -> ok; item2 vc-only -> failure.
        self.assertEqual(len(summary["pairing_failures"]), 1)
        self.assertEqual(summary["pairing_failures"][0]["raw_sha256"], "raw-item2")

        rows = {r["index"]: r for r in summary["presentations"]}
        self.assertTrue(rows[1]["delivery_verified"])
        self.assertFalse(rows[2]["delivery_verified"])   # hash mismatch caught
        # Overlap: PP run 700-1200 overlaps clip 1000-2500 by 200 ms; run
        # 2000-5200 overlaps by 500 ms and ONSETS inside the window.
        self.assertAlmostEqual(rows[2]["overlap_ms"], 700.0, places=1)
        self.assertEqual(rows[2]["pp_onsets_during_clip"], 1)
        self.assertEqual(rows[1]["pp_onsets_during_clip"], 0)
        self.assertEqual(rows[1]["pp_response_transcript"], "response 1")
        self.assertEqual(rows[3]["pp_yield_latency_ms"], 350.0)

        # Item x condition table groups by raw hash + manipulation.
        table = {(r["item_raw_sha256"], r["manipulation"]): r
                 for r in summary["item_condition_summary"]}
        self.assertEqual(table[("raw-item1"[:12], "unconverted")]["n_ok"], 1)
        self.assertEqual(
            table[("raw-item1"[:12], "vc")]["response_latency_mean_ms"], 900.0)
        self.assertEqual(
            table[("raw-item2"[:12], "vc")]["n_no_response"], 1)

        # Artifacts on disk: summary json/csv + cached transcripts.
        self.assertTrue((self.run_dir / "audit_summary.json").exists())
        self.assertTrue((self.run_dir / "audit_summary.csv").exists())
        self.assertEqual(
            len(list((self.run_dir / "transcripts").glob("*.json"))), 3)

        # Second pass without --force skips (returns None), keeps transcripts.
        self.assertIsNone(process_run(self.run_dir, transcriber=self._transcriber))
        self.assertEqual(len(self.calls), 3)

    def test_missing_inputs_reported(self):
        empty = Path(self._tmp.name) / "empty_run"
        empty.mkdir()
        result = process_run(empty, transcriber=self._transcriber)
        self.assertIn("error", result)


SCRIPT_MANIFEST = {
    "schema": "hmo.soundboard-audit-manifest.v2",
    "mode": "script",
    "manifest_sha256": "e" * 64,
    "reps": 2,
    "inter_turn_gap_ms": 500,
    "script": [
        {"turn": 1, "slot_id": "s1", "label": "line 1",
         "manipulation": "unconverted", "engine": None,
         "clip_sha256": "t1hash", "raw_sha256": "r1", "clip_duration_ms": 1200},
        {"turn": 2, "slot_id": "s2", "label": "line 2",
         "manipulation": "vc", "engine": "xvc",
         "clip_sha256": "t2hash", "raw_sha256": "r2", "clip_duration_ms": 1400},
    ],
}


def script_session(rep: int, second_turn_status: str = "ok") -> dict:
    return {
        "rep": rep, "status": "ok", "greeted": True,
        "turns": [
            {"turn": 1, "slot_id": "s1", "label": "line 1", "status": "ok",
             "t_play_start_ms": 1000.0, "t_play_end_ms": 2200.0,
             "response_latency_ms": 150.0, "pp_spoke_during_clip": True,
             "sent_clip_sha256": "t1hash", "notes": []},
            {"turn": 2, "slot_id": "s2", "label": "line 2",
             "status": second_turn_status,
             "t_play_start_ms": 6000.0, "t_play_end_ms": 7400.0,
             "response_latency_ms": 220.0 if second_turn_status == "ok" else None,
             "pp_spoke_during_clip": False,
             "sent_clip_sha256": "WRONG", "notes": []},
        ],
        # Energy events: a run overlapping turn 1's clip window (1500-2000),
        # then a response run after it.
        "pp_speech_events": [
            {"type": "pp_energy", "timestampMs": 1500.0},
            {"type": "pp_energy", "timestampMs": 1700.0},
            {"type": "pp_energy", "timestampMs": 2000.0},
            {"type": "pp_energy", "timestampMs": 2400.0},
        ],
        "pp_transcript": [{"text": "hi", "speaker": "personaplex"}],
    }


class ScriptModePostprocessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "20260802T130000Z_eeeeeeee"
        self.run_dir.mkdir(parents=True)
        wav = tiny_wav_bytes()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("manifest.json", json.dumps(SCRIPT_MANIFEST))
            archive.writestr("run_log.json", json.dumps({
                "records": [script_session(1),
                            script_session(2, second_turn_status="no_response")]}))
            for rep in (1, 2):
                archive.writestr(f"runs/rep_{rep:03d}/personaplex.wav", wav)
        (self.run_dir / "results.zip").write_bytes(buffer.getvalue())
        self.calls: list[str] = []

    def tearDown(self):
        self._tmp.cleanup()

    def _transcriber(self, path: str) -> dict:
        self.calls.append(path)
        return {"text": f"conversation {len(self.calls)}", "status": "complete"}

    def test_script_run_end_to_end(self):
        summary = process_run(self.run_dir, transcriber=self._transcriber)
        self.assertEqual(summary["mode"], "script")
        self.assertEqual(summary["counts"]["replays"], 2)
        self.assertEqual(summary["counts"]["turns"], 4)
        self.assertEqual(summary["counts"]["turns_ok"], 3)
        # Delivery: turn 1 hash matches in both reps; turn 2 never does.
        self.assertEqual(summary["counts"]["delivery_verified"], 2)
        table = {row["turn"]: row for row in summary["turn_summary"]}
        self.assertEqual(table[1]["n"], 2)
        self.assertEqual(table[1]["n_delivery_verified"], 2)
        self.assertEqual(table[1]["n_pp_barge_in"], 2)
        self.assertEqual(table[1]["response_latency_mean_ms"], 150.0)
        self.assertEqual(table[2]["n_delivery_verified"], 0)
        self.assertEqual(table[2]["n_no_response"], 1)
        # Overlap: energy run 1500-2020 overlaps clip window 1000-2200 fully
        # (520ms), run 2400-2420 is outside it.
        turn1 = [r for r in summary["turns"] if r["turn"] == 1][0]
        self.assertAlmostEqual(turn1["overlap_ms"], 520.0, places=1)
        self.assertEqual(turn1["pp_onsets_during_clip"], 1)
        # One whole-conversation transcript per replay.
        self.assertEqual(len(self.calls), 2)
        sessions = {s["rep"]: s for s in summary["sessions"]}
        self.assertEqual(sessions[1]["pp_full_transcript"], "conversation 1")
        self.assertTrue((self.run_dir / "audit_summary.csv").exists())


class InterleavedScriptPostprocessTests(unittest.TestCase):
    """Interleaved A/B replays: per-condition scripts, condition-suffixed run
    folders, gap taken from the frozen manifest constants."""

    MANIFEST = {
        "schema": "hmo.soundboard-audit-manifest.v2",
        "mode": "script",
        "interleaved": True,
        "manifest_sha256": "d" * 64,
        "reps": 1,
        "timing": {"ppSpeakingGapMs": 350, "ppEnergyThresholdRms": 0.02},
        "scripts": [
            {"condition": "natural", "turns": [
                {"turn": 1, "slot_id": "n1", "label": "line [nat]",
                 "manipulation": "unconverted", "engine": None,
                 "clip_sha256": "natHash", "raw_sha256": "r1",
                 "clip_duration_ms": 1000}]},
            {"condition": "converted", "turns": [
                {"turn": 1, "slot_id": "c1", "label": "line [xvc]",
                 "manipulation": "vc", "engine": "xvc",
                 "clip_sha256": "convHash", "raw_sha256": "r1",
                 "clip_duration_ms": 1000}]},
        ],
        "replay_plan": [
            {"rep": 1, "condition": "natural", "cycle": 1},
            {"rep": 2, "condition": "converted", "cycle": 1},
        ],
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "20260802T140000Z_dddddddd"
        self.run_dir.mkdir(parents=True)

        def session(rep, condition, sent):
            return {
                "rep": rep, "condition": condition, "status": "ok",
                "greeted": True,
                "turns": [{"turn": 1, "slot_id": "x", "label": "line",
                           "status": "ok",
                           "t_play_start_ms": 1000.0, "t_play_end_ms": 2000.0,
                           "response_latency_ms": 120.0 if condition == "natural" else 480.0,
                           "pp_spoke_during_clip": False,
                           "sent_clip_sha256": sent, "notes": []}],
                "pp_speech_events": [], "pp_transcript": [],
            }

        wav = tiny_wav_bytes()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("manifest.json", json.dumps(self.MANIFEST))
            archive.writestr("run_log.json", json.dumps({"records": [
                session(1, "natural", "natHash"),
                session(2, "converted", "convHash"),
            ]}))
            archive.writestr("runs/rep_001_natural/personaplex.wav", wav)
            archive.writestr("runs/rep_002_converted/personaplex.wav", wav)
        (self.run_dir / "results.zip").write_bytes(buffer.getvalue())

    def tearDown(self):
        self._tmp.cleanup()

    def test_interleaved_grouping_and_lookup(self):
        calls = []

        def transcriber(path):
            calls.append(path)
            return {"text": "t", "status": "complete"}

        summary = process_run(self.run_dir, transcriber=transcriber)
        self.assertTrue(summary["interleaved"])
        self.assertEqual(summary["detection"]["energy_run_gap_ms"], 350)
        self.assertEqual(summary["detection"]["source"], "manifest.timing")
        # Delivery verified per condition against that condition's script hash.
        self.assertEqual(summary["counts"]["delivery_verified"], 2)
        table = {(r["condition"], r["turn"]): r for r in summary["turn_summary"]}
        self.assertEqual(table[("natural", 1)]["response_latency_mean_ms"], 120.0)
        self.assertEqual(table[("converted", 1)]["response_latency_mean_ms"], 480.0)
        self.assertEqual(table[("converted", 1)]["manipulation"], "vc")
        # Condition-suffixed folders were found for transcription.
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
