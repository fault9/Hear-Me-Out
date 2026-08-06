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

from study.audit_postprocess import (  # noqa: E402
    _clustered_cell_summary,
    process_run,
)


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
     "sent_clip_sha256": "aa11",             # stimulus hash matches
     "t_play_start_ms": 1000.0, "t_play_end_ms": 2500.0,
     "response_latency_ms": 640.0, "pp_yield_latency_ms": None,
     "pp_speech_events": [
         {"type": "pp_speech_start", "timestampMs": 200.0},
         {"type": "pp_speech_end", "timestampMs": 800.0},
         {"type": "pp_speech_start", "timestampMs": 3140.0},
         {"type": "pp_speech_end", "timestampMs": 6000.0}]},
    {"index": 2, "slot_id": "vc1", "label": "item1 [xvc]", "status": "ok",
     "sent_clip_sha256": "DIFFERENT",        # stimulus hash mismatch
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
        self.assertEqual(summary["counts"]["presentations"], 3)
        self.assertEqual(summary["counts"]["ok"], 2)
        self.assertEqual(summary["counts"]["evaluable"], 0)
        self.assertEqual(summary["counts"]["delivery_verified"], 0)
        self.assertEqual(summary["counts"]["stimulus_integrity_verified"], 2)
        # Pairing: item1 has unconverted+vc -> ok; item2 vc-only -> failure.
        self.assertEqual(len(summary["pairing_failures"]), 1)
        self.assertEqual(summary["pairing_failures"][0]["raw_sha256"], "raw-item2")

        rows = {r["index"]: r for r in summary["presentations"]}
        self.assertIsNone(rows[1]["delivery_verified"])
        self.assertTrue(rows[1]["stimulus_integrity_verified"])
        self.assertFalse(rows[2]["stimulus_integrity_verified"])
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
        self.assertTrue((self.run_dir / "audit_presentations.csv").exists())
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

    def test_uncertainty_resamples_conversations_not_turns(self):
        rows = [
            {
                "rep": rep,
                "technical_evaluable": True,
                "overlap_ms": overlap,
                "premature_assistant_onset": False,
                "status": "ok",
            }
            for rep, overlap in ((1, 0.0), (1, 0.0), (2, 100.0), (2, 100.0))
        ]
        result = _clustered_cell_summary(rows, "rep", seed=17)
        self.assertEqual(result["n_attempted_conversation_clusters"], 2)
        self.assertEqual(result["n_contributing_conversation_clusters"], 2)
        self.assertEqual(result["overlap_ms_n"], 4)
        self.assertEqual(result["overlap_ms_mean"], 50.0)
        self.assertEqual(result["overlap_ms_ci95_low"], 0.0)
        self.assertEqual(result["overlap_ms_ci95_high"], 100.0)


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
        # Legacy fixtures verify stimulus hashes but have no relay receipt.
        self.assertEqual(summary["counts"]["delivery_verified"], 0)
        self.assertEqual(summary["counts"]["stimulus_integrity_verified"], 2)
        table = {row["turn"]: row for row in summary["turn_summary"]}
        self.assertEqual(table[1]["n"], 2)
        self.assertEqual(table[1]["n_delivery_verified"], 0)
        self.assertEqual(table[1]["n_stimulus_integrity_verified"], 2)
        self.assertEqual(table[1]["n_runner_assistant_overlap_candidate"], 2)
        self.assertEqual(table[1]["response_latency_mean_ms"], 150.0)
        self.assertEqual(table[2]["n_delivery_verified"], 0)
        self.assertEqual(table[2]["n_no_response"], 1)
        # Legacy packet-end events represent three 20-ms energetic packets in
        # turn 1. The 180/280-ms acoustic silences between them are not overlap.
        turn1 = [r for r in summary["turns"] if r["turn"] == 1][0]
        self.assertAlmostEqual(turn1["overlap_ms"], 60.0, places=1)
        self.assertEqual(turn1["pp_onsets_during_clip"], 1)
        self.assertEqual(turn1["timing_source"], "legacy_energy_events")
        # One whole-conversation transcript per replay.
        self.assertEqual(len(self.calls), 2)
        sessions = {s["rep"]: s for s in summary["sessions"]}
        self.assertEqual(sessions[1]["pp_full_transcript"], "conversation 1")
        self.assertTrue((self.run_dir / "audit_summary.csv").exists())
        self.assertTrue((self.run_dir / "audit_turns.csv").exists())
        self.assertTrue((self.run_dir / "audit_sessions.csv").exists())


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
        self.assertEqual(
            summary["detection"]["run_gap_source"], "manifest.timing"
        )
        # Delivery verified per condition against that condition's script hash.
        self.assertEqual(summary["counts"]["delivery_verified"], 0)
        self.assertEqual(summary["counts"]["stimulus_integrity_verified"], 2)
        table = {(r["condition"], r["turn"]): r for r in summary["turn_summary"]}
        self.assertEqual(table[("natural", 1)]["response_latency_mean_ms"], 120.0)
        self.assertEqual(table[("converted", 1)]["response_latency_mean_ms"], 480.0)
        self.assertEqual(table[("converted", 1)]["manipulation"], "vc")
        # Condition-suffixed folders were found for transcription.
        self.assertEqual(len(calls), 2)


class PlaybackTimelinePostprocessTests(unittest.TestCase):
    """Scheduled packet boundaries override approximate live-runner values."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "20260802T150000Z_cccccccc"
        self.run_dir.mkdir(parents=True)
        manifest = {
            "schema": "hmo.soundboard-audit-manifest.v3",
            "mode": "matched",
            "manifest_sha256": "c" * 64,
            "timing": {
                "ppSpeakingGapMs": 350,
                "ppEnergyThresholdRms": 0.02,
                "interruptFireDelayMs": 200,
                "interruptFireToleranceMs": 20,
            },
            "presentations": [{
                "index": 1,
                "slot_id": "natural",
                "label": "natural probe",
                "manipulation": "unconverted",
                "route": "natural",
                "source_speaker": "masculine_presenting",
                "engine": None,
                "target_id": None,
                "target_label": None,
                "target_sha256": None,
                "clip_sha256": "clip",
                "raw_sha256": "raw",
                "clip_duration_ms": 1000,
                "presentation_mode": "during_pp_speech",
                "input_timing_source_sha256": "raw",
                "input_timing": {
                    "sample_rate_hz": 1000,
                    "total_samples": 1000,
                    "frame_samples": 20,
                    "rms_threshold": 0.012,
                    "min_speech_samples": 80,
                    "min_pause_samples": 250,
                    "speech_start_sample": 0,
                    "speech_end_sample": 800,
                    "speech_intervals": [
                        {"start_sample": 0, "end_sample": 300},
                        {"start_sample": 500, "end_sample": 800},
                    ],
                    "pause_intervals": [
                        {"start_sample": 300, "end_sample": 500},
                    ],
                },
                "clip_audio": {
                    "sample_rate_hz": 24000,
                    "normalized": True,
                    "target_lufs": -23.0,
                    "measured_lufs": -23.1,
                    "duration_drift_ms": 0.0,
                },
            }],
        }
        manifest["presentations"].append({
            **manifest["presentations"][0],
            "index": 2,
            "slot_id": "converted",
            "label": "converted probe",
            "manipulation": "vc",
            "route": "converted",
            "engine": "xvc",
            "target_id": "p229",
            "target_label": "p229 feminine-presenting",
            "target_sha256": "target",
            "clip_sha256": "converted-clip",
        })
        record = {
            "index": 1,
            "slot_id": "natural",
            "label": "natural probe",
            "status": "ok",
            "t_handshake_ms": 500.0,
            "t_play_start_ms": 1000.0,
            "t_play_end_ms": 2000.0,
            "response_latency_ms": 999.0,
            "sent_clip_sha256": "clip",
            "delivery_audit": {
                "status": "complete",
                "browser_sent": {"frames": 2, "bytes": 42, "sha256": "wire"},
                "relay_received": {"frames": 2, "bytes": 42, "sha256": "wire"},
                "relay_forwarded": {"frames": 2, "bytes": 42, "sha256": "wire"},
            },
            "pp_speech_events": [],
        }
        converted_record = {
            **record,
            "index": 2,
            "slot_id": "converted",
            "label": "converted probe",
            "sent_clip_sha256": "converted-clip",
        }
        timeline = {
            "schema": "hmo.client-playback-timeline.v2",
            "epoch": "personaplex_handshake_performance_now",
            "queue_underrun_count": 0,
            "queue_underrun_total_ms": 0,
            "queue_underrun_max_ms": 0,
            "assistant_packets": [
                # Absolute 800-1450 ms: PP is speaking when input starts, then
                # yields 450 ms later. Its last 150 ms fall in the frozen pause.
                {"timeline_start_ms": 300, "timeline_end_ms": 950,
                 "rms": 0.2},
                # Silence must not extend the first assistant interval.
                {"timeline_start_ms": 950, "timeline_end_ms": 1100,
                 "rms": 0.001},
                # Absolute 2300-2325 ms: authoritative 300-ms response gap.
                {"timeline_start_ms": 1800, "timeline_end_ms": 1825,
                 "rms": 0.1},
            ],
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr(
                "run_log.json",
                json.dumps({"records": [record, converted_record]}),
            )
            archive.writestr("runs/001_natural/personaplex.wav", tiny_wav_bytes())
            archive.writestr("runs/001_natural/sent.wav", tiny_wav_bytes())
            archive.writestr(
                "runs/001_natural/playback_timeline.json", json.dumps(timeline)
            )
            archive.writestr("runs/002_converted/personaplex.wav", tiny_wav_bytes())
            archive.writestr("runs/002_converted/sent.wav", tiny_wav_bytes())
            archive.writestr(
                "runs/002_converted/playback_timeline.json", json.dumps(timeline)
            )
        (self.run_dir / "results.zip").write_bytes(buffer.getvalue())

    def tearDown(self):
        self._tmp.cleanup()

    def test_packet_timeline_is_authoritative(self):
        summary = process_run(
            self.run_dir,
            transcriber=lambda _: {"text": "", "status": "complete"},
        )
        row = summary["presentations"][0]
        self.assertEqual(row["timing_source"], "playback_timeline_v2")
        self.assertTrue(row["delivery_verified"])
        self.assertEqual(row["delivery_status"], "verified")
        self.assertTrue(row["technical_evaluable"])
        self.assertEqual(row["technical_invalid_reasons"], [])
        self.assertEqual(row["runner_response_latency_ms"], 999.0)
        self.assertIsNone(row["response_latency_ms"])
        self.assertEqual(row["post_clip_response_latency_ms"], 300.0)
        self.assertEqual(row["response_onset_offset_ms"], -1000.0)
        self.assertTrue(row["premature_assistant_onset"])
        self.assertEqual(row["overlap_ms"], 300.0)
        self.assertTrue(row["overlap_event"])
        self.assertEqual(row["pp_onsets_during_clip"], 0)
        self.assertEqual(row["prespecified_pause_count"], 1)
        self.assertEqual(
            row["assistant_speech_during_prespecified_pause_ms"], 150.0
        )
        self.assertEqual(
            row["assistant_onsets_during_prespecified_pause"], 0
        )
        self.assertTrue(row["assistant_speaking_at_interrupt"])
        self.assertEqual(row["fire_offset_into_assistant_turn_ms"], 200.0)
        self.assertEqual(row["interruption_fire_error_ms"], 0.0)
        self.assertEqual(row["assistant_stop_latency_ms"], 450.0)
        self.assertTrue(row["assistant_yielded_before_input_offset"])
        self.assertEqual(row["post_interruption_overlap_ms"], 300.0)
        self.assertTrue(row["response_resumed"])
        self.assertEqual(row["response_resumption_latency_ms"], 500.0)
        self.assertEqual(
            summary["detection"]["timing_source"], "playback_timeline_v2"
        )
        self.assertEqual(summary["schema"], "hmo.soundboard-audit-summary.v2")
        self.assertEqual(
            summary["clustered_uncertainty"]["cluster_key"], "index"
        )
        cell = next(
            item for item in summary["cell_summary"] if item["route"] == "natural"
        )
        self.assertEqual(cell["source_speaker"], "masculine_presenting")
        self.assertEqual(cell["route"], "natural")
        self.assertEqual(cell["n_attempted_conversation_clusters"], 1)
        self.assertEqual(cell["n_contributing_conversation_clusters"], 1)
        self.assertEqual(cell["n_fully_evaluable_conversation_clusters"], 1)
        self.assertEqual(cell["complete_audio_delivery_rate_mean"], 1.0)
        self.assertEqual(cell["overlap_ms_mean"], 300.0)
        self.assertEqual(cell["overlap_ms_ci95_low"], 300.0)
        self.assertEqual(cell["assistant_stop_latency_ms_mean"], 450.0)
        self.assertTrue((self.run_dir / "audit_cell_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
