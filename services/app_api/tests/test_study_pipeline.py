from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from study.artifacts import atomic_write_bytes, sha256_file
from study.counterbalance import (CounterbalanceError, allocate, balance_report,
                                  resolve_target_assignment)
from study.storage import SqliteBackend
from study.transition_analysis import prepare_session_analysis, route_regions


def write_wav(path: Path, seconds: float = 3.0, rate: int = 16000) -> None:
    samples = np.arange(round(seconds * rate))
    signal = (np.sin(2 * np.pi * 220 * samples / rate) * 12000).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(signal.tobytes())


class ArtifactTests(unittest.TestCase):
    def test_exclusive_write_preserves_original(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recording.bin"
            atomic_write_bytes(path, b"first", exclusive=True)
            digest = sha256_file(path)
            with self.assertRaises(FileExistsError):
                atomic_write_bytes(path, b"second", exclusive=True)
            self.assertEqual(path.read_bytes(), b"first")
            self.assertEqual(sha256_file(path), digest)


class StorageTests(unittest.TestCase):
    def test_restarted_scenario_gets_a_new_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = SqliteBackend(str(Path(temp) / "study.db"))
            study = backend.create_study("test")
            scenario = backend.add_scenario(study["id"], {
                "order_idx": 0, "title": "one", "scenario_card": {},
                "system_prompt": "prompt", "voice_schedule": [],
            })
            participant = backend.generate_participants(study["id"], 1, [scenario["id"]])[0]
            run = backend.start_run(participant["participant_id"], "restart")
            self.assertEqual(backend.next_session_attempt(participant["participant_id"], run["id"], 1), 1)
            backend.create_session("s1", participant["participant_id"], "scenario_1", 1,
                                   "natural", "", run["id"], run["attempt"], 1, [], {})
            self.assertEqual(backend.next_session_attempt(participant["participant_id"], run["id"], 1), 2)
            with self.assertRaises(sqlite3.IntegrityError):
                backend.create_session("s2", participant["participant_id"], "scenario_1", 1,
                                       "natural", "", run["id"], run["attempt"], 1, [], {})

    def test_deferred_participant_assignment_is_persisted_once(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = SqliteBackend(str(Path(temp) / "study.db"))
            study = backend.create_study("test")
            participant = backend.generate_participants(
                study["id"], 1, [10], [{"allocation_status": "awaiting_profile"}])[0]
            assigned = backend.assign_participant(participant["participant_id"], {
                "variant_id": "A", "target_ref": "masculine_presenting",
                "scenario_order": [10], "assignment": {"10": {"condition": "vc"}},
            }, "Woman")
            self.assertEqual(assigned["allocation_status"], "assigned")
            self.assertEqual(assigned["allocation_stratum"], "Woman")
            self.assertEqual(assigned["target_ref"], "masculine_presenting")
            with self.assertRaises(ValueError):
                backend.assign_participant(participant["participant_id"], {}, "Man")


class CounterbalanceTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = [{"id": 10, "order_idx": 0}, {"id": 20, "order_idx": 1}]
        self.targets = [{"ref": "a"}, {"ref": "b"}]
        schedule_a = [{"mode": "natural", "start_s": 0, "end_s": 1},
                      {"mode": "vc", "engine": "xvc", "start_s": 1, "end_s": None}]
        schedule_b = list(reversed(schedule_a))
        self.settings = {"counterbalancing": {
            "conditions": {"A": {"voice_schedule": schedule_a},
                           "B": {"voice_schedule": schedule_b}},
            "variants": [
                {"id": "v1", "target_ref": "a", "scenario_order": [1, 2],
                 "condition_assignment": {1: "A", 2: "B"}},
                {"id": "v2", "target_ref": "b", "scenario_order": [2, 1],
                 "condition_assignment": {1: "B", 2: "A"}},
            ],
        }}

    def test_least_filled_assignment_is_balanced_and_deterministic(self):
        assigned = allocate(self.settings, self.scenarios, self.targets, [], 5)
        self.assertEqual([row["variant_id"] for row in assigned], ["v1", "v2", "v1", "v2", "v1"])
        participants = [{"variant_id": row["variant_id"]} for row in assigned]
        report = balance_report(self.settings, self.scenarios, self.targets, participants)
        self.assertEqual(report["variant_counts"], {"v1": 3, "v2": 2})
        self.assertEqual(report["allocated_targets"], {"a": 3, "b": 2})

    def test_gender_answer_selects_opposite_presenting_target(self):
        settings = {"counterbalancing": {"target_assignment": {
            "questionnaire_kind": "background",
            "answer_id": "gender_identity",
            "target_by_answer": {
                "Woman": "masculine_presenting",
                "Man": "feminine_presenting",
            },
        }}}
        woman = resolve_target_assignment(
            settings, "background", {"gender_identity": "Woman"})
        man = resolve_target_assignment(
            settings, "background", {"gender_identity": "Man"})
        self.assertEqual(woman, {
            "allocation_stratum": "Woman", "target_ref": "masculine_presenting"})
        self.assertEqual(man, {
            "allocation_stratum": "Man", "target_ref": "feminine_presenting"})
        default = allocate(settings, [{
            "id": 10, "order_idx": 0,
            "voice_schedule": [{"mode": "vc", "engine": "xvc",
                                "start_s": 0, "end_s": None}],
        }], [{"ref": "masculine_presenting"},
             {"ref": "feminine_presenting"}], [], 1,
            target_ref=woman["target_ref"],
            allocation_stratum=woman["allocation_stratum"])[0]
        self.assertEqual(default["variant_id"], "default")
        self.assertEqual(
            default["assignment"]["10"]["voice_schedule"][0]["target_ref"],
            "masculine_presenting")
        with self.assertRaises(CounterbalanceError):
            resolve_target_assignment(settings, "background", {
                "gender_identity": "Prefer not to answer"})

    def test_variants_balance_independently_within_gender_groups(self):
        settings = {"counterbalancing": {
            "target_assignment": {
                "answer_id": "gender_identity",
                "target_by_answer": {
                    "Woman": "masculine_presenting",
                    "Man": "feminine_presenting",
                },
            },
            "conditions": self.settings["counterbalancing"]["conditions"],
            "variants": [
                {"id": "A", "scenario_order": [1, 2],
                 "condition_assignment": {1: "A", 2: "B"}},
                {"id": "B", "scenario_order": [2, 1],
                 "condition_assignment": {1: "B", 2: "A"}},
            ],
        }}
        targets = [{"ref": "masculine_presenting"},
                   {"ref": "feminine_presenting"}]
        participants = [
            {"variant_id": "A", "allocation_stratum": "Woman"},
            {"variant_id": "B", "allocation_stratum": "Woman"},
            {"variant_id": "A", "allocation_stratum": "Man"},
        ]
        woman = allocate(settings, self.scenarios, targets, participants, 1,
                         target_ref="masculine_presenting", allocation_stratum="Woman")[0]
        man = allocate(settings, self.scenarios, targets, participants, 1,
                       target_ref="feminine_presenting", allocation_stratum="Man")[0]
        self.assertEqual(woman["variant_id"], "A")
        self.assertEqual(woman["target_ref"], "masculine_presenting")
        self.assertEqual(man["variant_id"], "B")
        self.assertEqual(man["target_ref"], "feminine_presenting")
        for allocation in (woman, man):
            for override in allocation["assignment"].values():
                vc = [segment for segment in override["voice_schedule"]
                      if segment.get("mode") == "vc"]
                self.assertTrue(all(segment["target_ref"] == allocation["target_ref"]
                                    for segment in vc))


class TransitionTests(unittest.TestCase):
    def test_regions_and_boundary_clips_follow_sample_events(self):
        events = [
            {"event": "route_activated", "event_sequence": 1, "from_mode": None,
             "to_mode": "natural", "input_sample": 0, "transmitted_sample": 0,
             "requested_start_s": 0},
            {"event": "transmitted_window", "event_sequence": 2, "route_mode": "natural",
             "input_start_sample": 0, "input_end_sample": 16000,
             "transmitted_start_sample": 0, "transmitted_end_sample": 16000},
            {"event": "route_activated", "event_sequence": 3, "from_mode": "natural",
             "to_mode": "vc", "input_sample": 16000, "transmitted_sample": 16000,
             "requested_start_s": 0.9},
            {"event": "transmitted_window", "event_sequence": 4, "route_mode": "vc",
             "input_start_sample": 16000, "input_end_sample": 32000,
             "transmitted_start_sample": 16000, "transmitted_end_sample": 32000},
            {"event": "transmitted_window", "event_sequence": 5, "route_mode": "vc",
             "input_start_sample": 32000, "input_end_sample": 48000,
             "transmitted_start_sample": 32000, "transmitted_end_sample": 48000},
            {"event": "client_capture_summary", "event_sequence": 6,
             "estimated_dropped_samples": 32},
            {"event": "stream_stop", "event_sequence": 7, "input_samples": 48000},
        ]
        self.assertEqual([r["mode"] for r in route_regions(events)], ["natural", "vc"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session_dir = root / "sessions" / "attempt"
            session_dir.mkdir(parents=True)
            for name in ("participant.wav", "participant_raw.wav", "target.wav"):
                write_wav(session_dir / name)
            with (session_dir / "events.jsonl").open("w") as stream:
                for row in events:
                    stream.write(json.dumps(row) + "\n")
            session = {"session_id": "s1", "files": {
                "participant": "sessions/attempt/participant.wav",
                "participant_raw": "sessions/attempt/participant_raw.wav",
            }}
            result = prepare_session_analysis(session, root, "analysis-1")
            self.assertEqual(len(result["score_jobs"]), 1)
            self.assertEqual(result["transitions"][0]["activation_lag_ms"], 100.0)
            self.assertEqual(result["timeline_quality"]["client_capture"]["estimated_dropped_samples"], 32)
            scored_clip = root / result["score_jobs"][0]["converted"]
            with wave.open(str(scored_clip), "rb") as wav:
                self.assertEqual(wav.getnframes(), 16000)


if __name__ == "__main__":
    unittest.main()
