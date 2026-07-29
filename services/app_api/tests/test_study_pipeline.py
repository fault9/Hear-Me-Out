from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import yaml

from study.artifacts import atomic_write_bytes, sha256_file
from study.analysis import _participant_segments
from study.counterbalance import (CounterbalanceError, allocate, balance_report,
                                  choose_balanced_target, resolve_target_assignment,
                                  validate_and_compile)
from study.playback import (ensure_stable_converted_playback,
                            ensure_transition_playback)
from study.questionnaires import missing_required_answers
from study.storage import SqliteBackend
from study.timing_analysis import (assistant_intervals, prepare_timing_analysis,
                                   speech_intervals, validate_intervals)
from study.transcript_timing import whisper_timestamp_segments
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
    def test_consent_is_scoped_to_the_current_run(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = SqliteBackend(str(Path(temp) / "study.db"))
            study = backend.create_study("test")
            participant = backend.generate_participants(study["id"], 1, [10])[0]
            first = backend.start_run(participant["participant_id"], "restart")
            self.assertFalse(backend.has_answer(
                participant["participant_id"], first["id"], "consent"))
            backend.save_answer(participant["participant_id"], None, "consent", {
                "consent_participation": True,
            })
            self.assertTrue(backend.has_answer(
                participant["participant_id"], first["id"], "consent"))
            second = backend.start_run(participant["participant_id"], "restart")
            self.assertFalse(backend.has_answer(
                participant["participant_id"], second["id"], "consent"))

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

    def test_unmapped_gender_answer_uses_configured_fallback_pool(self):
        settings = {"counterbalancing": {"target_assignment": {
            "questionnaire_kind": "background",
            "answer_id": "gender_identity",
            "target_by_answer": {
                "Woman": "masculine_presenting",
                "Man": "feminine_presenting",
            },
            "fallback_targets": ["masculine_presenting", "feminine_presenting"],
        }}}
        resolved = resolve_target_assignment(
            settings, "background", {"gender_identity": "Non-binary"})
        self.assertEqual(resolved, {
            "allocation_stratum": "fallback",
            "target_candidates": ["masculine_presenting", "feminine_presenting"],
        })

    def test_fallback_target_prefers_the_least_used_voice(self):
        participants = [
            {"target_ref": "masculine_presenting"},
            {"target_ref": "masculine_presenting"},
            {"target_ref": "feminine_presenting"},
        ]
        self.assertEqual(choose_balanced_target(
            ["masculine_presenting", "feminine_presenting"], participants),
            "feminine_presenting",
        )

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

    def test_target_engine_must_match_counterbalanced_schedule(self):
        settings = {"counterbalancing": {
            "target_assignment": {
                "answer_id": "gender_identity",
                "target_by_answer": {"Woman": "masculine_presenting"},
            },
            "conditions": {
                "converted": {"voice_schedule": [{
                    "mode": "vc", "engine": "xvc", "start_s": 0, "end_s": None,
                }]},
            },
            "variants": [{
                "id": "v1", "scenario_order": [1, 2],
                "condition_assignment": {1: "converted", 2: "converted"},
            }],
        }}
        targets = [{"ref": "masculine_presenting", "engine": "meanvc"}]
        with self.assertRaisesRegex(CounterbalanceError, "requires xvc"):
            validate_and_compile(settings, self.scenarios, targets)


class PilotTemplateTests(unittest.TestCase):
    def test_protocol_template_balances_scenarios_conditions_and_positions(self):
        path = Path(__file__).resolve().parents[1] / "study" / "templates" / "pilot_study.yaml"
        protocol = yaml.safe_load(path.read_text())
        scenarios = [{**scenario, "id": index + 1, "order_idx": index}
                     for index, scenario in enumerate(protocol["scenarios"])]
        targets = [{"ref": target["ref"]} for target in protocol["targets"]]
        settings = {"counterbalancing": protocol["counterbalancing"]}

        compiled = validate_and_compile(settings, scenarios, targets)
        self.assertEqual(len(compiled), 4)
        analytical_conditions = {
            "stable_natural", "stable_converted", "vc_activation", "vc_deactivation",
        }
        for scenario_id in range(2, 6):
            self.assertEqual(
                {variant["assignment"][str(scenario_id)]["condition"] for variant in compiled},
                analytical_conditions,
            )
            self.assertEqual(
                {variant["scenario_order"].index(scenario_id) + 1 for variant in compiled},
                {2, 3, 4, 5},
            )
        for condition in analytical_conditions:
            positions = {
                variant["scenario_order"].index(int(scenario_id)) + 1
                for variant in compiled
                for scenario_id, assignment in variant["assignment"].items()
                if assignment["condition"] == condition
            }
            self.assertEqual(positions, {2, 3, 4, 5})
        for variant in compiled:
            self.assertEqual(variant["scenario_order"][0], 1)
            self.assertEqual(variant["assignment"]["1"]["condition"], "practice")

        for condition in ("vc_activation", "vc_deactivation"):
            schedule = protocol["counterbalancing"]["conditions"][condition]["voice_schedule"]
            self.assertEqual(schedule[0]["end_s"], 45)
            self.assertEqual(schedule[1]["start_s"], 45)

        self.assertEqual(protocol["scenarios"][0]["scenario_card"]["study_role"], "practice")
        self.assertTrue({"eligibility", "consent", "background", "audio_check", "post",
                         "pre_playback", "playback", "debrief"}.issubset(
                            protocol["questionnaires"]))
        self.assertTrue(all(
            scenario["post_items"][0].get("insert_after") == "outcome_confidence"
            for scenario in protocol["scenarios"][1:]
        ))


class QuestionnaireTests(unittest.TestCase):
    def test_hidden_required_branch_is_not_required(self):
        items = [
            {"id": "ended", "type": "radio", "required": True},
            {"id": "reason", "type": "radio", "required": True,
             "show_if": {"field": "ended", "in": ["Yes"]}},
        ]
        self.assertEqual(missing_required_answers(items, {"ended": "No"}), [])
        self.assertEqual(missing_required_answers(items, {"ended": "Yes"}), ["reason"])

    def test_checkbox_controller_uses_any_matching_answer(self):
        items = [{"id": "details", "type": "text", "required": True,
                  "show_if": {"field": "choices", "in": ["Other"]}}]
        self.assertEqual(
            missing_required_answers(items, {"choices": ["A", "Other"]}),
            ["details"],
        )


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

    def test_vc_scoring_concatenates_recorded_participant_speech(self):
        events = [
            {"event": "route_activated", "event_sequence": 1, "from_mode": None,
             "to_mode": "vc", "input_sample": 0, "transmitted_sample": 0,
             "requested_start_s": 0},
            {"event": "transmitted_window", "event_sequence": 2, "route_mode": "vc",
             "input_start_sample": 0, "input_end_sample": 48000,
             "transmitted_start_sample": 0, "transmitted_end_sample": 48000},
            {"event": "participant_speech_start", "event_sequence": 3,
             "input_sample": 16000},
            {"event": "participant_speech_end", "event_sequence": 4,
             "input_sample": 24000},
            {"event": "participant_speech_start", "event_sequence": 5,
             "input_sample": 32000},
            {"event": "participant_speech_end", "event_sequence": 6,
             "input_sample": 40000},
            {"event": "stream_stop", "event_sequence": 7, "input_samples": 48000},
        ]
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

            result = prepare_session_analysis(session, root, "analysis-speech")

            region = result["regions"][0]
            self.assertEqual(region["score_selection"]["speech_intervals"], 2)
            self.assertIn("_speech.wav", result["score_jobs"][0]["converted"])
            scored_clip = root / result["score_jobs"][0]["converted"]
            with wave.open(str(scored_clip), "rb") as wav:
                # Two padded 0.9 s utterances plus the 0.15 s separator.
                self.assertEqual(wav.getnframes(), 31200)

    def test_playback_clip_contains_converted_then_natural_speech(self):
        events = [
            {"event": "route_activated", "event_sequence": 1, "from_mode": None,
             "to_mode": "vc", "input_sample": 0, "transmitted_sample": 0},
            {"event": "transmitted_window", "event_sequence": 2, "route_mode": "vc",
             "input_start_sample": 0, "input_end_sample": 24000,
             "transmitted_start_sample": 0, "transmitted_end_sample": 24000},
            {"event": "route_activated", "event_sequence": 3, "from_mode": "vc",
             "to_mode": "natural", "input_sample": 24000,
             "transmitted_sample": 24000},
            {"event": "transmitted_window", "event_sequence": 4, "route_mode": "natural",
             "input_start_sample": 24000, "input_end_sample": 48000,
             "transmitted_start_sample": 24000, "transmitted_end_sample": 48000},
            {"event": "stream_stop", "event_sequence": 5, "input_samples": 48000},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session_dir = root / "sessions" / "attempt"
            session_dir.mkdir(parents=True)
            write_wav(session_dir / "participant.wav", seconds=3.0)
            with (session_dir / "events.jsonl").open("w") as stream:
                for row in events:
                    stream.write(json.dumps(row) + "\n")
            session = {"session_id": "s1", "files": {
                "participant": "sessions/attempt/participant.wav",
            }}
            output, manifest = ensure_transition_playback(session, root, 2)
            self.assertTrue(output.exists())
            self.assertGreater(manifest["converted_speech_s"], 0)
            self.assertGreater(manifest["natural_speech_s"], 0)
            with wave.open(str(output), "rb") as wav:
                self.assertLessEqual(wav.getnframes(), 2 * wav.getframerate())
            repeated, repeated_manifest = ensure_transition_playback(session, root, 2)
            self.assertEqual(repeated, output)
            self.assertEqual(repeated_manifest["output"]["sha256"], manifest["output"]["sha256"])

    def test_stable_converted_playback_is_speech_only_and_duration_limited(self):
        events = [
            {"event": "route_activated", "event_sequence": 1,
             "from_mode": None, "to_mode": "vc", "input_sample": 0,
             "transmitted_sample": 0},
            {"event": "transmitted_window", "event_sequence": 2,
             "route_mode": "vc", "input_start_sample": 0,
             "input_end_sample": 48000, "transmitted_start_sample": 0,
             "transmitted_end_sample": 48000},
            {"event": "stream_stop", "event_sequence": 3,
             "input_samples": 48000},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session_dir = root / "sessions" / "attempt"
            session_dir.mkdir(parents=True)
            write_wav(session_dir / "participant.wav", seconds=3.0)
            with (session_dir / "events.jsonl").open("w") as stream:
                for row in events:
                    stream.write(json.dumps(row) + "\n")
            session = {"session_id": "s1", "files": {
                "participant": "sessions/attempt/participant.wav",
            }}
            output, manifest = ensure_stable_converted_playback(
                session, root, max_duration_s=2)
            self.assertEqual(manifest["selection"],
                             "rms_speech_from_stable_converted")
            with wave.open(str(output), "rb") as wav:
                self.assertLessEqual(wav.getnframes(), 2 * wav.getframerate())


class TimingAnalysisTests(unittest.TestCase):
    def test_speech_playback_and_manual_validation_are_explicit_estimates(self):
        audio = np.zeros(16000, dtype=np.float32)
        audio[4000:8000] = 0.1
        participant = speech_intervals(audio, 16000, frame_ms=20, hangover_ms=0)
        assistant = assistant_intervals({
            "output_latency_ms": 20,
            "assistant_packets": [
                {"packet_sequence": 1, "timeline_start_ms": 200,
                 "timeline_end_ms": 300, "rms": 0.1},
                {"packet_sequence": 2, "timeline_start_ms": 310,
                 "timeline_end_ms": 400, "rms": 0.1},
            ],
        })
        self.assertEqual(participant[0]["start_ms"], 240)
        self.assertEqual(assistant[0]["start_ms"], 220)
        validation = validate_intervals(participant, [{"start_ms": 250, "end_ms": 500}])
        self.assertEqual(validation["precision"], 1.0)
        self.assertFalse(validation["validated"])

    def test_timing_snapshot_crosswalks_browser_and_proxy_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session_dir = root / "sessions" / "attempt"
            session_dir.mkdir(parents=True)
            raw = np.zeros(16000, dtype="<i2")
            raw[9600:12800] = 8000
            with wave.open(str(session_dir / "participant_raw.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(raw.tobytes())
            timeline = {
                "schema": "hmo.client-timeline.v1",
                "capture": {
                    "estimated_dropped_samples": 0,
                    "chunks": [
                        {"chunk_sequence": 1, "capture_start_sample": 0,
                         "sample_count": 2048, "timeline_start_ms": 0},
                        {"chunk_sequence": 2, "capture_start_sample": 2048,
                         "sample_count": 2048, "timeline_start_ms": 128},
                    ],
                },
                "playback": {
                    "schema": "hmo.client-playback-timeline.v2",
                    "decode_strategy": "serialized",
                    "initial_jitter_buffer_ms": 120,
                    "queue_underrun_count": 0,
                    "queue_underrun_total_ms": 0,
                    "queue_underrun_max_ms": 0,
                    "output_latency_ms": 0,
                    "assistant_packets": [{
                        "packet_sequence": 1, "timeline_start_ms": 500,
                        "timeline_end_ms": 1000, "rms": 0.1,
                    }],
                },
            }
            (session_dir / "client_timeline.json").write_text(json.dumps(timeline))
            events = [
                {"event": "input_chunk", "event_sequence": 1,
                 "input_start_sample": 0, "input_end_sample": 2048,
                 "browser_chunk_sequence": 1, "capture_sample_rate_hz": 16000},
                {"event": "input_chunk", "event_sequence": 2,
                 "input_start_sample": 2048, "input_end_sample": 4096,
                 "browser_chunk_sequence": 2, "capture_sample_rate_hz": 16000},
                {"event": "route_activated", "event_sequence": 3,
                 "from_mode": "natural", "to_mode": "vc", "input_sample": 2048},
                {"event": "personaplex_output_packet", "event_sequence": 4,
                 "packet_sequence": 1, "tag": 1, "payload_bytes": 100},
            ]
            with (session_dir / "events.jsonl").open("w") as stream:
                for event in events:
                    stream.write(json.dumps(event) + "\n")
            session = {"session_id": "s1", "files": {
                "participant_raw": "sessions/attempt/participant_raw.wav",
            }}
            result = prepare_timing_analysis(session, root, "analysis-1")
            self.assertTrue(result["integrity"]["crosswalk_complete"])
            self.assertEqual(
                result["route_switches"][0]["browser_chunk_sequence"], 2)
            self.assertEqual(
                result["route_switches"][0]["participant_timeline_ms"], 128)
            self.assertEqual(result["summary"]["barge_in_attempts"], 1)
            self.assertEqual(
                result["integrity"]["playback"]["decode_strategy"], "serialized")
            self.assertEqual(
                result["integrity"]["playback"]["initial_jitter_buffer_ms"], 120)
            self.assertEqual(result["schema"], "hmo.timing-analysis.v2")
            self.assertEqual(result["status"], "estimated_pending_validation")
            self.assertTrue((session_dir / "analysis" / "timing" / "analysis-1"
                             / "timing.json").exists())

    def test_assistant_silence_packets_do_not_become_speech(self):
        intervals = assistant_intervals({
            "output_latency_ms": 10,
            "assistant_packets": [
                {"packet_sequence": 1, "timeline_start_ms": 0,
                 "timeline_end_ms": 100, "rms": 0.0},
                {"packet_sequence": 2, "timeline_start_ms": 100,
                 "timeline_end_ms": 200, "rms": 0.05},
                {"packet_sequence": 3, "timeline_start_ms": 200,
                 "timeline_end_ms": 300, "rms": 0.0},
            ],
        })
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["start_ms"], 110)
        self.assertEqual(intervals[0]["end_ms"], 210)
        self.assertEqual(intervals[0]["detector"],
                         "decoded_packet_rms_audio_context")

    def test_short_participant_energy_spikes_are_not_barge_in_candidates(self):
        audio = np.zeros(16000, dtype=np.float32)
        audio[4000:4800] = 0.1
        self.assertEqual(speech_intervals(audio, 16000, hangover_ms=0), [])

    def test_existing_capture_uses_model_wav_energy_fallback(self):
        model = np.zeros(16000, dtype=np.float32)
        model[4000:8000] = 0.1
        intervals = assistant_intervals(
            {"output_latency_ms": 20, "assistant_packets": []},
            model_audio=model, model_sample_rate=16000,
        )
        self.assertEqual(intervals[0]["start_ms"], 260)
        self.assertEqual(intervals[0]["detector"],
                         "model_wav_rms_playback_fallback")


class TranscriptTimingTests(unittest.TestCase):
    def test_whisper_timestamp_tokens_become_ordered_segments(self):
        class Tokenizer:
            all_special_ids = [1, 2, 999]

            @staticmethod
            def convert_tokens_to_ids(token):
                return 999 if token == "<|notimestamps|>" else -1

            @staticmethod
            def decode(tokens, skip_special_tokens=True):
                del skip_special_tokens
                return " ".join({10: "hello", 11: "there", 12: "again"}[t]
                                for t in tokens)

        segments = whisper_timestamp_segments(
            [1, 1000, 10, 11, 1050, 1050, 12, 1100, 2], Tokenizer(), 3.0)
        self.assertEqual(segments, [
            {"text": "hello there", "start": 0.0, "end": 1.0},
            {"text": "again", "start": 1.0, "end": 2.0},
        ])

    def test_participant_segments_use_browser_clock_and_route_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            session_dir = Path(temp)
            raw_path = session_dir / "participant_raw.wav"
            write_wav(raw_path, seconds=2.0)
            (session_dir / "client_timeline.json").write_text(json.dumps({
                "capture": {"chunks": [{"timeline_start_ms": 100}]},
            }))
            events = [
                {"event": "route_activated", "event_sequence": 1,
                 "from_mode": None, "to_mode": "natural", "input_sample": 0},
                {"event": "transmitted_window", "event_sequence": 2,
                 "route_mode": "natural", "input_start_sample": 0,
                 "input_end_sample": 16000, "transmitted_start_sample": 0,
                 "transmitted_end_sample": 16000},
                {"event": "route_activated", "event_sequence": 3,
                 "from_mode": "natural", "to_mode": "vc", "input_sample": 16000},
                {"event": "transmitted_window", "event_sequence": 4,
                 "route_mode": "vc", "input_start_sample": 16000,
                 "input_end_sample": 32000, "transmitted_start_sample": 16000,
                 "transmitted_end_sample": 32000},
                {"event": "stream_stop", "event_sequence": 5,
                 "input_samples": 32000},
            ]
            with (session_dir / "events.jsonl").open("w") as stream:
                for event in events:
                    stream.write(json.dumps(event) + "\n")

            result = _participant_segments([
                {"text": "first", "start": 0.2, "end": 0.4},
                {"text": "second", "start": 1.2, "end": 1.4},
            ], str(raw_path))

        self.assertEqual([item["voice_mode"] for item in result], ["natural", "vc"])
        self.assertEqual(result[0]["start"], 0.3)
        self.assertEqual(result[0]["timeline"], "browser_audio_clock")


if __name__ == "__main__":
    unittest.main()
