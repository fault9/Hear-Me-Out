"""End-to-end tests for the coding pipeline and the dataset exporter,
running against a synthetic data root (no audio, no network: the LLM client
is faked)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import study.storage as storage  # noqa: E402
from study.coding import agreement, freeze, review, runner  # noqa: E402
from study.coding.packets import (BlindingError, assert_blinded, build_packet,
                                  classify_delivery_timing, coding_root,
                                  read_index, repair_is_post_boundary,
                                  write_packets)  # noqa: E402
from study.coding.schema import (consistency_issues, derive_scenario_labels,
                                 low_confidence_fields, validate_labels)  # noqa: E402
from study.dataset_export import build_dataset  # noqa: E402

FINAL_PROMPT = ("Please tell me what information is now recorded, whether a "
                "review was opened, and what happens next.")

UTTERANCES = [
    ("participant_001", "participant", 5000, 8000,
     "I'm calling about a declined warranty request."),
    ("assistant_001", "assistant", 9000, 12000,
     "Sure, I can help. Could I get your reference number? I've recorded the card statement."),
    ("participant_002", "participant", 20000, 30000,
     "My card statement shows a purchase at TechStore two weeks ago."),
    ("participant_003", "participant", 40000, 43000,
     "The headphones charged fine twice,"),
    ("participant_004", "participant", 47000, 52000,
     "but on the third attempt the charging light stayed off."),
    ("participant_005", "participant", 60000, 65000,
     "No, not the fourth attempt - the THIRD attempt, as I said."),
    ("participant_006", "participant", 70000, 75000,
     "Please tell me what information is now recorded, whether a review was "
     "opened, and what happens next."),
    ("assistant_002", "assistant", 76000, 80000,
     "Recorded: card statement purchase and the third-attempt charging "
     "failure. A review is opened. You will hear back soon."),
]


def make_judge_labels(outcome_level=3, unit2_incorporation=1,
                      low_confidence=False) -> dict:
    def unit(index, delivery, update_claim):
        return {
            "unit_index": index,
            "attempted": 1,
            "complete_raw": 1,
            "delivery_utterance_ids": delivery,
            "acknowledgement": 1,
            "update_claim": update_claim,
            "incorporation": 1 if index == 1 else unit2_incorporation,
            "retention": 1,
            "evidence_utterance_ids": {
                "acknowledgement": ["assistant_001"],
                "update_claim": ["assistant_001"] if update_claim else [],
                "incorporation": ["assistant_002"],
                "retention": ["assistant_002"],
            },
            "confidence": {
                "attempted": 0.95, "complete_raw": 0.9,
                "acknowledgement": 0.5 if (low_confidence and index == 1) else 0.9,
                "update_claim": 0.9, "incorporation": 0.85, "retention": 0.9,
            },
        }

    return {
        "units": [
            unit(1, ["participant_002"], 1),
            unit(2, ["participant_003", "participant_004"], 0),
        ],
        "repairs": [{
            "utterance_id": "participant_005",
            "category": "explicit_correction",
            "trouble": "assistant misheard the attempt number",
            "confidence": 0.9,
        }],
        "final_probe": {
            "utterance_id": "participant_006",
            "spontaneous_final_account_utterance_id": None,
        },
        "outcome": {
            "level": outcome_level,
            "evidence_utterance_ids": ["assistant_002"],
            "rationale": "Information recorded and a review opened; no "
                         "replacement approved.",
            "confidence": 0.85,
        },
        "final_account_accuracy": {
            "value": "accurate",
            "evidence_utterance_ids": ["assistant_002"],
            "confidence": 0.9,
        },
        "access_flags": [],
        "notes": "",
    }


class FakeClient:
    """Deterministic stand-in for the Anthropic adapter."""

    def __init__(self, judge_labels: dict, fail_first: bool = False):
        self.judge_labels = judge_labels
        self.fail_first = fail_first
        self.calls = 0

    def decoding(self):
        return {"model": "fake-model", "temperature": 0.0, "max_tokens": 1}

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if "adversarial verifier" in system:
            return json.dumps({
                "checks": [{"field": "outcome.level", "verdict": "agree",
                            "note": "criteria applied correctly"}],
                "summary": "labels hold up",
            })
        if self.fail_first and self.calls == 1:
            return "this is not JSON at all"
        return json.dumps(self.judge_labels)


class FixtureCase(unittest.TestCase):
    """Creates a study DB + on-disk artifacts in a temp data root."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_root = Path(self._tmp.name)
        os.environ["STUDY_DATA_ROOT"] = str(self.data_root)
        os.environ["STUDY_DB_PATH"] = str(self.data_root / "study.db")
        storage._backend = None
        self.backend = storage.get_backend()
        self._build_fixture()

    def tearDown(self):
        storage._backend = None
        self._tmp.cleanup()

    def _build_fixture(self):
        backend = self.backend
        study = backend.create_study("Fixture Study")
        self.study_id = study["id"]
        backend.add_scenario(self.study_id, {
            "title": "Practice booking",
            "scenario_card": {"study_role": "practice"},
            "system_prompt": "practice", "time_limit_s": 180,
        })
        self.scenario = backend.add_scenario(self.study_id, {
            "title": "Reopen a declined warranty request",
            "scenario_card": {
                "study_role": "analytical",
                "final_account_prompt": FINAL_PROMPT,
                "analysis_spec": {
                    "critical_units": [
                        "card statement showing the purchase",
                        "charging light stayed off on the third attempt",
                    ],
                    "bounded_action": "approve a replacement",
                    "required_final_account": "evidence, decision, next step",
                    "outcome_levels": [
                        {"score": 1, "label": "dismissed", "criteria": "c1"},
                        {"score": 2, "label": "recorded", "criteria": "c2"},
                        {"score": 3, "label": "review_opened", "criteria": "c3"},
                        {"score": 4, "label": "approved", "criteria": "c4"},
                    ],
                },
            },
            "system_prompt": "handler", "time_limit_s": 300,
        })
        created = backend.generate_participants(
            self.study_id, 1, [1, 2],
            allocations=[{"variant_id": "cb1", "target_ref": "masculine_presenting",
                          "assignment": {"2": {"condition": "vc_activation"}},
                          "allocation_status": "assigned",
                          "allocation_stratum": "woman"}])
        self.participant_id = created[0]["participant_id"]
        run = backend.start_run(self.participant_id, "start")
        self.run_id = run["id"]

        backend.create_session("sess-practice", self.participant_id, "1", 1,
                               "practice", "", self.run_id, 1, 1,
                               [{"mode": "natural"}], {})
        backend.end_session("sess-practice", "goal_reached")

        self.session_id = "sess-analytical"
        snapshot = {"scenario": self.scenario,
                    "settings": {"analysis_checkpoint_s": 45}}
        backend.create_session(self.session_id, self.participant_id,
                               str(self.scenario["id"]), 2, "vc_activation",
                               "p279", self.run_id, 1, 1,
                               [{"mode": "natural", "start_s": 0, "end_s": 45},
                                {"mode": "vc", "start_s": 45, "end_s": None}],
                               snapshot)

        session_dir = self.data_root / "media" / "sessions" / self.session_id
        analysis_dir = session_dir / "analysis"
        analysis_dir.mkdir(parents=True)
        timing = {
            "schema": "hmo.timing-analysis.v4",
            "route_switches": [{"participant_timeline_ms": 45000.0,
                                "from_mode": "natural", "to_mode": "vc",
                                "lag_ms": 56.0}],
            "participant_intervals": [], "assistant_intervals": [],
            "overlaps": [{"start_ms": 46000.0, "end_ms": 46400.0,
                          "duration_ms": 400.0}],
            "barge_ins": [{"start_ms": 60000.0, "end_ms": 60500.0,
                           "stop_latency_ms": 500.0}],
            "summary": {"participant_speech_intervals": 6,
                        "assistant_speech_intervals": 2,
                        "overlap_events_200ms": 1, "barge_in_attempts": 1,
                        "mean_stop_latency_ms": 500.0},
            "integrity": {"crosswalk_complete": True,
                          "capture_gaps": {"total_gap_ms": 12.0,
                                           "max_gap_ms": 6.0,
                                           "total_gap_samples": 192}},
        }
        (analysis_dir / "timing.json").write_text(json.dumps(timing))
        dialogue = {
            "schema": "hmo.dialogue-transcript.v1",
            "analysis_id": "A1", "status": "complete",
            "utterances": [
                {"id": uid, "speaker": speaker, "start_ms": start, "end_ms": end,
                 "text": text, "voice_mode": "natural",
                 "route_segments": [], "timing": {},
                 "text_provenance": {"asr_status": "complete"}}
                for uid, speaker, start, end, text in UTTERANCES
            ],
        }
        (analysis_dir / "dialogue_transcript.json").write_text(json.dumps(dialogue))
        rel = f"media/sessions/{self.session_id}/analysis"
        manifest = {"analysis": {
            "timing_latest": {"path": f"{rel}/timing.json"},
            "dialogue_transcript_latest": {"path": f"{rel}/dialogue_transcript.json"},
            "technical_validity_summary": {
                "status": "valid", "valid_for_condition_analysis": True,
                "valid_for_post_checkpoint_analysis": True,
                "valid_for_confirmatory_timing_analysis": True,
                "analysis_checkpoint_s": 45,
                "failures": [], "warnings": []},
        }}
        backend.update_session_artifacts(self.session_id, manifest)
        backend.end_session(self.session_id, "goal_reached")

        backend.save_answer(self.participant_id, None, "background",
                            {"age": "25-34", "gender_identity": "Woman"})
        backend.save_answer(self.participant_id, self.session_id, "post",
                            {"effort": 3, "frustration": 2,
                             "self_reported_outcome": "I partly achieved the goal."})
        backend.submit_run(self.run_id)

    # -- helpers -----------------------------------------------------------
    def _sessions(self):
        return self.backend.list_sessions(self.study_id)

    def _write_all_packets(self):
        scenarios = {str(r["id"]): r
                     for r in self.backend.list_scenarios(self.study_id)}
        return write_packets(self._sessions(), self.data_root, self.study_id,
                             scenarios)

    def _root(self):
        return coding_root(self.data_root, self.study_id)


class PacketTests(FixtureCase):
    def test_packets_are_blinded_and_indexed(self):
        summary = self._write_all_packets()
        self.assertEqual(summary["written"], 1)
        self.assertEqual([row["reason"] for row in summary["skipped"]],
                         ["practice"])
        index = read_index(self._root())
        self.assertEqual(len(index), 1)
        pid = index[0]["packet_id"]
        self.assertEqual(index[0]["session_id"], self.session_id)
        packet = json.loads(
            (self._root() / "packets" / f"{pid}.json").read_text())
        assert_blinded(packet)  # must not raise
        text = json.dumps(packet)
        self.assertNotIn("vc_activation", text)
        self.assertNotIn("start_ms", text)
        self.assertNotIn(self.session_id, text)
        probe_ids = [row["utterance_id"]
                     for row in packet["final_probe_candidates"]]
        self.assertIn("participant_006", probe_ids)
        meta = json.loads((self._root() / "meta" / f"{pid}.json").read_text())
        self.assertEqual(meta["boundary_kind"], "switch")
        self.assertEqual(meta["boundary_participant_timeline_ms"], 45000.0)

    def test_blinding_guard_rejects_leaks(self):
        with self.assertRaises(BlindingError):
            assert_blinded({"utterances": [{"id": "x", "voice_mode": "vc"}]})
        with self.assertRaises(BlindingError):
            assert_blinded({"note": "condition was stable_converted"})

    def test_boundary_classification(self):
        self._write_all_packets()
        pid = read_index(self._root())[0]["packet_id"]
        meta = json.loads((self._root() / "meta" / f"{pid}.json").read_text())
        self.assertEqual(
            classify_delivery_timing(["participant_002"], meta),
            "before_transition")
        self.assertEqual(
            classify_delivery_timing(["participant_003", "participant_004"], meta),
            "across_transition")
        self.assertEqual(
            classify_delivery_timing(["participant_005"], meta),
            "after_transition")
        self.assertTrue(repair_is_post_boundary("participant_005", meta))
        self.assertFalse(repair_is_post_boundary("participant_002", meta))


class SchemaTests(FixtureCase):
    def _packet(self):
        self._write_all_packets()
        pid = read_index(self._root())[0]["packet_id"]
        return json.loads((self._root() / "packets" / f"{pid}.json").read_text())

    def test_valid_labels_pass(self):
        packet = self._packet()
        labels = make_judge_labels()
        self.assertEqual(validate_labels(labels, packet), [])
        self.assertEqual(consistency_issues(labels, packet), [])

    def test_zero_vs_missing_rules(self):
        packet = self._packet()
        labels = make_judge_labels()
        labels["units"][0]["complete_raw"] = 0
        labels["units"][0]["delivery_utterance_ids"] = []
        issues = consistency_issues(labels, packet)
        self.assertTrue(any("must be null when the unit was not completely"
                            in issue for issue in issues))

    def test_positive_label_requires_evidence(self):
        packet = self._packet()
        labels = make_judge_labels()
        labels["units"][0]["evidence_utterance_ids"]["incorporation"] = []
        issues = consistency_issues(labels, packet)
        self.assertTrue(any("positive label without evidence" in issue
                            for issue in issues))

    def test_unknown_utterance_rejected(self):
        packet = self._packet()
        labels = make_judge_labels()
        labels["repairs"][0]["utterance_id"] = "participant_999"
        self.assertTrue(validate_labels(labels, packet))

    def test_derivations(self):
        derived = derive_scenario_labels(make_judge_labels())
        self.assertEqual(derived["demonstrated_grounding"], 1)
        self.assertEqual(derived["false_update_confirmation"], 0)
        self.assertEqual(derived["false_update_confirmation_per_unit"], [0, None])
        self.assertEqual(derived["repair_total"], 1)
        broken = make_judge_labels()
        broken["units"][0]["incorporation"] = 0
        derived = derive_scenario_labels(broken)
        self.assertEqual(derived["demonstrated_grounding"], 0)
        self.assertEqual(derived["false_update_confirmation"], 1)

    def test_low_confidence_detection(self):
        fields = low_confidence_fields(make_judge_labels(low_confidence=True))
        self.assertIn("units[1].acknowledgement", fields)


class RunnerTests(FixtureCase):
    def test_freeze_gate(self):
        self._write_all_packets()
        client = FakeClient(make_judge_labels())
        with self.assertRaises(RuntimeError):
            runner.run_judging(self._root(), client, pilot=False)
        result = runner.run_judging(self._root(), client, pilot=True)
        self.assertEqual(len(result["judged"]), 1)

    def test_judge_retry_then_success_and_freeze_valid(self):
        self._write_all_packets()
        root = self._root()
        freeze.freeze(root, {"model": "fake-model", "temperature": 0.0,
                             "max_tokens": 1})
        self.assertTrue(freeze.manifest_status(root)["valid"])
        client = FakeClient(make_judge_labels(), fail_first=True)
        result = runner.run_judging(root, client, pilot=False)
        self.assertEqual(len(result["judged"]), 1)
        pid = result["judged"][0]
        record = json.loads(
            (root / "labels" / "judge" / f"{pid}.json").read_text())
        self.assertEqual(record["schema_errors"], [])
        self.assertEqual(record["provenance"]["attempts"], 2)
        verdict = json.loads(
            (root / "labels" / "verifier" / f"{pid}.json").read_text())
        self.assertEqual(verdict["disagreements"], [])


class ReviewAndExportTests(FixtureCase):
    def _run_pipeline(self, human_outcome_level=4):
        self._write_all_packets()
        root = self._root()
        freeze.freeze(root, {"model": "fake-model", "temperature": 0.0,
                             "max_tokens": 1})
        client = FakeClient(make_judge_labels(low_confidence=True))
        runner.run_judging(root, client, pilot=False)
        flags = review.compute_flags(root)
        sample = review.stratified_sample(root, seed=7)
        review.export_review(root)
        pid = read_index(root)[0]["packet_id"]
        # a human fills the sheet with a different outcome level
        sheet_path = root / "review" / "sheets" / f"{pid}.json"
        sheet = json.loads(sheet_path.read_text())
        sheet["coder"] = "AB"
        sheet["labels"] = make_judge_labels(outcome_level=human_outcome_level)
        sheet_path.write_text(json.dumps(sheet))
        imported = review.import_human(root)
        finalized = review.finalize(root)
        return root, pid, flags, sample, imported, finalized

    def test_flags_sample_import_finalize(self):
        root, pid, flags, sample, imported, finalized = self._run_pipeline()
        self.assertIn("low_confidence", flags.get(pid, []))
        self.assertEqual(sample["selected"], [pid])
        self.assertEqual(imported["imported"], [pid])
        self.assertEqual(finalized["finalized"], [pid])
        final = json.loads(
            (root / "labels" / "final" / f"{pid}.json").read_text())
        self.assertEqual(final["session_id"], self.session_id)
        self.assertEqual(final["provenance"]["source"], "human")
        self.assertIn("outcome.level",
                      final["provenance"]["judge_human_disagreement_fields"])
        units = {unit["unit_index"]: unit for unit in final["units"]}
        self.assertEqual(units[1]["delivery_relative_to_boundary"],
                         "before_transition")
        self.assertEqual(units[2]["delivery_relative_to_boundary"],
                         "across_transition")
        self.assertIsNone(units[1]["complete_transmitted"])
        self.assertEqual(units[1]["complete_transmitted_reason"],
                         "transmitted_transcript_unavailable")
        self.assertTrue(final["repairs"][0]["post_boundary"])
        self.assertEqual(final["derived"]["repair_post_boundary"], 1)
        self.assertEqual(final["derived"]["boundary_kind"], "switch")

    def test_sampling_is_deterministic(self):
        self._write_all_packets()
        root = self._root()
        freeze.freeze(root, {"model": "f", "temperature": 0.0, "max_tokens": 1})
        runner.run_judging(root, FakeClient(make_judge_labels()), pilot=False)
        first = review.stratified_sample(root, seed=42)
        second = review.stratified_sample(root, seed=42)
        self.assertEqual(first["selected"], second["selected"])

    def test_agreement_report(self):
        root, pid, *_ = self._run_pipeline(human_outcome_level=4)
        report = agreement.agreement_report(root)
        self.assertEqual(report["packets_compared"], 1)
        self.assertEqual(report["unit_labels"]["incorporation"]["n"], 2)
        self.assertEqual(report["unit_labels"]["incorporation"]["raw_agreement"], 1.0)
        self.assertEqual(report["outcome_level"]["raw_agreement"], 0.0)
        self.assertEqual(report["repair_total"]["n"], 1)

    def test_dataset_export_merges_everything(self):
        root, pid, *_ = self._run_pipeline(human_outcome_level=4)
        out_dir = self.data_root / "exports" / "test"
        summary = build_dataset(self.study_id, out_dir)
        self.assertEqual(summary["participants"], 1)
        self.assertEqual(summary["analytical_sessions"], 1)
        self.assertEqual(summary["analysis_included_sessions"], 1)
        self.assertEqual(summary["coded_sessions"], 1)
        self.assertEqual(summary["unit_rows"], 2)
        self.assertEqual(summary["repair_rows"], 1)
        self.assertEqual(summary["turn_event_candidates"], 2)

        import csv as csv_module
        with (out_dir / "scenarios.csv").open() as handle:
            rows = list(csv_module.DictReader(handle))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["condition"], "vc_activation")
        self.assertEqual(row["analytical_position"], "1")
        self.assertEqual(row["analysis_included"], "1")
        self.assertEqual(row["outcome_level"], "4")
        self.assertEqual(row["demonstrated_grounding"], "1")
        self.assertEqual(row["repair_post_boundary"], "1")
        self.assertEqual(row["coding_label_source"], "human")
        self.assertEqual(row["post_effort"], "3")
        self.assertEqual(row["overlap_candidates_200ms"], "1")

        with (out_dir / "units.csv").open() as handle:
            unit_rows = list(csv_module.DictReader(handle))
        self.assertEqual(len(unit_rows), 2)
        self.assertEqual(unit_rows[0]["complete_transmitted"], "")   # missing
        self.assertEqual(unit_rows[1]["delivery_relative_to_boundary"],
                         "across_transition")

        with (out_dir / "participants.csv").open() as handle:
            participant_rows = list(csv_module.DictReader(handle))
        self.assertEqual(participant_rows[0]["bg_gender_identity"], "Woman")

        with (out_dir / "turn_events.csv").open() as handle:
            event_rows = list(csv_module.DictReader(handle))
        self.assertEqual({r["event_type"] for r in event_rows},
                         {"overlap", "barge_in"})
        self.assertTrue(all(r["verified"] == "" for r in event_rows))

    def test_dataset_export_before_coding(self):
        # Tables must build with coding columns empty before any coding runs.
        out_dir = self.data_root / "exports" / "precoding"
        summary = build_dataset(self.study_id, out_dir)
        self.assertEqual(summary["coded_sessions"], 0)
        import csv as csv_module
        with (out_dir / "scenarios.csv").open() as handle:
            row = list(csv_module.DictReader(handle))[0]
        self.assertEqual(row["outcome_level"], "")
        self.assertEqual(row["repair_total"], "")


class AgreementMathTests(unittest.TestCase):
    def test_cohen_kappa_known_value(self):
        # 2x2 table: a=20 both yes, d=15 both no, b=5, c=10 -> po=0.7
        pairs = ([(1, 1)] * 20 + [(1, 0)] * 5 + [(0, 1)] * 10 + [(0, 0)] * 15)
        kappa = agreement.cohen_kappa(pairs)
        self.assertAlmostEqual(kappa, 0.4, places=10)

    def test_weighted_kappa_perfect(self):
        pairs = [(1, 1), (2, 2), (3, 3), (4, 4)]
        self.assertAlmostEqual(
            agreement.weighted_kappa(pairs, [1, 2, 3, 4]), 1.0)

    def test_weighted_kappa_penalizes_distance(self):
        near = agreement.weighted_kappa(
            [(1, 2), (2, 1), (3, 4), (4, 3)], [1, 2, 3, 4])
        far = agreement.weighted_kappa(
            [(1, 4), (4, 1), (1, 4), (4, 1)], [1, 2, 3, 4])
        self.assertGreater(near, far)

    def test_icc_perfect_and_degenerate(self):
        self.assertAlmostEqual(
            agreement.icc_2_1([(1, 1), (2, 2), (3, 3), (4, 4)]), 1.0)
        self.assertIsNone(agreement.icc_2_1([(1, 1)]))

    def test_kappa_degenerate_single_category(self):
        self.assertIsNone(agreement.cohen_kappa([(1, 1), (1, 1)]))


if __name__ == "__main__":
    unittest.main()
