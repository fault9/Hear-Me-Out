"""End-to-end tests for the coding pipeline and the dataset exporter,
running against a synthetic data root (no audio, no network: the LLM client
is faked)."""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import study.storage as storage  # noqa: E402
from study.artifacts import (load_manifest_artifact,
                             resolve_artifact_path)  # noqa: E402
from study.coding import agreement, freeze, review, runner  # noqa: E402
from study.coding.packets import (BlindingError, assert_blinded, build_packet,
                                  classify_delivery_timing, coding_root,
                                  read_index, repair_is_post_boundary,
                                  write_packets)  # noqa: E402
from study.coding.schema import (consistency_issues, derive_scenario_labels,
                                 low_confidence_fields, validate_checks,
                                 validate_labels)  # noqa: E402
from study.dataset_export import build_dataset  # noqa: E402
from study.turn_verification import finalize as finalize_turn_verification  # noqa: E402

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

    def __init__(self, judge_labels: dict, fail_first: bool = False,
                 verifier_payload: dict | None = None):
        self.judge_labels = judge_labels
        self.fail_first = fail_first
        self.verifier_payload = verifier_payload or {
            "checks": [{"field": "outcome.level", "verdict": "agree",
                        "note": "criteria applied correctly"}],
            "summary": "labels hold up",
        }
        self.calls = 0

    def decoding(self):
        return {"model": "fake-model", "temperature": 0.0, "max_tokens": 1}

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if "adversarial verifier" in system:
            return json.dumps(self.verifier_payload)
        if self.fail_first and self.calls == 1:
            return "this is not JSON at all"
        return json.dumps(self.judge_labels)


class FixtureCase(unittest.TestCase):
    """Creates a study DB + on-disk artifacts in a temp data root."""

    # Production manifests record artifact paths relative to the media dir
    # (file_record(relative_to=STUDY_DATA_DIR)), e.g. "sessions/...". Older
    # fixtures wrote "media/sessions/..."; both must resolve, so the default
    # keeps the legacy form and ProductionLayoutTests flips it.
    manifest_paths_relative_to_media = False

    def _manifest_rel(self) -> str:
        prefix = "" if self.manifest_paths_relative_to_media else "media/"
        return f"{prefix}sessions/{self.session_id}/analysis"

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
            # Existing production artifacts are v4. The exporter must derive
            # linked directional episodes from these retained intervals.
            "participant_intervals": [
                {"start_ms": 1000.0, "end_ms": 2000.0},
                {"start_ms": 4000.0, "end_ms": 5000.0},
                {"start_ms": 7000.0, "end_ms": 7500.0},
            ],
            "assistant_intervals": [
                {"start_ms": 500.0, "end_ms": 1500.0},
                {"start_ms": 4500.0, "end_ms": 5500.0},
                # A 50 ms sliver onto the participant's last turn: a directional
                # candidate below the 200 ms overlap floor, so it must reach
                # turn_events.csv but never the verification queue.
                {"start_ms": 7450.0, "end_ms": 8000.0},
            ],
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
        self._write_transmitted_artifact()   # faithful by default
        rel = self._manifest_rel()
        manifest = {"analysis": {
            "timing_latest": {"path": f"{rel}/timing.json"},
            "dialogue_transcript_latest": {"path": f"{rel}/dialogue_transcript.json"},
            "transmitted_transcript_latest": {"path": f"{rel}/transmitted_transcript.json"},
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

    def _write_transmitted_artifact(self, overrides: dict | None = None,
                                    statuses: dict | None = None):
        """Transmitted-track transcript with per-utterance text overrides."""
        analysis_dir = (self.data_root / "media" / "sessions"
                        / self.session_id / "analysis")
        transmitted = {
            "schema": "hmo.transmitted-transcript.v1",
            "analysis_id": "T1", "status": "complete",
            "alignment": "assumed_chunk_aligned",
            "utterances": [
                {"id": uid, "speaker": "participant",
                 "start_ms": start, "end_ms": end,
                 "text": (overrides or {}).get(uid, text),
                 "text_provenance": {
                     "asr_status": (statuses or {}).get(uid, "complete")}}
                for uid, speaker, start, end, text in UTTERANCES
                if speaker == "participant"
            ],
        }
        (analysis_dir / "transmitted_transcript.json").write_text(
            json.dumps(transmitted))

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
        # The method supplies no final-readback markers: the judge locates the
        # final-summary request itself from the transcript.
        self.assertNotIn("final_probe_candidates", packet)
        # Transmitted speech travels under the same utterance ids, and the
        # assistant's system prompt is part of the scenario specification.
        self.assertEqual([row["id"] for row in packet["transmitted_utterances"]],
                         [row["id"] for row in packet["utterances"]
                          if row["speaker"] == "participant"])
        self.assertTrue(packet["scenario"]["system_prompt"])
        meta = json.loads((self._root() / "meta" / f"{pid}.json").read_text())
        self.assertEqual(meta["boundary_kind"], "switch")
        self.assertEqual(meta["boundary_participant_timeline_ms"], 45000.0)

    def test_explicitly_excluded_session_never_becomes_a_coding_packet(self):
        sessions = self._sessions()
        for session in sessions:
            session["analysis_included"] = False
            session["analysis_exclusion_reasons"] = ["technical:route_mismatch"]
        scenarios = {str(r["id"]): r
                     for r in self.backend.list_scenarios(self.study_id)}

        summary = write_packets(
            sessions, self.data_root, self.study_id, scenarios)

        self.assertEqual(summary["written"], 0)
        self.assertTrue(all(row["reason"].startswith("analysis_excluded:")
                            for row in summary["skipped"]))

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

    def test_verifier_verdicts_must_be_readable(self):
        self.assertEqual(validate_checks(
            {"checks": [{"field": "outcome.level", "verdict": "disagree",
                         "note": "level 4 needs the bounded action"}]}), [])
        self.assertTrue(validate_checks({"checks": []}))
        self.assertTrue(validate_checks(
            {"checks": [{"field": "outcome.level", "verdict": "Disagree",
                         "note": "n"}]}))
        self.assertTrue(validate_checks(
            {"checks": [{"field": "outcome.level", "verdict": "uncertain",
                         "note": ""}]}))

    def test_low_confidence_detection(self):
        fields = low_confidence_fields(make_judge_labels(low_confidence=True))
        self.assertIn("units[1].acknowledgement", fields)


class TransportRetryTests(unittest.TestCase):
    """The endpoint client talks to the provider over the standard library, so
    it absorbs rate limits and gateway blips itself."""

    @staticmethod
    def _http_error(code: int, body: bytes = b"", headers: dict | None = None):
        return urllib.error.HTTPError(
            "https://endpoint/v1/chat/completions", code, "err",
            headers or {}, io.BytesIO(body))

    def test_rate_limit_is_retried_then_succeeds(self):
        attempts = []

        def send(payload):
            attempts.append(payload)
            if len(attempts) < 3:
                raise self._http_error(429, headers={"Retry-After": "0"})
            return {"ok": True}

        with mock.patch.object(runner.time, "sleep") as sleep:
            self.assertEqual(runner.request_with_retry(send, {"m": 1}), {"ok": True})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)

    def test_rejected_parameter_surfaces_the_endpoint_body(self):
        def send(payload):
            raise self._http_error(400, b'{"error":"response_format unsupported"}')

        with self.assertRaises(RuntimeError) as caught:
            runner.request_with_retry(send, {})
        self.assertIn("response_format unsupported", str(caught.exception))


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

    def test_resume_retries_failures_and_fills_a_missing_verifier(self):
        self._write_all_packets()
        root = self._root()
        freeze.freeze(root, {"model": "fake-model", "temperature": 0.0,
                             "max_tokens": 1})

        class Broken(FakeClient):
            def complete(self, system: str, user: str) -> str:
                return "not JSON"

        self.assertTrue(runner.run_judging(
            root, Broken(make_judge_labels()), pilot=False)["schema_failed"])

        # A stored failure is retried rather than counted as coded.
        result = runner.run_judging(root, FakeClient(make_judge_labels()),
                                    pilot=False)
        self.assertEqual(len(result["judged"]), 1)
        self.assertEqual(result["skipped_existing"], [])

        # A missing verifier record is filled without a second judge pass.
        pid = result["judged"][0]
        (root / "labels" / "verifier" / f"{pid}.json").unlink()
        client = FakeClient(make_judge_labels())
        self.assertEqual(
            len(runner.run_judging(root, client, pilot=False)["judged"]), 1)
        self.assertEqual(client.calls, 1)

    def test_unreadable_verifier_verdict_is_flagged_not_silent(self):
        """A verdict spelled outside the enum yields no disagreements, which
        would otherwise be indistinguishable from the verifier agreeing."""
        self._write_all_packets()
        root = self._root()
        freeze.freeze(root, {"model": "fake-model", "temperature": 0.0,
                             "max_tokens": 1})
        client = FakeClient(make_judge_labels(), verifier_payload={
            "checks": [{"field": "outcome.level", "verdict": "Disagree",
                        "note": "level 4 needs the bounded action"}],
            "summary": "one label does not hold"})
        pid = runner.run_judging(root, client, pilot=False)["judged"][0]
        verdict = json.loads(
            (root / "labels" / "verifier" / f"{pid}.json").read_text())
        self.assertTrue(verdict["schema_errors"])
        self.assertEqual(verdict["disagreements"], [])
        self.assertIn("verifier_schema_invalid", review.compute_flags(root)[pid])


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
        self.assertEqual(units[1]["complete_transmitted"], 1)
        self.assertEqual(units[1]["transmitted_content_recall"], 1.0)
        self.assertIsNone(units[1]["complete_transmitted_reason"])
        self.assertEqual(final["derived"]["demonstrated_grounding"], 1)
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
        self.assertIn("outcome_level", report["reliability"]["failed_fields"])

    def test_failed_reliability_queues_every_remaining_valid_packet(self):
        self._write_all_packets()
        root = self._root()
        freeze.freeze(root, {"model": "fake", "temperature": 0.0,
                             "max_tokens": 1})
        runner.run_judging(root, FakeClient(make_judge_labels()), pilot=False)
        report = agreement.agreement_report(root)

        expansion = review.expand_for_reliability(root)
        exported = review.export_review(root)

        self.assertTrue(report["reliability"]["requires_full_human_review"])
        self.assertEqual(expansion["packet_ids"],
                         [read_index(root)[0]["packet_id"]])
        self.assertEqual(exported["queued"], 1)

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
        self.assertEqual(summary["turn_event_candidates"], 3)
        self.assertEqual(summary["turn_verification_candidates"], 2)
        self.assertEqual(summary["turn_gap_verification_candidates"], 1)
        self.assertEqual(summary["turn_sessions_requiring_full_review"], 1)
        self.assertEqual(summary["overlap_200ms_candidates"], 2)
        self.assertEqual(summary["participant_barge_in_candidates"], 1)
        self.assertEqual(summary["assistant_premature_onset_candidates"], 2)
        self.assertEqual(summary["positive_response_gap_candidates"], 1)

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
        self.assertEqual(row["overlap_candidates_200ms"], "2")
        self.assertEqual(row["participant_barge_in_candidates"], "1")
        self.assertEqual(row["assistant_premature_onset_candidates"], "2")
        self.assertEqual(row["assistant_response_gap_candidates"], "0")
        self.assertEqual(row["participant_response_gap_candidates"], "1")

        with (out_dir / "units.csv").open() as handle:
            unit_rows = list(csv_module.DictReader(handle))
        self.assertEqual(len(unit_rows), 2)
        self.assertEqual(unit_rows[0]["complete_transmitted"], "1")
        self.assertEqual(unit_rows[0]["transmitted_content_recall"], "1.0")
        self.assertEqual(unit_rows[1]["delivery_relative_to_boundary"],
                         "across_transition")

        with (out_dir / "participants.csv").open() as handle:
            participant_rows = list(csv_module.DictReader(handle))
        self.assertEqual(participant_rows[0]["bg_gender_identity"], "Woman")

        with (out_dir / "turn_events.csv").open() as handle:
            event_rows = list(csv_module.DictReader(handle))
        self.assertEqual(len(event_rows), 3)
        self.assertEqual(
            sum(r["participant_barge_in_candidate"] == "1" for r in event_rows),
            1,
        )
        self.assertEqual(
            sum(r["assistant_premature_onset_candidate"] == "1"
                for r in event_rows),
            2,
        )
        self.assertTrue(all(r["verified_overlap"] == "" for r in event_rows))
        self.assertTrue(all(r["successful_assistant_yielding"] == ""
                            for r in event_rows))
        self.assertTrue(all(r["disruptive_assistant_interruption"] == ""
                            for r in event_rows))

        with (out_dir / "turn_verification_queue.csv").open() as handle:
            verification_rows = list(csv_module.DictReader(handle))
        self.assertEqual(len(verification_rows), 2)
        # The sliver is an audit row, not a listening task.
        sliver = [r for r in event_rows
                  if float(r["overlap_duration_ms"]) < 200]
        self.assertEqual(len(sliver), 1)
        self.assertNotIn(sliver[0]["event_key"],
                         {r["event_key"] for r in verification_rows})

        with (out_dir / "turn_gaps.csv").open() as handle:
            gap_rows = list(csv_module.DictReader(handle))
        self.assertEqual(
            {r["direction"] for r in gap_rows},
            {"assistant_to_participant"},
        )

    def test_turn_verification_finalizes_only_completed_manual_review(self):
        out_dir = self.data_root / "exports" / "turn-finalize"
        build_dataset(self.study_id, out_dir)

        def edit(name, update):
            path = out_dir / name
            with path.open() as handle:
                reader = csv.DictReader(handle)
                fields, rows = list(reader.fieldnames or []), list(reader)
            for row in rows:
                update(row)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        def verify_event(row):
            row["verifier_initials"] = "AB"
            if row["overlap_200ms_candidate"] == "1":
                row["verified_overlap"] = "yes"
            if row["participant_barge_in_candidate"] == "1":
                row["verified_participant_barge_in"] = "yes"
                row["successful_assistant_yielding"] = "yes"
            if row["assistant_premature_onset_candidate"] == "1":
                row["verified_assistant_premature_onset"] = "yes"
                row["disruptive_assistant_interruption"] = "no"

        edit("turn_verification_queue.csv", verify_event)
        edit("turn_gap_verification_queue.csv", lambda row: row.update({
            "verified_positive_gap": "yes", "verifier_initials": "AB",
        }))
        edit("turn_session_review_queue.csv", lambda row: row.update({
            "full_session_reviewed": "yes", "additional_event_count": "0",
            "verifier_initials": "AB",
        }))

        manifest = finalize_turn_verification(out_dir)

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["verified_events"], 2)
        self.assertEqual(manifest["verified_gaps"], 1)
        with (out_dir / "turn_session_summary_verified.csv").open() as handle:
            summary = list(csv.DictReader(handle))[0]
        self.assertEqual(summary["participant_barge_in_count"], "1")
        self.assertEqual(summary["assistant_premature_onset_count"], "1")

    def test_turn_verification_rejects_incomplete_full_track_review(self):
        out_dir = self.data_root / "exports" / "turn-incomplete"
        build_dataset(self.study_id, out_dir)

        result = finalize_turn_verification(out_dir)

        self.assertEqual(result["status"], "invalid_or_incomplete")
        self.assertTrue(result["errors"])
        self.assertFalse((out_dir / "turn_events_verified.csv").exists())

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

    def test_dataset_export_preserves_vc_quality_scores_and_failures(self):
        self.backend.update_session_vc_quality(self.session_id, "complete", {
            "status": "complete",
            "analysis_id": "VCQ1",
            "metric_profile": "xvc_objective_v2",
            "inputs": {
                "source_audio": {"sample_rate_hz": 16000},
                "target_audio": {
                    "path": "targets/p279.wav", "sha256": "target-hash",
                    "duration_s": 10.0,
                },
                "regions": [{
                    "index": 2, "mode": "vc", "input_start_sample": 720000,
                    "input_end_sample": 1440000,
                    "transmitted_start_sample": 719000,
                    "transmitted_end_sample": 1439000, "windows": 300,
                    "stable_guard_s": 0.5,
                    "score_source": {
                        "path": "source-speech.wav", "sha256": "source-hash",
                        "duration_s": 12.5,
                    },
                    "score_transmitted": {
                        "path": "converted-speech.wav", "sha256": "converted-hash",
                        "duration_s": 12.5,
                    },
                    "score_selection": {
                        "mode": "participant_rms_speech_concatenation",
                        "speech_intervals": 4, "boundary_padding_s": 0.2,
                    },
                }],
            },
            "scores": [{
                "region": 2,
                "metrics": {
                    "wer": 0.1, "wer_status": "complete",
                    "ref_kind": "source_asr_free_speech",
                    "sim": 0.61, "sim_status": "complete",
                    "utmos": 3.17, "utmos_status": "complete",
                },
            }],
            "unavailable_metrics": [],
            "result_artifact": {
                "path": "sessions/sess-analytical/analysis/vc_quality/VCQ1/results.json",
                "sha256": "result-hash",
            },
        })
        complete_dir = self.data_root / "exports" / "vc-complete"
        summary = build_dataset(self.study_id, complete_dir)

        with (complete_dir / "vc_quality_regions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["target_ref"], "masculine_presenting")
        self.assertEqual(row["target_speaker_id"], "p279")
        self.assertEqual(row["region_index"], "2")
        self.assertEqual(row["input_start_s"], "45.0")
        self.assertEqual(row["score_selection"],
                         "participant_rms_speech_concatenation")
        self.assertEqual(row["wer_reference_kind"], "source_asr_free_speech")
        self.assertEqual(row["wer"], "0.1")
        self.assertEqual(row["sim"], "0.61")
        self.assertEqual(row["utmos"], "3.17")
        self.assertEqual(summary["vc_quality_region_rows"], 1)
        self.assertEqual(summary["vc_quality_complete_region_rows"], 1)
        self.assertEqual(summary["vc_quality_included_incomplete_sessions"], 0)

        self.backend.update_session_vc_quality(self.session_id, "failed", {
            "status": "failed", "analysis_id": "VCQ2",
            "error": "RuntimeError: scorer unavailable",
        })
        failed_dir = self.data_root / "exports" / "vc-failed"
        failed_summary = build_dataset(self.study_id, failed_dir)
        with (failed_dir / "vc_quality_regions.csv").open() as handle:
            failed_rows = list(csv.DictReader(handle))
        self.assertEqual(len(failed_rows), 1)
        self.assertEqual(failed_rows[0]["vc_quality_result_status"], "failed")
        self.assertEqual(failed_rows[0]["session_error"],
                         "RuntimeError: scorer unavailable")
        self.assertEqual(failed_rows[0]["wer"], "")
        self.assertEqual(failed_summary["vc_quality_complete_region_rows"], 0)
        self.assertEqual(
            failed_summary["vc_quality_included_incomplete_sessions"], 1)

    def test_verification_queue_excludes_technically_invalid_session(self):
        session = self.backend.get_session(self.session_id)
        manifest = session["artifact_manifest"]
        validity = manifest["analysis"]["technical_validity_summary"]
        validity.update({
            "status": "invalid",
            "valid_for_condition_analysis": False,
            "valid_for_confirmatory_timing_analysis": False,
            "failures": [{"code": "capture_crosswalk"}],
        })
        self.backend.update_session_artifacts(self.session_id, manifest)

        out_dir = self.data_root / "exports" / "invalid-timing"
        summary = build_dataset(self.study_id, out_dir)

        self.assertEqual(summary["turn_event_candidates"], 3)
        self.assertEqual(summary["turn_verification_candidates"], 0)
        with (out_dir / "turn_events.csv").open() as handle:
            event_rows = list(csv.DictReader(handle))
        with (out_dir / "turn_verification_queue.csv").open() as handle:
            verification_rows = list(csv.DictReader(handle))
        self.assertTrue(event_rows)
        self.assertEqual(verification_rows, [])


class TransmittedGatingTests(FixtureCase):
    """The mechanical complete_transmitted rule and its downstream gating."""

    def _judge_and_finalize(self):
        self._write_all_packets()
        root = self._root()
        freeze.freeze(root, {"model": "f", "temperature": 0.0, "max_tokens": 1})
        runner.run_judging(root, FakeClient(make_judge_labels()), pilot=False)
        review.finalize(root)
        pid = read_index(root)[0]["packet_id"]
        return json.loads((root / "labels" / "final" / f"{pid}.json").read_text())

    def test_degraded_transmission_zeroes_unit_and_gates_grounding(self):
        # Unit 2's second delivery utterance arrives garbled on the
        # transmitted track (complete ASR, wrong content) -> recall below
        # threshold -> observed 0 -> grounding stages become missing.
        self._write_transmitted_artifact(
            overrides={"participant_004": "hmm buzzing noise"})
        final = self._judge_and_finalize()
        units = {unit["unit_index"]: unit for unit in final["units"]}
        self.assertEqual(units[2]["complete_transmitted"], 0)
        self.assertLess(units[2]["transmitted_content_recall"], 0.6)
        self.assertEqual(units[2]["grounding_gated_reason"],
                         "unit_not_completely_transmitted")
        for stage in ("acknowledgement", "update_claim", "incorporation",
                      "retention"):
            self.assertIsNone(units[2][stage])
        # Unit 1 transmitted faithfully -> untouched.
        self.assertEqual(units[1]["complete_transmitted"], 1)
        self.assertEqual(units[1]["incorporation"], 1)
        # Scenario-level demonstrated grounding: unit 2 unobservable, no
        # observed failure -> missing, not 0.
        self.assertIsNone(final["derived"]["demonstrated_grounding"])

    def test_incomplete_transmitted_asr_is_missing_not_zero(self):
        self._write_transmitted_artifact(
            overrides={"participant_004": ""},
            statuses={"participant_004": "failed"})
        final = self._judge_and_finalize()
        units = {unit["unit_index"]: unit for unit in final["units"]}
        self.assertIsNone(units[2]["complete_transmitted"])
        self.assertEqual(units[2]["complete_transmitted_reason"],
                         "transmitted_asr_incomplete")
        # Proxy gating keeps the raw-coded stages.
        self.assertEqual(units[2]["incorporation"], 1)
        self.assertEqual(units[2]["transmitted_proxy"], "raw")

    def test_unavailable_transcript_falls_back_to_raw_proxy(self):
        # Remove the manifest key entirely.
        session = self.backend.get_session(self.session_id)
        manifest = session["artifact_manifest"]
        manifest["analysis"].pop("transmitted_transcript_latest")
        self.backend.update_session_artifacts(self.session_id, manifest)
        final = self._judge_and_finalize()
        units = {unit["unit_index"]: unit for unit in final["units"]}
        for index in (1, 2):
            self.assertIsNone(units[index]["complete_transmitted"])
            self.assertEqual(units[index]["complete_transmitted_reason"],
                             "transmitted_transcript_unavailable")
            self.assertEqual(units[index]["transmitted_proxy"], "raw")
        self.assertEqual(final["derived"]["demonstrated_grounding"], 1)

    def test_completeness_rule_edges(self):
        from study.coding.packets import transmitted_completeness
        meta = {
            "utterance_texts_raw": {"participant_001": "charging light stayed off"},
            "transmitted": {"available": True, "texts": {
                "participant_001": {"text": "charging light stayed off",
                                    "asr_status": "partial"}}},
        }
        # High recall counts even when transmitted ASR is only partial.
        result = transmitted_completeness(["participant_001"], meta)
        self.assertEqual(result["value"], 1)
        # No delivery -> inapplicable.
        self.assertEqual(
            transmitted_completeness([], meta)["reason"],
            "unit_not_completely_delivered")


class TransmittedTranscriptModuleTests(FixtureCase):
    """prepare_transmitted_transcript over a real (silent) WAV with a fake
    transcriber: same interval ids as the dialogue transcript, artifact
    registered relative to the data root."""

    def test_prepare_and_load(self):
        import wave as wave_module

        from study.transmitted_transcript import (load_latest,
                                                  prepare_transmitted_transcript)
        session_dir = self.data_root / "media" / "sessions" / self.session_id
        wav_path = session_dir / "participant.wav"
        with wave_module.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 16000 * 82)   # 82 s of silence
        self.backend.save_session(
            self.session_id,
            {"participant": f"media/sessions/{self.session_id}/participant.wav"},
            {"model": []}, None, False)
        session = self.backend.get_session(self.session_id)
        timing = {
            "schema": "hmo.timing-analysis.v4",
            "participant_intervals": [
                {"start_ms": start, "end_ms": end}
                for uid, speaker, start, end, _ in UTTERANCES
                if speaker == "participant"
            ],
            "integrity": {"capture_crosswalk_complete": True,
                          "missing_at_proxy": 0},
        }
        calls = []

        def transcriber(path):
            calls.append(path)
            # One word inside each participant interval.
            return {"status": "complete", "error": None, "words": [
                {"word": f"w{index}", "start": start / 1000 + 0.1,
                 "end": start / 1000 + 0.4}
                for index, (uid, speaker, start, end, _) in enumerate(UTTERANCES)
                if speaker == "participant"]}

        result = prepare_transmitted_transcript(
            session, self.data_root, "TT1", timing, transcriber)
        self.assertEqual(result["schema"], "hmo.transmitted-transcript.v2")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["alignment"], "assumed_chunk_aligned")
        ids = [row["id"] for row in result["utterances"]]
        self.assertEqual(ids, [uid for uid, speaker, *_ in UTTERANCES
                               if speaker == "participant"])
        # One pass over the transmitted track, not one slice per interval.
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(row["text"] for row in result["utterances"]))
        # Register and reload through the manifest helper.
        manifest = session["artifact_manifest"]
        manifest["analysis"]["transmitted_transcript_latest"] = \
            result["result_artifact"]
        self.backend.update_session_artifacts(self.session_id, manifest)
        session = self.backend.get_session(self.session_id)
        loaded = load_latest(session, self.data_root)
        self.assertEqual(loaded["analysis_id"], "TT1")


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


class VerdictGatingTests(unittest.TestCase):
    """Fields below a "no" are not judgments and must stay blank."""

    def test_no_barge_in_blanks_yielding_but_not_disruption(self):
        from study.dataset_export import gate_turn_verdict

        gated = gate_turn_verdict({
            "verified_overlap": "1",
            "verified_participant_barge_in": "0",
            "verified_assistant_premature_onset": "1",
            "successful_assistant_yielding": "0",
            "disruptive_assistant_interruption": "0",
            "assistant_backchannel_onset": "1",
            "verified_assistant_stop_latency_ms": "240",
            "verified_participant_stop_latency_ms": "310",
        })
        self.assertIsNone(gated["successful_assistant_yielding"])
        self.assertIsNone(gated["verified_assistant_stop_latency_ms"])
        self.assertEqual(gated["disruptive_assistant_interruption"], "0")
        self.assertEqual(gated["assistant_backchannel_onset"], "1")
        self.assertEqual(gated["verified_participant_stop_latency_ms"], "310")

    def test_no_real_speech_blanks_everything_downstream(self):
        from study.dataset_export import gate_turn_verdict

        gated = gate_turn_verdict({
            "verified_overlap": "0",
            "verified_participant_barge_in": "1",
            "successful_assistant_yielding": "0",
            "disruptive_assistant_interruption": "0",
            "verification_note": "wind noise",
        })
        for field in ("verified_participant_barge_in",
                      "verified_assistant_premature_onset",
                      "successful_assistant_yielding",
                      "disruptive_assistant_interruption",
                      "assistant_backchannel_onset"):
            self.assertIsNone(gated[field])
        self.assertEqual(gated["verification_note"], "wind noise")


class AsrLowConfidenceTests(unittest.TestCase):
    def test_only_short_participant_intervals_are_flagged(self):
        from study.coding.packets import asr_low_confidence

        self.assertTrue(asr_low_confidence(
            {"speaker": "participant", "start_ms": 1000.0, "end_ms": 1300.0}))
        self.assertFalse(asr_low_confidence(
            {"speaker": "participant", "start_ms": 1000.0, "end_ms": 1700.0}))
        self.assertFalse(asr_low_confidence(
            {"speaker": "assistant", "start_ms": 1000.0, "end_ms": 1300.0}))
        self.assertFalse(asr_low_confidence({"speaker": "participant"}))


class TurnEventKeyTests(unittest.TestCase):
    def test_key_survives_episode_renumbering(self):
        from study.dataset_export import turn_event_key

        before = {"episode_id": "overlap_0003",
                  "participant_onset_ms": 36874.3, "assistant_onset_ms": 35290.0}
        after = {"episode_id": "overlap_0002",
                 "participant_onset_ms": "36874.3", "assistant_onset_ms": "35290.0"}
        self.assertEqual(turn_event_key("P1_R01_S02_A01", before),
                         turn_event_key("P1_R01_S02_A01", after))
        self.assertEqual(turn_event_key("P1_R01_S02_A01", before),
                         "P1_R01_S02_A01::p36874::a35290")

    def test_missing_onsets_still_produce_a_key(self):
        from study.dataset_export import turn_event_key

        self.assertEqual(turn_event_key("P1_R01_S02_A01", {}),
                         "P1_R01_S02_A01::pna::ana")


class DatasetDownloadTests(FixtureCase):
    """The admin dataset download: tables built on demand, streamed as a zip."""

    manifest_paths_relative_to_media = True

    def _client(self):
        from unittest.mock import Mock, patch

        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from study import router as study_router

        # Router paths are module-level (read at import), so point them at this
        # fixture's data root the same way the finalization tests do.
        media = self.data_root / "media"
        for item in (patch.object(study_router, "_DATA_ROOT", str(self.data_root)),
                     patch.object(study_router, "STUDY_DATA_DIR", media),
                     patch.object(study_router, "SESSIONS_DIR", media / "sessions"),
                     patch.object(study_router, "TARGETS_DIR", media / "targets"),
                     patch.object(study_router, "get_backend", return_value=self.backend),
                     patch.object(study_router, "get_manager", return_value=Mock())):
            item.start()
            self.addCleanup(item.stop)

        app = FastAPI()
        app.include_router(study_router.build_study_router())
        return TestClient(app), {"X-Study-Admin-Token": study_router.ADMIN_TOKEN}

    def test_dataset_download_contains_every_table(self):
        import io
        import zipfile

        client, headers = self._client()
        response = client.get(f"/api/study/studies/{self.study_id}/dataset",
                              headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = set(archive.namelist())
            summary = json.loads(archive.read("export_summary.json"))
            scenarios = archive.read("scenarios.csv").decode()
        # Every table the method's measures draw on, plus its documentation.
        self.assertLessEqual({
            "participants.csv", "scenarios.csv", "units.csv", "repairs.csv",
            "turn_events.csv", "turn_verification_queue.csv", "turn_gaps.csv",
            "turn_gap_verification_queue.csv", "turn_session_review_queue.csv",
            "turn_event_manual_additions.csv", "vc_quality_regions.csv",
            "answers_long.csv", "DATA_DICTIONARY.md", "export_summary.json",
        }, names)
        self.assertEqual(summary["analytical_sessions"], 1)
        self.assertIn(self.session_id, scenarios)

    def test_unknown_study_is_rejected(self):
        client, headers = self._client()
        response = client.get("/api/study/studies/9999/dataset", headers=headers)
        self.assertEqual(response.status_code, 404)

    def test_admin_token_is_required(self):
        client, _ = self._client()
        response = client.get(f"/api/study/studies/{self.study_id}/dataset")
        self.assertEqual(response.status_code, 401)


class TurnReviewTests(DatasetDownloadTests):
    """Manual turn verification: blinded queue, condition-neutral audio,
    single assignment per event, and verdicts that reach the export."""

    def _seed_export(self, client, headers):
        """A dataset export pins the review pass's item set."""
        response = client.get(f"/api/study/studies/{self.study_id}/dataset",
                              headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_queue_withholds_condition(self):
        client, headers = self._client()
        self._seed_export(client, headers)
        response = client.get(
            f"/api/study/studies/{self.study_id}/review/turn-queue", headers=headers)
        self.assertEqual(response.status_code, 200)
        body = response.text.lower()
        for token in ("vc_activation", "stable_natural", "stable_converted",
                      "vc_deactivation"):
            self.assertNotIn(token, body)
        payload = response.json()
        self.assertTrue(payload["export"].startswith("dataset_"))
        for event in payload["events"]:
            self.assertNotIn("condition", event)

    def test_transmitted_track_is_never_served(self):
        client, headers = self._client()
        self._seed_export(client, headers)
        response = client.get(
            f"/api/study/studies/{self.study_id}/review/audio",
            params={"session_id": self.session_id, "track": "merged"},
            headers=headers)
        # merged.wav is built from the transmitted (converted) participant
        # audio, so serving it would reveal the condition.
        self.assertEqual(response.status_code, 422)

    def test_context_drops_condition_revealing_fields(self):
        client, headers = self._client()
        self._seed_export(client, headers)
        response = client.get(
            f"/api/study/studies/{self.study_id}/review/context",
            params={"session_id": self.session_id, "from_ms": 0, "to_ms": 999999},
            headers=headers)
        self.assertEqual(response.status_code, 200)
        utterances = response.json()["utterances"]
        self.assertTrue(utterances)
        for utterance in utterances:
            self.assertEqual(set(utterance),
                             {"id", "speaker", "text", "start_ms", "end_ms"})

    def test_claim_hands_each_event_to_one_reviewer(self):
        client, headers = self._client()
        self._seed_export(client, headers)
        first = client.post(f"/api/study/studies/{self.study_id}/review/claim",
                            json={"reviewer": "AA"}, headers=headers).json()
        second = client.post(f"/api/study/studies/{self.study_id}/review/claim",
                             json={"reviewer": "BB"}, headers=headers).json()
        self.assertIsNotNone(first["event_key"])
        self.assertNotEqual(first["event_key"], second["event_key"])

    def test_verdict_reaches_the_export_and_keeps_history(self):
        client, headers = self._client()
        self._seed_export(client, headers)
        queue = client.get(f"/api/study/studies/{self.study_id}/review/turn-queue",
                           headers=headers).json()
        event = queue["events"][0]
        for initials in ("AA", "BB"):  # a revision must append, not overwrite
            response = client.post(
                f"/api/study/studies/{self.study_id}/review/turn-verdict",
                json={"event_key": event["event_key"], "verified_overlap": "1",
                      "verifier_initials": initials}, headers=headers)
            self.assertEqual(response.status_code, 200)
        verdict_file = (self.data_root / "review" / f"study{self.study_id}"
                        / "turn_verdicts.jsonl")
        self.assertEqual(len(verdict_file.read_text().strip().splitlines()), 2)

        out_dir = self.data_root / "exports" / "verified"
        summary = build_dataset(self.study_id, out_dir)
        self.assertEqual(summary["turn_events_verified"], 1)
        with (out_dir / "turn_events.csv").open() as handle:
            rows = {r["session_id"] + "::" + r["episode_id"]: r
                    for r in csv.DictReader(handle)}
        self.assertEqual(rows[event["event_key"]]["verifier_initials"], "BB")
        self.assertEqual(rows[event["event_key"]]["verified_overlap"], "1")


class ProductionLayoutTests(FixtureCase):
    """The real container layout: manifest paths relative to the media dir,
    which readers holding STUDY_DATA_ROOT silently failed to resolve."""

    manifest_paths_relative_to_media = True

    def test_export_loads_timing_and_emits_turn_events(self):
        out_dir = self.data_root / "exports" / "production"
        summary = build_dataset(self.study_id, out_dir)
        self.assertEqual(summary["turn_event_candidates"], 3)
        self.assertEqual(summary["artifact_coverage"]["timing_latest.loaded"], 1)
        self.assertEqual(summary["warnings"], [])
        with (out_dir / "scenarios.csv").open() as handle:
            row = list(csv.DictReader(handle))[0]
        self.assertEqual(row["overlap_candidates_200ms"], "2")
        self.assertEqual(row["barge_in_candidates"], "1")
        self.assertEqual(row["assistant_premature_onset_candidates"], "2")
        self.assertEqual(row["crosswalk_complete"], "1")
        self.assertEqual(row["capture_gap_total_ms"], "12.0")

    def test_packets_load_dialogue_timing_and_transmitted(self):
        summary = self._write_all_packets()
        self.assertEqual(summary["written"], 1)
        packet_id = read_index(self._root())[0]["packet_id"]
        packet = json.loads(
            (self._root() / "packets" / f"{packet_id}.json").read_text())
        meta = json.loads(
            (self._root() / "meta" / f"{packet_id}.json").read_text())
        self.assertTrue(packet["utterances"])                       # dialogue
        self.assertEqual(meta["switch_participant_timeline_ms"], 45000.0)  # timing
        self.assertEqual(meta["boundary_kind"], "switch")
        self.assertTrue(meta["transmitted"]["available"])           # transmitted
        self.assertTrue(meta["transmitted"]["texts"])

    def test_absolute_manifest_path_inside_root_resolves(self):
        session = self._analytical_session()
        absolute = str(self.data_root / "media" / "sessions" / self.session_id
                       / "analysis" / "timing.json")
        session["artifact_manifest"]["analysis"]["timing_latest"]["path"] = absolute
        loaded = load_manifest_artifact(session, self.data_root, "timing_latest")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["schema"], "hmo.timing-analysis.v4")

    def _analytical_session(self) -> dict:
        return next(s for s in self._sessions()
                    if s["session_id"] == self.session_id)


class LegacyLayoutCompatibilityTests(FixtureCase):
    """Fixtures and any older manifests using "media/sessions/..." keep working."""

    def test_legacy_prefixed_path_still_loads(self):
        session = next(s for s in self._sessions()
                       if s["session_id"] == self.session_id)
        self.assertTrue(
            session["artifact_manifest"]["analysis"]["timing_latest"]["path"]
            .startswith("media/"))
        loaded = load_manifest_artifact(session, self.data_root, "timing_latest")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["schema"], "hmo.timing-analysis.v4")

    def test_media_dir_as_data_root_also_resolves(self):
        """Offline copies may hand readers the media dir itself."""
        session = next(s for s in self._sessions()
                       if s["session_id"] == self.session_id)
        loaded = load_manifest_artifact(
            session, self.data_root / "media", "timing_latest")
        self.assertIsNotNone(loaded)


class ArtifactResolverTests(unittest.TestCase):
    """Resolution rules independent of the study fixture."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        target = self.root / "media" / "sessions" / "s1" / "analysis"
        target.mkdir(parents=True)
        (target / "timing.json").write_text(json.dumps({"ok": True}))
        self.outside = self.root.parent / "outside_root.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_resolves_production_and_legacy_forms(self):
        for path in ("sessions/s1/analysis/timing.json",
                     "media/sessions/s1/analysis/timing.json"):
            self.assertIsNotNone(resolve_artifact_path(self.root, path), path)

    def test_missing_file_returns_none(self):
        self.assertIsNone(
            resolve_artifact_path(self.root, "sessions/s1/analysis/nope.json"))

    def test_empty_path_returns_none(self):
        self.assertIsNone(resolve_artifact_path(self.root, None))
        self.assertIsNone(resolve_artifact_path(self.root, ""))

    def test_traversal_outside_root_is_rejected(self):
        self.outside.write_text(json.dumps({"secret": True}))
        try:
            self.assertIsNone(
                resolve_artifact_path(self.root, "../../outside_root.json"))
            self.assertIsNone(
                resolve_artifact_path(self.root, str(self.outside)))
        finally:
            self.outside.unlink()


class MissingArtifactVisibilityTests(FixtureCase):
    """A path/layout problem must never read as valid zero-event data."""

    manifest_paths_relative_to_media = True

    def test_unreadable_timing_is_counted_and_warned(self):
        (self.data_root / "media" / "sessions" / self.session_id
         / "analysis" / "timing.json").unlink()
        summary = build_dataset(self.study_id,
                                self.data_root / "exports" / "broken")
        self.assertEqual(summary["turn_event_candidates"], 0)
        self.assertEqual(summary["artifact_coverage"].get("timing_latest.loaded", 0), 0)
        self.assertEqual(summary["artifact_coverage"]["timing_latest.unreadable"], 1)
        self.assertTrue(any("no timing artifact could be loaded" in w
                            for w in summary["warnings"]))

    def test_session_without_timing_is_not_reported_as_unreadable(self):
        session_id = self.session_id
        manifest = next(s for s in self._sessions()
                        if s["session_id"] == session_id)["artifact_manifest"]
        manifest["analysis"].pop("timing_latest")
        self.backend.update_session_artifacts(session_id, manifest)
        summary = build_dataset(self.study_id,
                                self.data_root / "exports" / "notiming")
        self.assertEqual(summary["artifact_coverage"]["timing_latest.not_recorded"], 1)
        self.assertEqual(summary["artifact_coverage"].get("timing_latest.unreadable", 0), 0)


if __name__ == "__main__":
    unittest.main()
