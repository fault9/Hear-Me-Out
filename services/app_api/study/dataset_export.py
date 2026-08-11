"""Build the analysis-ready dataset (tidy CSV tables) from a study data root.

Run as:  python -m study.dataset_export --study-id N [--out DIR]
(cwd = services/app_api; STUDY_DATA_ROOT may point at a copied data root, so
this runs offline on any machine.)

Tables (empty cell = missing/inapplicable; observed failures are explicit 0):

  participants.csv   one row per participant (allocation + background answers)
  scenarios.csv      one row per analytical session attempt (canonical flags,
                     condition/position, validity, timing summary, post
                     questionnaire, coded outcomes when present)
  units.csv          one row per critical unit per session (delivery +
                     grounding codes, zero-vs-missing semantics preserved)
  repairs.csv        one row per coded repair move
  turn_events.csv    one linked row per nominated overlap episode, with
                     participant barge-in and assistant premature-onset flags
  turn_verification_queue.csv
                     included, timing-valid event rows awaiting verification
  turn_gaps.csv      positive participant-to-assistant and
                     assistant-to-participant response gaps
  turn_gap_verification_queue.csv
                     included, timing-valid gap rows awaiting verification
  turn_session_review_queue.csv
                     one full-track review row per eligible session
  turn_event_manual_additions.csv
                     template for events missed by automatic nomination
  vc_quality_regions.csv
                     one row per route-level VC-quality score
  answers_long.csv   every questionnaire answer in long form
  DATA_DICTIONARY.md column definitions and coding conventions

Coded outcomes are merged from the coding pipeline's final labels
(coding/study<id>/labels/final/*.json) when they exist; the tables are fully
buildable before coding has run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.artifacts import load_manifest_artifact  # noqa: E402
from study.coding.packets import coding_root  # noqa: E402
from study.session_scope import annotate_analysis_scopes  # noqa: E402
from study.storage import get_backend  # noqa: E402
from study.timing_analysis import (BOUNDARY_CONFIRMATION_STATUS,
                                   resolve_boundary_validation)  # noqa: E402
from study.turn_taking import (OVERLAP_MINIMUM_MS,
                               turn_events_from_timing)  # noqa: E402

MISSING = ""


def _cell(value) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})


def _load_artifact(data_root: Path, session: dict, key: str,
                   coverage: Counter | None = None) -> dict | None:
    """Load one manifest artifact, tallying whether it was recorded and read.

    The tally separates "never produced" from "recorded but unreadable" so an
    export cannot silently present a path/layout problem as valid zero data.
    """
    loaded = load_manifest_artifact(session, data_root, key)
    if coverage is not None:
        analysis = (session.get("artifact_manifest") or {}).get("analysis") or {}
        record = analysis.get(key) or {}
        recorded = bool(record.get("path")) if isinstance(record, dict) else False
        coverage[f"{key}.sessions"] += 1
        if loaded is not None:
            coverage[f"{key}.loaded"] += 1
        elif recorded:
            coverage[f"{key}.unreadable"] += 1
        else:
            coverage[f"{key}.not_recorded"] += 1
    return loaded


def _final_labels_by_session(data_root: Path, study_id: int) -> dict[str, dict]:
    final_dir = coding_root(data_root, study_id) / "labels" / "final"
    out: dict[str, dict] = {}
    if final_dir.exists():
        for path in sorted(final_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            session_id = record.get("session_id")
            if session_id:
                out[str(session_id)] = record
    return out


def _answers_maps(answers: list[dict]) -> tuple[dict, dict, list[dict]]:
    """Return (per-participant latest payload by kind, per-session latest
    payload by kind, long rows). Later submissions overwrite earlier ones."""
    by_participant: dict[tuple[str, str], dict] = {}
    by_session: dict[tuple[str, str], dict] = {}
    long_rows: list[dict] = []
    for row in sorted(answers, key=lambda r: r.get("created_at") or 0):
        kind = str(row.get("kind") or "")
        payload = row.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        participant = str(row.get("participant_id") or "")
        session = row.get("session_id")
        by_participant[(participant, kind)] = payload
        if session:
            by_session[(str(session), kind)] = payload
        for question, value in payload.items():
            long_rows.append({
                "participant_id": participant,
                "run_id": row.get("run_id"),
                "session_id": session,
                "kind": kind,
                "question_id": question,
                "value": value,
                "answered_at_unix_s": row.get("created_at"),
            })
    return by_participant, by_session, long_rows


def _vc_quality_rows(sessions: list[dict],
                     target_refs: dict[str, str | None]) -> list[dict]:
    """Flatten route-region VC-quality output without collapsing missingness."""
    rows: list[dict] = []
    for session in sessions:
        result = session.get("vc_quality") or {}
        inputs = result.get("inputs") or {}
        target = inputs.get("target_audio") or {}
        region_specs = {
            int(region["index"]): region
            for region in inputs.get("regions") or []
            if region.get("index") is not None and region.get("mode") == "vc"
        }
        scores = {
            int(score["region"]): score.get("metrics") or {}
            for score in result.get("scores") or []
            if score.get("region") is not None
        }
        # A failed VC session may not have reached input preparation. Preserve
        # one explicit missing row rather than silently disappearing it.
        has_vc_route = any(row.get("mode") == "vc"
                           for row in session.get("schedule") or [])
        indices = sorted(set(region_specs) | set(scores))
        if has_vc_route and not indices:
            indices = [None]
        unavailable = result.get("unavailable_metrics") or []
        unavailable_by_region = {
            (item.get("region"), item.get("metric")): item.get("error")
            for item in unavailable
        }
        for index in indices:
            region = region_specs.get(index, {}) if index is not None else {}
            metrics = scores.get(index, {}) if index is not None else {}
            score_source = region.get("score_source") or {}
            score_transmitted = region.get("score_transmitted") or {}
            selection = region.get("score_selection") or {}
            # Route input offsets are defined on the proxy's fixed 16 kHz
            # participant-input timeline. Derived WAV records normally carry
            # the same rate, but the constant is the authoritative fallback.
            source_rate = ((region.get("source") or {}).get("sample_rate_hz")
                           or (inputs.get("source_audio") or {}).get("sample_rate_hz")
                           or 16000)
            start_sample = region.get("input_start_sample")
            end_sample = region.get("input_end_sample")
            rows.append({
                "session_id": session.get("session_id"),
                "participant_id": session.get("participant_id"),
                "condition": session.get("voice_condition"),
                "analysis_included": session.get("analysis_included"),
                "canonical_run": session.get("canonical_run"),
                "canonical_attempt": session.get("canonical_attempt"),
                "target_ref": target_refs.get(str(session.get("participant_id"))),
                "target_speaker_id": session.get("target_speaker_id"),
                "vc_quality_storage_status": session.get("vc_quality_status"),
                "vc_quality_result_status": result.get("status"),
                "analysis_id": result.get("analysis_id"),
                "metric_profile": result.get("metric_profile"),
                "region_index": index,
                "route_mode": region.get("mode") or ("vc" if index is not None else None),
                "input_sample_rate_hz": source_rate,
                "input_start_sample": start_sample,
                "input_end_sample": end_sample,
                "input_start_s": (start_sample / source_rate
                                  if isinstance(start_sample, (int, float))
                                  and source_rate else None),
                "input_end_s": (end_sample / source_rate
                                if isinstance(end_sample, (int, float))
                                and source_rate else None),
                "transmitted_start_sample": region.get("transmitted_start_sample"),
                "transmitted_end_sample": region.get("transmitted_end_sample"),
                "inference_windows": region.get("windows"),
                "stable_guard_s": region.get("stable_guard_s"),
                "score_selection": selection.get("mode"),
                "speech_intervals": selection.get("speech_intervals"),
                "boundary_padding_s": selection.get("boundary_padding_s"),
                "source_score_duration_s": score_source.get("duration_s"),
                "transmitted_score_duration_s": score_transmitted.get("duration_s"),
                "source_score_path": score_source.get("path"),
                "source_score_sha256": score_source.get("sha256"),
                "transmitted_score_path": score_transmitted.get("path"),
                "transmitted_score_sha256": score_transmitted.get("sha256"),
                "target_path": target.get("path"),
                "target_sha256": target.get("sha256"),
                "target_duration_s": target.get("duration_s"),
                "wer": metrics.get("wer"),
                "wer_status": metrics.get("wer_status"),
                "wer_error": (metrics.get("wer_error")
                              or unavailable_by_region.get((index, "wer"))),
                "wer_reference_kind": metrics.get("ref_kind"),
                "sim": metrics.get("sim"),
                "sim_status": metrics.get("sim_status"),
                "sim_error": (metrics.get("sim_error")
                              or unavailable_by_region.get((index, "sim"))),
                "utmos": metrics.get("utmos"),
                "utmos_status": metrics.get("utmos_status"),
                "utmos_error": (metrics.get("utmos_error")
                                or unavailable_by_region.get((index, "utmos"))),
                "session_error": result.get("error"),
                "result_artifact_path": (result.get("result_artifact") or {}).get("path"),
                "result_artifact_sha256": (result.get("result_artifact") or {}).get("sha256"),
            })
    return rows


def _analytical_positions(sessions: list[dict]) -> dict[str, int]:
    """Ordinal analytical position (1..4) per session id, from each
    participant's analytical scenario_order slots."""
    slots: dict[str, set[int]] = {}
    for session in sessions:
        if session.get("study_role") == "practice":
            continue
        slots.setdefault(str(session["participant_id"]), set()).add(
            int(session.get("scenario_order") or 0))
    ranks = {
        participant: {order: i + 1 for i, order in enumerate(sorted(orders))}
        for participant, orders in slots.items()
    }
    return {
        str(session["session_id"]): ranks
        .get(str(session["participant_id"]), {})
        .get(int(session.get("scenario_order") or 0))
        for session in sessions
        if session.get("study_role") != "practice"
    }


def _manual_turn_verification_eligible(session: dict, timing: dict) -> bool:
    """Apply the frozen post-hoc Praat audit to legacy session snapshots.

    Existing technical-validity summaries predate the completed audit and must
    remain immutable. This analysis-time flag combines their saved clock
    evidence with the now-frozen validation artifact.
    """
    if not timing:
        return False
    validity = session.get("technical_validity") or {}
    integrity = timing.get("integrity") or {}
    if (validity.get("valid_for_manual_turn_verification") is True
            or validity.get("valid_for_confirmatory_timing_analysis") is True):
        return integrity.get("crosswalk_complete") is True
    reconstruction_valid = validity.get("valid_for_timing_reconstruction")
    if reconstruction_valid is None:
        reconstruction_valid = integrity.get("valid_for_timing") is True
    study = (session.get("config_snapshot") or {}).get("study") or {}
    settings = study.get("settings") or {}
    boundary = resolve_boundary_validation(settings.get("timing") or {})
    return bool(
        reconstruction_valid
        and integrity.get("crosswalk_complete") is True
        and boundary.get("status") == BOUNDARY_CONFIRMATION_STATUS
    )


VERDICT_FIELDS = (
    "verified_overlap", "verified_participant_barge_in",
    "verified_assistant_premature_onset", "successful_assistant_yielding",
    "disruptive_assistant_interruption", "assistant_backchannel_onset",
    "verified_assistant_stop_latency_ms",
    "verified_participant_stop_latency_ms", "verifier_initials",
    "verification_note",
)

# A verdict is hierarchical: everything below a "no" is not a judgment at all.
# Yielding applies only to a verified barge-in, disruption only to a verified
# premature onset, and neither exists without real simultaneous speech.
# Blanked at ingest and again on read, so "no barge-in" can never reach the
# frames as "failed to yield" (missing != zero, per the data dictionary).
TURN_VERDICT_GATES = (
    ("verified_overlap", (
        "verified_participant_barge_in", "verified_assistant_premature_onset",
        "successful_assistant_yielding", "disruptive_assistant_interruption",
        "assistant_backchannel_onset",
        "verified_assistant_stop_latency_ms",
        "verified_participant_stop_latency_ms")),
    ("verified_participant_barge_in", (
        "successful_assistant_yielding", "verified_assistant_stop_latency_ms")),
    ("verified_assistant_premature_onset", (
        "disruptive_assistant_interruption", "assistant_backchannel_onset",
        "verified_participant_stop_latency_ms")),
)


def gate_turn_verdict(record: dict) -> dict:
    gated = dict(record)
    for parent, children in TURN_VERDICT_GATES:
        if str(gated.get(parent) or "").strip() != "1":
            for child in children:
                gated[child] = None
    return gated


def turn_event_key(session_id: str, event: dict) -> str:
    """Identity that survives re-analysis. The episode index is positional and
    shifts whenever the episode set changes, but a turn's onsets are
    deterministic recomputations from the stored audio, so verdicts keyed on
    them still resolve after a re-run."""
    parts = []
    for field in ("participant_onset_ms", "assistant_onset_ms"):
        try:
            parts.append(str(int(round(float(event.get(field))))))
        except (TypeError, ValueError):
            parts.append("na")
    return f"{session_id}::p{parts[0]}::a{parts[1]}"


def load_turn_verdicts(data_root: Path, study_id: int) -> dict[str, dict]:
    """Manual turn-verification decisions, keyed by turn_event_key.

    The file is append-only so the review pass keeps its history; the last
    record for an event is the one the dataset uses.
    """
    path = Path(data_root) / "review" / f"study{int(study_id)}" / "turn_verdicts.jsonl"
    verdicts: dict[str, dict] = {}
    if not path.is_file():
        return verdicts
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        key = record.get("event_key")
        if key:
            verdicts[str(key)] = gate_turn_verdict(record)
    return verdicts


GAP_VERDICT_FIELDS = (
    "verified_positive_gap", "verified_gap_start_ms", "verified_gap_end_ms",
    "verified_gap_duration_ms", "verifier_initials", "verification_note",
)


def gap_key(session_id: object, gap_id: object) -> str:
    return f"{session_id}:{gap_id}"


def load_gap_verdicts(data_root: Path, study_id: int) -> dict[str, dict]:
    """Manual response-gap decisions, keyed by session and gap id.

    Same append-only store as the episode verdicts and read the same way: the
    last record for a gap is the one the dataset uses.
    """
    path = (Path(data_root) / "review" / f"study{int(study_id)}"
            / "turn_gap_verdicts.jsonl")
    verdicts: dict[str, dict] = {}
    if not path.is_file():
        return verdicts
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        key = record.get("gap_key")
        if key:
            verdicts[str(key)] = record
    return verdicts


def gate_gap_verdict(verdict: dict | None) -> dict:
    # A gap judged not positive carries no corrected boundaries: those are a
    # judgment about a gap that exists, and must not read as zero-length.
    fields = {field: (verdict or {}).get(field) for field in GAP_VERDICT_FIELDS}
    if str(fields.get("verified_positive_gap")) not in ("1", "True", "true"):
        for field in ("verified_gap_start_ms", "verified_gap_end_ms",
                      "verified_gap_duration_ms"):
            fields[field] = None
    return fields


def _verdict_fields(verdict: dict | None) -> dict:
    return {field: (verdict or {}).get(field) for field in VERDICT_FIELDS}


def _coverage_warnings(coverage: Counter, analytical_count: int) -> list[str]:
    """Flag artifact-loading shortfalls so an empty export is never mistaken
    for a study with genuinely no timing or events."""
    warnings: list[str] = []
    if not analytical_count:
        return warnings
    loaded = coverage.get("timing_latest.loaded", 0)
    unreadable = coverage.get("timing_latest.unreadable", 0)
    if not loaded:
        warnings.append(
            f"no timing artifact could be loaded for any of {analytical_count} "
            "analytical sessions: timing columns and turn-event candidates are "
            "empty. Check that STUDY_DATA_ROOT is the data root that contains "
            "media/ (artifact paths are recorded relative to media/).")
    elif loaded < analytical_count:
        warnings.append(
            f"timing artifact loaded for {loaded}/{analytical_count} analytical "
            "sessions")
    if unreadable:
        warnings.append(
            f"{unreadable} session(s) record a timing artifact path that could "
            "not be resolved or read")
    return warnings


def compile_verification(out_dir: Path, *,
                         require_full_session_review: bool = True) -> dict:
    """Compile the manual turn review into this export.

    An incomplete review is not a failure here: finalize records what is
    outstanding and returns rather than raising, so one export command still
    produces every table the review can currently support.
    """
    from study.turn_verification import VerificationError, finalize

    try:
        report = finalize(
            out_dir, require_full_session_review=require_full_session_review)
    except (VerificationError, OSError) as exc:
        return {"status": "not_compiled", "reason": str(exc)}
    keep = ("status", "sessions_reviewed", "sessions_nomination_only",
            "full_session_review_complete", "candidate_events",
            "candidate_gaps", "manual_additions", "verified_events",
            "verified_gaps")
    compiled = {key: report[key] for key in keep if key in report}
    compiled["outstanding"] = len(report.get("errors") or [])
    return compiled


def _carry_file(previous: Path, out_dir: Path, name: str,
                key_fields: tuple[str, ...]) -> int:
    """Fill this export's blank cells from the matching row of an earlier one.

    Only blanks are filled, which is what makes the merge safe: a fresh export
    writes every human-entry cell empty and every machine column populated, so
    a recomputed value can never be overwritten by a stale one.
    """
    old_path, new_path = previous / name, out_dir / name
    if not old_path.exists() or not new_path.exists():
        return 0
    with old_path.open(newline="") as handle:
        prior = {tuple(row.get(key, "") for key in key_fields): row
                 for row in csv.DictReader(handle)}
    with new_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    carried = 0
    for row in rows:
        source = prior.get(tuple(row.get(key, "") for key in key_fields))
        if not source:
            continue
        # Evaluated before testing: any() short-circuits, and carrying only the
        # first blank of a row silently drops the rest of a verdict.
        filled = [_fill_blank(row, source, column) for column in columns]
        if any(filled):
            carried += 1
    _write_csv(new_path, rows, columns)
    return carried


def _fill_blank(row: dict, source: dict, column: str) -> bool:
    if str(row.get(column) or "").strip():
        return False
    value = str(source.get(column) or "").strip()
    if not value:
        return False
    row[column] = value
    return True


def carry_verification(previous: Path, out_dir: Path) -> dict:
    """Carry a turn-verification review forward into a newer export.

    The review is entered into the export's own queue files, so re-exporting to
    pick up new sessions would otherwise abandon a half-finished one.
    """
    if previous.resolve() == out_dir.resolve():
        raise ValueError("carry source and destination are the same export")
    carried = {
        "events": _carry_file(previous, out_dir,
                              "turn_verification_queue.csv", ("event_key",)),
        "gaps": _carry_file(previous, out_dir,
                            "turn_gap_verification_queue.csv",
                            ("session_id", "gap_id")),
        "session_reviews": _carry_file(previous, out_dir,
                                       "turn_session_review_queue.csv",
                                       ("session_id",)),
    }
    # Manual additions are hand-authored in full and a new export writes the
    # template empty, so they are copied rather than merged.
    old_path = previous / "turn_event_manual_additions.csv"
    new_path = out_dir / "turn_event_manual_additions.csv"
    added = []
    if old_path.exists() and new_path.exists():
        with old_path.open(newline="") as handle:
            added = list(csv.DictReader(handle))
        with new_path.open(newline="") as handle:
            columns = csv.DictReader(handle).fieldnames or []
        if added:
            _write_csv(new_path, added, columns)
    carried["manual_additions"] = len(added)
    carried["from"] = str(previous)
    return carried


def _scenario_key(value: object) -> str:
    """The scenarios table keys on the bare id while a session carries it as
    "scenario_20". Joined on the raw value every title came out empty, and an
    empty covariate in every model formula empties the model frame."""
    text = str(value or "")
    prefix = "scenario_"
    return text[len(prefix):] if text.startswith(prefix) else text


def build_dataset(study_id: int, out_dir: Path) -> dict:
    data_root = Path(os.path.expanduser(os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))
    backend = get_backend()
    study = backend.get_study(study_id) or {}
    participants = backend.list_participants(study_id)
    scenarios = {_scenario_key(row["id"]): row
                 for row in backend.list_scenarios(study_id)}
    runs = backend.list_runs(study_id)
    sessions = annotate_analysis_scopes(backend.list_sessions(study_id), runs)
    answers = backend.list_answers(study_id)
    by_participant, by_session, long_rows = _answers_maps(answers)
    final_labels = _final_labels_by_session(data_root, study_id)
    positions = _analytical_positions(sessions)

    # ---- participants.csv ----
    background_keys: list[str] = []
    for payload in (by_participant.get((str(p["participant_id"]), "background"), {})
                    for p in participants):
        for key in payload:
            if key not in background_keys:
                background_keys.append(key)
    participant_rows = []
    for p in participants:
        pid = str(p["participant_id"])
        row = {
            "participant_id": pid,
            "code": p.get("code"),
            "variant_id": p.get("variant_id"),
            "target_ref": p.get("target_ref"),
            "allocation_stratum": p.get("allocation_stratum"),
            "allocation_status": p.get("allocation_status"),
        }
        background = by_participant.get((pid, "background"), {})
        for key in background_keys:
            row[f"bg_{key}"] = background.get(key)
        participant_rows.append(row)
    participant_columns = (["participant_id", "code", "variant_id", "target_ref",
                            "allocation_stratum", "allocation_status"]
                           + [f"bg_{key}" for key in background_keys])

    # ---- scenarios.csv (analytical sessions, all attempts) ----
    analytical = [s for s in sessions if s.get("study_role") != "practice"]
    target_refs = {
        str(participant["participant_id"]): participant.get("target_ref")
        for participant in participants
    }
    vc_quality_rows = _vc_quality_rows(analytical, target_refs)
    # Load each session's timing artifact once; scenarios.csv and turn_events.csv
    # both read it, and one pass keeps the coverage tally per session.
    verdicts = load_turn_verdicts(data_root, study_id)
    gap_verdicts = load_gap_verdicts(data_root, study_id)
    coverage: Counter = Counter()
    timing_by_session = {
        str(session["session_id"]):
            (_load_artifact(data_root, session, "timing_latest", coverage) or {})
        for session in analytical
    }
    turn_by_session = {
        sid: turn_events_from_timing(timing)
        for sid, timing in timing_by_session.items()
    }
    post_keys: list[str] = []
    for session in analytical:
        payload = by_session.get((str(session["session_id"]), "post"), {})
        for key in payload:
            if key not in post_keys:
                post_keys.append(key)
    scenario_rows = []
    for session in analytical:
        sid = str(session["session_id"])
        validity = session.get("technical_validity") or {}
        timing = timing_by_session.get(sid, {})
        manual_turn_eligible = _manual_turn_verification_eligible(session, timing)
        timing_summary = timing.get("summary") or {}
        turn_episodes, response_gaps = turn_by_session.get(sid, ([], []))
        overlap_candidates = [
            row for row in turn_episodes if row.get("overlap_200ms_candidate")
        ]
        barge_in_candidates = [
            row for row in turn_episodes
            if row.get("participant_barge_in_candidate")
        ]
        premature_onset_candidates = [
            row for row in turn_episodes
            if row.get("assistant_premature_onset_candidate")
        ]
        stop_latencies = [
            float(row["assistant_stop_latency_ms"])
            for row in barge_in_candidates
            if row.get("assistant_stop_latency_ms") is not None
        ]
        assistant_response_gaps = [
            row for row in response_gaps
            if row.get("direction") == "participant_to_assistant"
        ]
        participant_response_gaps = [
            row for row in response_gaps
            if row.get("direction") == "assistant_to_participant"
        ]
        integrity = timing.get("integrity") or {}
        capture_gaps = integrity.get("capture_gaps") or {}
        scenario = scenarios.get(_scenario_key(session.get("scenario_id"))) or {}
        started, ended = session.get("started_at"), session.get("ended_at")
        final = final_labels.get(sid)
        derived = (final or {}).get("derived") or {}
        outcome = ((final or {}).get("labels") or {}).get("outcome") or {}
        account = ((final or {}).get("labels") or {}).get("final_account_accuracy") or {}
        row = {
            "session_id": sid,
            "participant_id": session.get("participant_id"),
            "scenario_id": session.get("scenario_id"),
            "scenario_title": scenario.get("title"),
            "scenario_order_slot": session.get("scenario_order"),
            "analytical_position": positions.get(sid),
            "condition": session.get("voice_condition"),
            "run_id": session.get("run_id"),
            "run_attempt": session.get("run_attempt"),
            "scenario_attempt": session.get("scenario_attempt"),
            "canonical_run": session.get("canonical_run"),
            "canonical_attempt": session.get("canonical_attempt"),
            "analysis_included": session.get("analysis_included"),
            "analysis_exclusion_reasons": session.get("analysis_exclusion_reasons"),
            "started_at_unix_s": started,
            "ended_at_unix_s": ended,
            "duration_s": (round(ended - started, 3)
                           if isinstance(started, (int, float))
                           and isinstance(ended, (int, float)) else None),
            "end_reason": session.get("end_reason"),
            "technical_status": validity.get("status"),
            "valid_for_condition_analysis": validity.get("valid_for_condition_analysis"),
            "valid_for_post_checkpoint_analysis": validity.get(
                "valid_for_post_checkpoint_analysis"),
            "valid_for_confirmatory_timing_analysis": validity.get(
                "valid_for_confirmatory_timing_analysis"),
            "valid_for_manual_turn_verification": manual_turn_eligible,
            "technical_failure_codes": [f.get("code")
                                        for f in validity.get("failures") or []],
            "participant_speech_intervals": timing_summary.get(
                "participant_speech_intervals"),
            "assistant_speech_intervals": timing_summary.get(
                "assistant_speech_intervals"),
            "overlap_candidates_200ms": (
                len(overlap_candidates) if timing else None),
            "participant_barge_in_candidates": (
                len(barge_in_candidates) if timing else None),
            # Compatibility alias retained for existing analysis notebooks.
            "barge_in_candidates": (
                len(barge_in_candidates) if timing else None),
            "assistant_premature_onset_candidates": (
                len(premature_onset_candidates) if timing else None),
            "mean_stop_latency_ms_candidates": (
                sum(stop_latencies) / len(stop_latencies)
                if stop_latencies else None),
            "assistant_response_gap_candidates": (
                len(assistant_response_gaps) if timing else None),
            "participant_response_gap_candidates": (
                len(participant_response_gaps) if timing else None),
            "crosswalk_complete": integrity.get("crosswalk_complete"),
            "capture_gap_total_ms": capture_gaps.get("total_gap_ms"),
            "capture_gap_max_ms": capture_gaps.get("max_gap_ms"),
            "vc_quality_status": session.get("vc_quality_status"),
            # coded outcomes (empty until the coding pipeline has finalized)
            "outcome_level": outcome.get("level"),
            "final_account_accuracy": account.get("value"),
            "demonstrated_grounding": derived.get("demonstrated_grounding"),
            "false_update_confirmation": derived.get("false_update_confirmation"),
            "any_repair": derived.get("any_repair"),
            "repair_total": derived.get("repair_total"),
            "repair_post_boundary": derived.get("repair_post_boundary"),
            "coding_boundary_kind": derived.get("boundary_kind"),
            "coding_label_source": ((final or {}).get("provenance") or {}).get("source"),
            # Which fields the human coder changed from the judge's labels;
            # empty when a packet has only one of the two.
            "coding_judge_human_disagreement": (
                (final or {}).get("provenance") or {}
            ).get("judge_human_disagreement_fields"),
        }
        post = by_session.get((sid, "post"), {})
        for key in post_keys:
            row[f"post_{key}"] = post.get(key)
        scenario_rows.append(row)
    scenario_columns = ([
        "session_id", "participant_id", "scenario_id", "scenario_title",
        "scenario_order_slot", "analytical_position", "condition", "run_id",
        "run_attempt", "scenario_attempt", "canonical_run", "canonical_attempt",
        "analysis_included", "analysis_exclusion_reasons", "started_at_unix_s",
        "ended_at_unix_s", "duration_s", "end_reason", "technical_status",
        "valid_for_condition_analysis", "valid_for_post_checkpoint_analysis",
        "valid_for_confirmatory_timing_analysis",
        "valid_for_manual_turn_verification", "technical_failure_codes",
        "participant_speech_intervals", "assistant_speech_intervals",
        "overlap_candidates_200ms", "participant_barge_in_candidates",
        "barge_in_candidates", "assistant_premature_onset_candidates",
        "mean_stop_latency_ms_candidates", "assistant_response_gap_candidates",
        "participant_response_gap_candidates", "crosswalk_complete",
        "capture_gap_total_ms", "capture_gap_max_ms", "vc_quality_status",
        "outcome_level", "final_account_accuracy", "demonstrated_grounding",
        "false_update_confirmation", "any_repair", "repair_total",
        "repair_post_boundary", "coding_boundary_kind", "coding_label_source",
        "coding_judge_human_disagreement",
    ] + [f"post_{key}" for key in post_keys])

    # ---- units.csv / repairs.csv ----
    unit_rows, repair_rows = [], []
    for session in analytical:
        sid = str(session["session_id"])
        final = final_labels.get(sid)
        if not final:
            continue
        scenario = scenarios.get(_scenario_key(session.get("scenario_id"))) or {}
        spec = ((scenario.get("scenario_card") or {}).get("analysis_spec") or {})
        unit_texts = list(spec.get("critical_units") or [])
        for unit in final.get("units") or []:
            index = unit.get("unit_index")
            unit_rows.append({
                "session_id": sid,
                "participant_id": session.get("participant_id"),
                "condition": session.get("voice_condition"),
                "unit_index": index,
                "unit_text": (unit_texts[index - 1]
                              if isinstance(index, int) and 0 < index <= len(unit_texts)
                              else None),
                "attempted": unit.get("attempted"),
                "complete_raw": unit.get("complete_raw"),
                "complete_transmitted": unit.get("complete_transmitted"),
                "transmitted_content_recall": unit.get("transmitted_content_recall"),
                "complete_transmitted_reason": unit.get("complete_transmitted_reason"),
                "grounding_gated_reason": unit.get("grounding_gated_reason"),
                "delivery_relative_to_boundary": unit.get("delivery_relative_to_boundary"),
                "delivery_utterance_ids": unit.get("delivery_utterance_ids"),
                "acknowledgement": unit.get("acknowledgement"),
                "update_claim": unit.get("update_claim"),
                "incorporation": unit.get("incorporation"),
                "retention": unit.get("retention"),
            })
        for i, move in enumerate(final.get("repairs") or [], start=1):
            repair_rows.append({
                "session_id": sid,
                "participant_id": session.get("participant_id"),
                "condition": session.get("voice_condition"),
                "move_index": i,
                "category": move.get("category"),
                "utterance_id": move.get("utterance_id"),
                "post_boundary": move.get("post_boundary"),
                "trouble": move.get("trouble"),
            })
    unit_columns = ["session_id", "participant_id", "condition", "unit_index",
                    "unit_text", "attempted", "complete_raw",
                    "complete_transmitted", "transmitted_content_recall",
                    "complete_transmitted_reason", "grounding_gated_reason",
                    "delivery_relative_to_boundary", "delivery_utterance_ids",
                    "acknowledgement", "update_claim", "incorporation",
                    "retention"]
    repair_columns = ["session_id", "participant_id", "condition", "move_index",
                      "category", "utterance_id", "post_boundary", "trouble"]

    # ---- directional turn episodes and positive response gaps ----
    event_rows: list[dict] = []
    gap_rows: list[dict] = []
    for session in analytical:
        sid = str(session["session_id"])
        timing = timing_by_session.get(sid, {})
        integrity = timing.get("integrity") or {}
        validity = session.get("technical_validity") or {}
        manual_turn_eligible = _manual_turn_verification_eligible(session, timing)
        files = session.get("files") or {}
        common = {
            "session_id": sid,
            "participant_id": session.get("participant_id"),
            "condition": session.get("voice_condition"),
            "analysis_included": session.get("analysis_included"),
            "valid_for_confirmatory_timing_analysis": validity.get(
                "valid_for_confirmatory_timing_analysis"),
            "valid_for_manual_turn_verification": manual_turn_eligible,
            "crosswalk_complete": integrity.get("crosswalk_complete"),
            "participant_raw_path": files.get("participant_raw"),
            "assistant_model_path": files.get("model"),
        }
        episodes, response_gaps = turn_by_session.get(sid, ([], []))
        for event in episodes:
            if not any((event.get("overlap_200ms_candidate"),
                        event.get("participant_barge_in_candidate"),
                        event.get("assistant_premature_onset_candidate"))):
                continue
            event_rows.append({
                **common,
                "episode_id": event.get("episode_id"),
                "event_key": turn_event_key(sid, event),
                "participant_interval": event.get("participant_interval"),
                "assistant_interval": event.get("assistant_interval"),
                "initiator": event.get("initiator"),
                "participant_onset_ms": event.get("participant_onset_ms"),
                "participant_offset_ms": event.get("participant_offset_ms"),
                "assistant_onset_ms": event.get("assistant_onset_ms"),
                "assistant_offset_ms": event.get("assistant_offset_ms"),
                "overlap_start_ms": event.get("overlap_start_ms"),
                "overlap_end_ms": event.get("overlap_end_ms"),
                "overlap_duration_ms": event.get("overlap_duration_ms"),
                "overlap_200ms_candidate": event.get(
                    "overlap_200ms_candidate"),
                "participant_barge_in_candidate": event.get(
                    "participant_barge_in_candidate"),
                "assistant_premature_onset_candidate": event.get(
                    "assistant_premature_onset_candidate"),
                "assistant_stop_latency_ms_candidate": event.get(
                    "assistant_stop_latency_ms"),
                "participant_stop_latency_ms_candidate": event.get(
                    "participant_stop_latency_ms"),
                "legacy_reconstruction": event.get("legacy_reconstruction", False),
                # Manual decisions come from the review pass; an unverified
                # event keeps them empty. Onset direction is automatic;
                # yielding/disruption require listening.
                **_verdict_fields(verdicts.get(turn_event_key(sid, event))),
            })
        for gap in response_gaps:
            gap_rows.append({
                **common,
                **gap,
                **gate_gap_verdict(
                    gap_verdicts.get(gap_key(sid, gap.get("gap_id")))),
            })
    event_columns = [
        "session_id", "participant_id", "condition", "analysis_included",
        "valid_for_confirmatory_timing_analysis",
        "valid_for_manual_turn_verification", "crosswalk_complete",
        "participant_raw_path", "assistant_model_path",
        "episode_id", "event_key", "participant_interval", "assistant_interval",
        "initiator",
        "participant_onset_ms", "participant_offset_ms", "assistant_onset_ms",
        "assistant_offset_ms", "overlap_start_ms", "overlap_end_ms",
        "overlap_duration_ms", "overlap_200ms_candidate",
        "participant_barge_in_candidate", "assistant_premature_onset_candidate",
        "assistant_stop_latency_ms_candidate",
        "participant_stop_latency_ms_candidate", "legacy_reconstruction",
        "verified_overlap", "verified_participant_barge_in",
        "verified_assistant_premature_onset", "successful_assistant_yielding",
        "disruptive_assistant_interruption", "assistant_backchannel_onset",
        "verified_assistant_stop_latency_ms",
        "verified_participant_stop_latency_ms", "verifier_initials",
        "verification_note",
    ]
    gap_columns = [
        "session_id", "participant_id", "condition", "analysis_included",
        "valid_for_confirmatory_timing_analysis",
        "valid_for_manual_turn_verification", "crosswalk_complete",
        "participant_raw_path", "assistant_model_path",
        "gap_id", "direction", "from_speaker", "to_speaker", "from_interval",
        "to_interval", "gap_start_ms", "gap_end_ms", "gap_duration_ms",
        "schema", "verified_positive_gap", "verified_gap_start_ms",
        "verified_gap_end_ms", "verified_gap_duration_ms",
        "verifier_initials", "verification_note",
    ]
    # Below the prespecified 200 ms of summed overlap there is no simultaneous
    # speech to affirm - the detectors work in 20 ms frames - so such episodes
    # stay in turn_events.csv as audit rows but are not queued for listening.
    verification_rows = [
        row for row in event_rows
        if (row.get("analysis_included")
            and row.get("valid_for_manual_turn_verification")
            and row.get("crosswalk_complete")
            and float(row.get("overlap_duration_ms") or 0) >= OVERLAP_MINIMUM_MS)
    ]
    gap_verification_rows = [
        row for row in gap_rows
        if (row.get("analysis_included")
            and row.get("valid_for_manual_turn_verification")
            and row.get("crosswalk_complete"))
    ]
    session_review_rows = []
    for session in analytical:
        sid = str(session["session_id"])
        timing = timing_by_session.get(sid, {})
        integrity = timing.get("integrity") or {}
        if not (session.get("analysis_included")
                and _manual_turn_verification_eligible(session, timing)
                and integrity.get("crosswalk_complete")):
            continue
        files = session.get("files") or {}
        session_review_rows.append({
            "session_id": sid,
            "participant_id": session.get("participant_id"),
            "condition": session.get("voice_condition"),
            "participant_raw_path": files.get("participant_raw"),
            "assistant_model_path": files.get("model"),
            "full_session_reviewed": None,
            "additional_event_count": None,
            "verifier_initials": None,
            "verification_note": None,
        })
    session_review_columns = [
        "session_id", "participant_id", "condition", "participant_raw_path",
        "assistant_model_path", "full_session_reviewed",
        "additional_event_count", "verifier_initials", "verification_note",
    ]
    manual_addition_columns = [
        "session_id", "participant_id", "condition", "manual_event_id",
        "participant_onset_ms", "participant_offset_ms", "assistant_onset_ms",
        "assistant_offset_ms", "overlap_start_ms", "overlap_end_ms",
        "overlap_duration_ms", "verified_overlap",
        "verified_participant_barge_in",
        "verified_assistant_premature_onset", "successful_assistant_yielding",
        "disruptive_assistant_interruption",
        "verified_assistant_stop_latency_ms",
        "verified_participant_stop_latency_ms", "verifier_initials",
        "verification_note",
    ]
    vc_quality_columns = [
        "session_id", "participant_id", "condition", "analysis_included",
        "canonical_run", "canonical_attempt", "target_ref",
        "target_speaker_id", "vc_quality_storage_status",
        "vc_quality_result_status", "analysis_id", "metric_profile",
        "region_index", "route_mode", "input_sample_rate_hz",
        "input_start_sample", "input_end_sample", "input_start_s", "input_end_s",
        "transmitted_start_sample", "transmitted_end_sample", "inference_windows",
        "stable_guard_s", "score_selection", "speech_intervals",
        "boundary_padding_s", "source_score_duration_s",
        "transmitted_score_duration_s", "source_score_path",
        "source_score_sha256", "transmitted_score_path",
        "transmitted_score_sha256", "target_path", "target_sha256",
        "target_duration_s", "wer", "wer_status", "wer_error",
        "wer_reference_kind", "sim", "sim_status", "sim_error", "utmos",
        "utmos_status", "utmos_error", "session_error",
        "result_artifact_path", "result_artifact_sha256",
    ]

    answer_columns = ["participant_id", "run_id", "session_id", "kind",
                      "question_id", "value", "answered_at_unix_s"]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "participants.csv", participant_rows, participant_columns)
    _write_csv(out_dir / "scenarios.csv", scenario_rows, scenario_columns)
    _write_csv(out_dir / "units.csv", unit_rows, unit_columns)
    _write_csv(out_dir / "repairs.csv", repair_rows, repair_columns)
    _write_csv(out_dir / "turn_events.csv", event_rows, event_columns)
    _write_csv(out_dir / "turn_verification_queue.csv", verification_rows,
               event_columns)
    _write_csv(out_dir / "turn_gaps.csv", gap_rows, gap_columns)
    _write_csv(out_dir / "turn_gap_verification_queue.csv",
               gap_verification_rows, gap_columns)
    _write_csv(out_dir / "turn_session_review_queue.csv", session_review_rows,
               session_review_columns)
    _write_csv(out_dir / "turn_event_manual_additions.csv", [],
               manual_addition_columns)
    _write_csv(out_dir / "vc_quality_regions.csv", vc_quality_rows,
               vc_quality_columns)
    _write_csv(out_dir / "answers_long.csv", long_rows, answer_columns)
    (out_dir / "DATA_DICTIONARY.md").write_text(_data_dictionary(study))

    warnings = _coverage_warnings(coverage, len(analytical))
    for warning in warnings:
        print(f"[dataset_export] WARNING: {warning}", file=sys.stderr)

    summary = {
        "study_id": study_id,
        "study_name": study.get("name"),
        "out_dir": str(out_dir),
        "artifact_coverage": dict(sorted(coverage.items())),
        "warnings": warnings,
        "participants": len(participant_rows),
        "analytical_sessions": len(scenario_rows),
        "analysis_included_sessions": sum(
            1 for row in scenario_rows if row.get("analysis_included")),
        "coded_sessions": len([r for r in scenario_rows if r.get("outcome_level")]),
        "unit_rows": len(unit_rows),
        "repair_rows": len(repair_rows),
        "turn_event_candidates": len(event_rows),
        "turn_events_verified": sum(
            1 for row in event_rows if row.get("verifier_initials")),
        "turn_events_pending_verification": sum(
            1 for row in verification_rows if not row.get("verifier_initials")),
        "turn_verification_candidates": len(verification_rows),
        "turn_gap_verification_candidates": len(gap_verification_rows),
        "turn_gaps_verified": sum(
            1 for row in gap_rows if row.get("verifier_initials")),
        "turn_gaps_pending_verification": sum(
            1 for row in gap_verification_rows if not row.get("verifier_initials")),
        "turn_sessions_requiring_full_review": len(session_review_rows),
        "overlap_200ms_candidates": sum(
            bool(row.get("overlap_200ms_candidate")) for row in event_rows),
        "participant_barge_in_candidates": sum(
            bool(row.get("participant_barge_in_candidate")) for row in event_rows),
        "assistant_premature_onset_candidates": sum(
            bool(row.get("assistant_premature_onset_candidate"))
            for row in event_rows),
        "positive_response_gap_candidates": len(gap_rows),
        "vc_quality_region_rows": len(vc_quality_rows),
        "vc_quality_complete_region_rows": sum(
            row.get("wer_status") == "complete"
            and row.get("sim_status") == "complete"
            and row.get("utmos_status") == "complete"
            for row in vc_quality_rows),
        "vc_quality_included_incomplete_sessions": len({
            str(row["session_id"])
            for row in vc_quality_rows
            if (row.get("analysis_included")
                and row.get("vc_quality_result_status") != "complete")
        }),
        "generated_at_unix_s": time.time(),
    }
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _data_dictionary(study: dict) -> str:
    return f"""# Data dictionary — {study.get('name') or 'study'}

Missing-value convention: an EMPTY cell means missing/inapplicable (an
unobservable stage, an uncoded session, or a not-yet-run pipeline step); an
explicit `0` means an observed opportunity in which the criterion failed.
Booleans are 1/0. List values are `;`-joined.

## participants.csv
One row per issued participant code. `bg_*` columns are the background
questionnaire answers (latest submission). `variant_id` is the
counterbalancing configuration; `target_ref` the assigned conversion target;
`allocation_stratum` the gender-derived assignment stratum.

## scenarios.csv
One row per analytical session ATTEMPT (practice excluded). Model-ready
filter for condition-effect analyses: `analysis_included == 1` (canonical
submitted run, canonical attempt, technically valid); the sensitivity
analysis additionally drops participants having any attempt with
`valid_for_condition_analysis == 0`. `analytical_position` is the 1-4
ordinal position among the participant's analytical scenarios.
`overlap_candidates_200ms`, `participant_barge_in_candidates`,
`assistant_premature_onset_candidates`, and stop-latency values are automatic
nominations only. Participant barge-in means participant speech began while
the assistant was speaking; assistant premature onset is the reverse.
`assistant_response_gap_candidates` and `participant_response_gap_candidates`
count positive, silence-bounded speaker changes. The method retains nominated
overlap, directional onset, and stop-latency events only after manual
verification. Coded outcome columns are merged from the coding
pipeline's final labels and stay empty until `python -m study.coding
finalize` has run: `outcome_level` (1-4), `demonstrated_grounding`
(both units incorporated AND retained), `false_update_confirmation`
(update claim followed by failed incorporation/retention; empty when no
update claim occurred), `repair_total`, `repair_post_boundary` (repairs at
or after the route switch, or the matched 45 s checkpoint in stable
conditions — see `coding_boundary_kind`). `post_*` columns are the
post-scenario questionnaire.

## units.csv
One row per critical-information unit per CODED session. Grounding stages
(`acknowledgement`, `update_claim`, `incorporation`, `retention`) are empty
when the unit was not completely delivered OR not completely transmitted
(codebook zero-vs-missing rule; `grounding_gated_reason` names the second
case). `complete_transmitted` is computed mechanically from the
transmitted-track transcript on the same intervals as the raw transcript:
content-word recall (`transmitted_content_recall`) at or above the frozen
threshold codes 1; complete transmitted ASR below it codes 0; an
unavailable transmitted transcript or failed ASR leaves it empty with
`complete_transmitted_reason` set, and `complete_raw` gates as proxy.

## repairs.csv
One row per coded repair move. `post_boundary` uses the same boundary as
`repair_post_boundary` above.

## turn_events.csv
One row per linked participant/assistant overlap episode. `event_key` names an
episode by session and onset times and is what verdicts are recorded against,
so they survive a re-analysis; `episode_id` is positional within one export.
The automatic flags
separate neutral overlap (at least 200 ms), participant-initiated barge-in, and
assistant-initiated premature onset without duplicating one episode across
rows. Fill the applicable `verified_*` fields while listening to the
raw-participant and assistant tracks. `successful_assistant_yielding` applies
only to verified participant barge-ins and records whether the assistant ceded
the floor. `disruptive_assistant_interruption` applies only to verified
assistant premature onsets and records whether that onset cut off or disrupted
the participant. `assistant_backchannel_onset` applies to the same verified
premature onsets and records the cooperative case: the early onset was a
continuer ("mm-hm", "okay") rather than an attempt to take the turn, so the
two are expected to be mutually exclusive on any one event. These are manual
judgments, not threshold-derived labels.
`participant_raw_path` and `assistant_model_path` identify the two tracks to
inspect. Only verified measures enter confirmatory analysis.

## turn_verification_queue.csv
The subset of `turn_events.csv` belonging to canonical, analysis-included
sessions that are eligible for manual turn verification, have a complete
browser capture/playback crosswalk, and carry at least the prespecified
200 ms of summed overlap - below that the detectors (20 ms frames) offer no
simultaneous speech to affirm, so verified stop latencies are conditional on
an overlap of at least 200 ms. This is the worksheet to annotate; the
full event table remains an audit trail for excluded and superseded attempts.

## turn_gaps.csv
Positive silence gaps at unambiguous speaker changes. Direction is
`participant_to_assistant` for assistant response gaps and
`assistant_to_participant` for participant response gaps. These are derived
from the same participant-experienced browser clock as the overlap episodes.
The verification queue carries empty `verified_*` fields for manual review.

## turn_session_review_queue.csv / turn_event_manual_additions.csv
Every timing-eligible included session receives a full-track review row. This
review is the false-negative check for events not nominated automatically.
Record the number of missed events and add them to the additions table. Run
`python -m study.turn_verification --dataset <export-dir>` after all event,
gap, and session-review rows are complete. The command validates applicable
fields, preserves input hashes, and writes adjudicated and verified outputs;
automatic candidates never become final outcomes merely by being present.

## vc_quality_regions.csv
One row per converted route region in every analytical attempt. Failed or
partial scoring remains visible as an explicit row with status/error fields and
empty metric cells. `WER` compares the converted-region ASR with raw-source ASR
for the same participant speech (`wer_reference_kind` records that provenance);
`SIM` compares converted speech with the participant's frozen target voice; and
`UTMOS` estimates converted-speech naturalness. Input offsets are samples on the
16 kHz proxy input timeline. `score_selection`, speech-interval count, guard,
paths, and SHA-256 hashes identify the exact derived audio scored. Filter
`analysis_included == 1` for the canonical condition analysis, and report
metric-specific missingness rather than treating unavailable scores as zero.

## answers_long.csv
Every questionnaire answer (all kinds) in long form, one row per question.
"""


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m study.dataset_export")
    parser.add_argument("--study-id", type=int,
                        default=int(os.environ.get("CODING_STUDY_ID", "1")))
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-nomination-only", action="store_true",
                        help="compile the turn review without the full-track "
                             "session review")
    parser.add_argument("--skip-verification", action="store_true",
                        help="do not compile the manual turn review "
                             "into this export")
    parser.add_argument("--carry-verification-from", default=None, type=Path,
                        help="earlier export whose turn-verification review "
                             "should be carried into this one")
    args = parser.parse_args()
    data_root = Path(os.path.expanduser(os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))
    out_dir = Path(args.out) if args.out else (
        data_root / "exports" / f"study{args.study_id}"
        / time.strftime("dataset_%Y%m%dT%H%M%SZ", time.gmtime()))
    summary = build_dataset(args.study_id, out_dir)
    if args.carry_verification_from:
        summary["carried_verification"] = carry_verification(
            args.carry_verification_from, out_dir)
    if not args.skip_verification:
        summary["turn_verification"] = compile_verification(
            out_dir,
            require_full_session_review=not args.allow_nomination_only)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
