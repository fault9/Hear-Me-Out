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
  turn_events.csv    nominated overlap/barge-in candidates with an empty
                     `verified` column (manual-verification worksheet; the
                     method retains these events only after verification)
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.coding.packets import coding_root  # noqa: E402
from study.session_scope import annotate_analysis_scopes  # noqa: E402
from study.storage import get_backend  # noqa: E402

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


def _load_artifact(data_root: Path, session: dict, key: str) -> dict | None:
    analysis = (session.get("artifact_manifest") or {}).get("analysis") or {}
    record = analysis.get(key) or {}
    path = record.get("path") if isinstance(record, dict) else None
    if not path:
        return None
    try:
        return json.loads((data_root / path).read_text())
    except (OSError, ValueError):
        return None


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


def build_dataset(study_id: int, out_dir: Path) -> dict:
    data_root = Path(os.path.expanduser(os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))
    backend = get_backend()
    study = backend.get_study(study_id) or {}
    participants = backend.list_participants(study_id)
    scenarios = {str(row["id"]): row for row in backend.list_scenarios(study_id)}
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
        timing = _load_artifact(data_root, session, "timing_latest") or {}
        timing_summary = timing.get("summary") or {}
        integrity = timing.get("integrity") or {}
        capture_gaps = integrity.get("capture_gaps") or {}
        scenario = scenarios.get(str(session.get("scenario_id"))) or {}
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
            "technical_failure_codes": [f.get("code")
                                        for f in validity.get("failures") or []],
            "participant_speech_intervals": timing_summary.get(
                "participant_speech_intervals"),
            "assistant_speech_intervals": timing_summary.get(
                "assistant_speech_intervals"),
            "overlap_candidates_200ms": timing_summary.get("overlap_events_200ms"),
            "barge_in_candidates": timing_summary.get("barge_in_attempts"),
            "mean_stop_latency_ms_candidates": timing_summary.get("mean_stop_latency_ms"),
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
        "valid_for_confirmatory_timing_analysis", "technical_failure_codes",
        "participant_speech_intervals", "assistant_speech_intervals",
        "overlap_candidates_200ms", "barge_in_candidates",
        "mean_stop_latency_ms_candidates", "crosswalk_complete",
        "capture_gap_total_ms", "capture_gap_max_ms", "vc_quality_status",
        "outcome_level", "final_account_accuracy", "demonstrated_grounding",
        "false_update_confirmation", "any_repair", "repair_total",
        "repair_post_boundary", "coding_boundary_kind", "coding_label_source",
    ] + [f"post_{key}" for key in post_keys])

    # ---- units.csv / repairs.csv ----
    unit_rows, repair_rows = [], []
    for session in analytical:
        sid = str(session["session_id"])
        final = final_labels.get(sid)
        if not final:
            continue
        scenario = scenarios.get(str(session.get("scenario_id"))) or {}
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
                "complete_transmitted_reason": unit.get("complete_transmitted_reason"),
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
                    "complete_transmitted", "complete_transmitted_reason",
                    "delivery_relative_to_boundary", "delivery_utterance_ids",
                    "acknowledgement", "update_claim", "incorporation",
                    "retention"]
    repair_columns = ["session_id", "participant_id", "condition", "move_index",
                      "category", "utterance_id", "post_boundary", "trouble"]

    # ---- turn_events.csv (candidates awaiting manual verification) ----
    event_rows = []
    for session in analytical:
        sid = str(session["session_id"])
        timing = _load_artifact(data_root, session, "timing_latest") or {}
        for kind, events in (("overlap", timing.get("overlaps") or []),
                             ("barge_in", timing.get("barge_ins") or [])):
            for i, event in enumerate(events, start=1):
                event_rows.append({
                    "session_id": sid,
                    "participant_id": session.get("participant_id"),
                    "condition": session.get("voice_condition"),
                    "event_type": kind,
                    "event_index": i,
                    "start_ms": event.get("start_ms"),
                    "end_ms": event.get("end_ms"),
                    "duration_ms": event.get("duration_ms"),
                    "stop_latency_ms": event.get("stop_latency_ms"),
                    "verified": None,   # fill yes/no after listening to the tracks
                    "verifier_initials": None,
                    "verification_note": None,
                })
    event_columns = ["session_id", "participant_id", "condition", "event_type",
                     "event_index", "start_ms", "end_ms", "duration_ms",
                     "stop_latency_ms", "verified", "verifier_initials",
                     "verification_note"]

    answer_columns = ["participant_id", "run_id", "session_id", "kind",
                      "question_id", "value", "answered_at_unix_s"]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "participants.csv", participant_rows, participant_columns)
    _write_csv(out_dir / "scenarios.csv", scenario_rows, scenario_columns)
    _write_csv(out_dir / "units.csv", unit_rows, unit_columns)
    _write_csv(out_dir / "repairs.csv", repair_rows, repair_columns)
    _write_csv(out_dir / "turn_events.csv", event_rows, event_columns)
    _write_csv(out_dir / "answers_long.csv", long_rows, answer_columns)
    (out_dir / "DATA_DICTIONARY.md").write_text(_data_dictionary(study))

    summary = {
        "study_id": study_id,
        "study_name": study.get("name"),
        "out_dir": str(out_dir),
        "participants": len(participant_rows),
        "analytical_sessions": len(scenario_rows),
        "analysis_included_sessions": sum(
            1 for row in scenario_rows if row.get("analysis_included")),
        "coded_sessions": len([r for r in scenario_rows if r.get("outcome_level")]),
        "unit_rows": len(unit_rows),
        "repair_rows": len(repair_rows),
        "turn_event_candidates": len(event_rows),
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
`overlap_candidates_200ms`, `barge_in_candidates`, and
`mean_stop_latency_ms_candidates` are AUTOMATIC nominations only — the
method retains such events solely after manual verification
(turn_events.csv). Coded outcome columns are merged from the coding
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
when the unit was not completely delivered (codebook zero-vs-missing rule).
`complete_transmitted` remains empty with
`complete_transmitted_reason=transmitted_transcript_unavailable` until the
transmitted-track transcription step exists; `complete_raw` acts as the
gating proxy and is recorded as such.

## repairs.csv
One row per coded repair move. `post_boundary` uses the same boundary as
`repair_post_boundary` above.

## turn_events.csv
Overlap/barge-in candidates nominated by the timing analysis. Fill
`verified` (yes/no), `verifier_initials`, and `verification_note` while
listening to the raw-participant and assistant tracks; only verified events
enter the analysis.

## answers_long.csv
Every questionnaire answer (all kinds) in long form, one row per question.
"""


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m study.dataset_export")
    parser.add_argument("--study-id", type=int,
                        default=int(os.environ.get("CODING_STUDY_ID", "1")))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    data_root = Path(os.path.expanduser(os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))
    out_dir = Path(args.out) if args.out else (
        data_root / "exports" / f"study{args.study_id}"
        / time.strftime("dataset_%Y%m%dT%H%M%SZ", time.gmtime()))
    summary = build_dataset(args.study_id, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
