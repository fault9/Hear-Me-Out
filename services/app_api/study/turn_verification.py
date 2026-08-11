"""Validate and finalize the manual participant turn-taking review.

The dataset exporter creates blinded-to-hypothesis worksheets but never treats
automatic candidates as outcomes. This module imports those worksheets,
requires a full-track false-negative review, and writes provenance-preserving
adjudicated tables for analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import atomic_write_json, sha256_file
from .turn_taking import OVERLAP_MINIMUM_MS

SCHEMA = "hmo.turn-verification.v1"


class VerificationError(ValueError):
    pass


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise VerificationError(f"missing worksheet: {path.name}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)


def _bool(value: object, *, field: str, required: bool = False) -> bool | None:
    text = str(value or "").strip().lower()
    if not text:
        if required:
            raise VerificationError(f"{field} is required")
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise VerificationError(f"{field} must be yes/no (or 1/0), got {value!r}")


def _float(value: object, *, field: str, required: bool = False) -> float | None:
    text = str(value or "").strip()
    if not text:
        if required:
            raise VerificationError(f"{field} is required")
        return None
    try:
        result = float(text)
    except ValueError as exc:
        raise VerificationError(f"{field} must be numeric, got {value!r}") from exc
    if result < 0:
        raise VerificationError(f"{field} cannot be negative")
    return result


def _integer(value: object, *, field: str, required: bool = False) -> int | None:
    number = _float(value, field=field, required=required)
    if number is None:
        return None
    if not number.is_integer():
        raise VerificationError(f"{field} must be a whole number")
    return int(number)


def _initials(row: dict, key: str) -> str:
    initials = str(row.get("verifier_initials") or "").strip()
    if not initials:
        raise VerificationError(f"{key}: verifier_initials is required")
    return initials


def _event_key(row: dict) -> str:
    return f"{row.get('session_id')}:{row.get('episode_id')}"


def _reviewed_in_full(row: dict) -> bool:
    return (str(row.get("full_session_reviewed") or "").strip().lower()
            in ("1", "true", "yes"))


def _finalize_candidate(row: dict) -> dict:
    key = _event_key(row)
    _initials(row, key)
    overlap_candidate = _bool(
        row.get("overlap_200ms_candidate"), field=f"{key}.overlap_candidate") is True
    barge_candidate = _bool(
        row.get("participant_barge_in_candidate"), field=f"{key}.barge_candidate") is True
    premature_candidate = _bool(
        row.get("assistant_premature_onset_candidate"),
        field=f"{key}.premature_candidate") is True

    overlap = _bool(row.get("verified_overlap"), field=f"{key}.verified_overlap",
                    required=overlap_candidate)
    barge = _bool(row.get("verified_participant_barge_in"),
                  field=f"{key}.verified_participant_barge_in",
                  required=barge_candidate)
    premature = _bool(row.get("verified_assistant_premature_onset"),
                      field=f"{key}.verified_assistant_premature_onset",
                      required=premature_candidate)
    yielding = _bool(row.get("successful_assistant_yielding"),
                     field=f"{key}.successful_assistant_yielding",
                     required=barge is True)
    disruptive = _bool(row.get("disruptive_assistant_interruption"),
                       field=f"{key}.disruptive_assistant_interruption",
                       required=premature is True)
    assistant_stop = _float(
        row.get("verified_assistant_stop_latency_ms"),
        field=f"{key}.verified_assistant_stop_latency_ms")
    participant_stop = _float(
        row.get("verified_participant_stop_latency_ms"),
        field=f"{key}.verified_participant_stop_latency_ms")

    if barge is True and assistant_stop is None:
        assistant_stop = _float(
            row.get("assistant_stop_latency_ms_candidate"),
            field=f"{key}.assistant_stop_latency_ms_candidate", required=True)
    if premature is True and participant_stop is None:
        participant_stop = _float(
            row.get("participant_stop_latency_ms_candidate"),
            field=f"{key}.participant_stop_latency_ms_candidate", required=True)

    final = {
        **row,
        "manual_added": 0,
        "verification_status": "verified_candidate" if any(
            value is True for value in (overlap, barge, premature))
            else "rejected_candidate",
        "final_overlap": int(overlap is True),
        "final_participant_barge_in": int(barge is True),
        "final_assistant_premature_onset": int(premature is True),
        "final_successful_assistant_yielding": (
            int(yielding) if barge is True else ""),
        "final_disruptive_assistant_interruption": (
            int(disruptive) if premature is True else ""),
        "final_overlap_duration_ms": (
            row.get("overlap_duration_ms") if overlap is True else ""),
        "final_assistant_stop_latency_ms": (
            assistant_stop if barge is True else ""),
        "final_participant_stop_latency_ms": (
            participant_stop if premature is True else ""),
    }
    return final


def _finalize_addition(row: dict, eligible_sessions: set[str]) -> dict:
    sid = str(row.get("session_id") or "").strip()
    event_id = str(row.get("manual_event_id") or "").strip()
    key = f"{sid}:{event_id or '<missing>'}"
    if sid not in eligible_sessions:
        raise VerificationError(f"{key}: session is not in the eligible review queue")
    if not event_id:
        raise VerificationError(f"{key}: manual_event_id is required")
    _initials(row, key)
    overlap = _bool(row.get("verified_overlap"), field=f"{key}.verified_overlap")
    barge = _bool(row.get("verified_participant_barge_in"),
                  field=f"{key}.verified_participant_barge_in")
    premature = _bool(row.get("verified_assistant_premature_onset"),
                      field=f"{key}.verified_assistant_premature_onset")
    if not any(value is True for value in (overlap, barge, premature)):
        raise VerificationError(f"{key}: at least one verified event type is required")
    yielding = _bool(row.get("successful_assistant_yielding"),
                     field=f"{key}.successful_assistant_yielding",
                     required=barge is True)
    disruptive = _bool(row.get("disruptive_assistant_interruption"),
                       field=f"{key}.disruptive_assistant_interruption",
                       required=premature is True)
    overlap_duration = _float(row.get("overlap_duration_ms"),
                              field=f"{key}.overlap_duration_ms",
                              required=overlap is True)
    if overlap is True and overlap_duration < OVERLAP_MINIMUM_MS:
        raise VerificationError(
            f"{key}: verified overlap must be at least "
            f"{OVERLAP_MINIMUM_MS:g} ms")
    assistant_stop = _float(row.get("verified_assistant_stop_latency_ms"),
                            field=f"{key}.assistant_stop_latency_ms",
                            required=barge is True)
    participant_stop = _float(row.get("verified_participant_stop_latency_ms"),
                              field=f"{key}.participant_stop_latency_ms",
                              required=premature is True)
    return {
        **row,
        "episode_id": f"manual:{event_id}",
        "manual_added": 1,
        "verification_status": "verified_manual_addition",
        "final_overlap": int(overlap is True),
        "final_participant_barge_in": int(barge is True),
        "final_assistant_premature_onset": int(premature is True),
        "final_successful_assistant_yielding": (
            int(yielding) if barge is True else ""),
        "final_disruptive_assistant_interruption": (
            int(disruptive) if premature is True else ""),
        "final_overlap_duration_ms": overlap_duration if overlap is True else "",
        "final_assistant_stop_latency_ms": assistant_stop if barge is True else "",
        "final_participant_stop_latency_ms": participant_stop if premature is True else "",
    }


def _finalize_gap(row: dict) -> dict:
    key = f"{row.get('session_id')}:{row.get('gap_id')}"
    _initials(row, key)
    verified = _bool(row.get("verified_positive_gap"),
                     field=f"{key}.verified_positive_gap", required=True)
    duration = _float(row.get("verified_gap_duration_ms"),
                      field=f"{key}.verified_gap_duration_ms")
    if verified and duration is None:
        duration = _float(row.get("gap_duration_ms"),
                          field=f"{key}.gap_duration_ms", required=True)
    return {
        **row,
        "verification_status": "verified_gap" if verified else "rejected_gap",
        "final_positive_gap": int(verified),
        "final_gap_start_ms": (
            row.get("verified_gap_start_ms") or row.get("gap_start_ms")
            if verified else ""),
        "final_gap_end_ms": (
            row.get("verified_gap_end_ms") or row.get("gap_end_ms")
            if verified else ""),
        "final_gap_duration_ms": duration if verified else "",
    }


def _summary(events: list[dict], gaps: list[dict],
             session_reviews: list[dict],
             reviewed_sessions: set[str] | None = None) -> list[dict]:
    by_session: dict[str, dict] = defaultdict(lambda: {
        "overlap_count": 0, "overlap_duration_ms": 0.0,
        "participant_barge_in_count": 0, "successful_assistant_yielding_count": 0,
        "assistant_stop_latencies": [], "assistant_premature_onset_count": 0,
        "disruptive_assistant_interruption_count": 0,
        "participant_stop_latencies": [], "assistant_response_gaps": [],
        "participant_response_gaps": [],
    })
    identity: dict[str, dict] = {
        str(row["session_id"]): {key: row.get(key) for key in
                                 ("session_id", "participant_id", "condition")}
        for row in session_reviews
    }
    for row in events:
        sid = str(row["session_id"])
        identity[sid] = {key: row.get(key) for key in
                         ("session_id", "participant_id", "condition")}
        target = by_session[sid]
        if str(row.get("final_overlap")) == "1":
            target["overlap_count"] += 1
            target["overlap_duration_ms"] += float(row["final_overlap_duration_ms"])
        if str(row.get("final_participant_barge_in")) == "1":
            target["participant_barge_in_count"] += 1
            target["successful_assistant_yielding_count"] += int(
                row["final_successful_assistant_yielding"])
            target["assistant_stop_latencies"].append(
                float(row["final_assistant_stop_latency_ms"]))
        if str(row.get("final_assistant_premature_onset")) == "1":
            target["assistant_premature_onset_count"] += 1
            target["disruptive_assistant_interruption_count"] += int(
                row["final_disruptive_assistant_interruption"])
            target["participant_stop_latencies"].append(
                float(row["final_participant_stop_latency_ms"]))
    for row in gaps:
        sid = str(row["session_id"])
        identity[sid] = {key: row.get(key) for key in
                         ("session_id", "participant_id", "condition")}
        if str(row.get("final_positive_gap")) != "1":
            continue
        key = ("assistant_response_gaps" if row.get("direction") ==
               "participant_to_assistant" else "participant_response_gaps")
        by_session[sid][key].append(float(row["final_gap_duration_ms"]))
    rows = []
    for sid in sorted(identity):
        source = by_session[sid]
        barge = source["participant_barge_in_count"]
        premature = source["assistant_premature_onset_count"]
        assistant_gaps = source["assistant_response_gaps"]
        participant_gaps = source["participant_response_gaps"]
        rows.append({
            **identity[sid],
            # Whether the automatic nomination for this session was
            # checked against the full recording. Where it was not, the
            # counts below bound only what nomination found.
            "full_session_reviewed": (
                1 if reviewed_sessions is None or sid in reviewed_sessions
                else 0),
            "overlap_count": source["overlap_count"],
            "overlap_duration_ms": source["overlap_duration_ms"],
            "participant_barge_in_count": barge,
            "successful_assistant_yielding_count": source[
                "successful_assistant_yielding_count"],
            "successful_assistant_yielding_rate": (
                source["successful_assistant_yielding_count"] / barge
                if barge else ""),
            "mean_assistant_stop_latency_ms": (
                sum(source["assistant_stop_latencies"])
                / len(source["assistant_stop_latencies"])
                if source["assistant_stop_latencies"] else ""),
            "assistant_premature_onset_count": premature,
            "disruptive_assistant_interruption_count": source[
                "disruptive_assistant_interruption_count"],
            "disruptive_assistant_interruption_rate": (
                source["disruptive_assistant_interruption_count"] / premature
                if premature else ""),
            "mean_participant_stop_latency_ms": (
                sum(source["participant_stop_latencies"])
                / len(source["participant_stop_latencies"])
                if source["participant_stop_latencies"] else ""),
            "assistant_response_gap_count": len(assistant_gaps),
            "mean_assistant_response_gap_ms": (
                sum(assistant_gaps) / len(assistant_gaps) if assistant_gaps else ""),
            "participant_response_gap_count": len(participant_gaps),
            "mean_participant_response_gap_ms": (
                sum(participant_gaps) / len(participant_gaps)
                if participant_gaps else ""),
        })
    return rows


def finalize(dataset: Path, *,
             require_full_session_review: bool = True) -> dict:
    """Compile the manual turn review for one export.

    The full-track review bounds false negatives: without it a rate is
    "of automatically nominated episodes", not "of episodes". It can be
    waived, but never silently - the manifest and every summary row record
    whether the session behind them was reviewed in full.
    """
    dataset = Path(dataset)
    manifest_path = dataset / "turn_verification_manifest.json"
    if manifest_path.exists():
        raise VerificationError(
            "this dataset export is already finalized; preserve it and use a "
            "new export for any corrected verification pass")
    event_fields, candidates = _read_csv(dataset / "turn_verification_queue.csv")
    gap_fields, gap_candidates = _read_csv(
        dataset / "turn_gap_verification_queue.csv")
    _, session_reviews = _read_csv(dataset / "turn_session_review_queue.csv")
    _, additions = _read_csv(dataset / "turn_event_manual_additions.csv")

    errors: list[str] = []
    reviewed_sessions: set[str] = set()
    nomination_only: set[str] = set()
    expected_additions: Counter = Counter()
    for row in session_reviews:
        sid = str(row.get("session_id") or "")
        if not require_full_session_review and not _reviewed_in_full(row):
            continue
        try:
            _initials(row, sid)
            reviewed = _bool(row.get("full_session_reviewed"),
                             field=f"{sid}.full_session_reviewed", required=True)
            if reviewed is not True:
                raise VerificationError(
                    f"{sid}: full_session_reviewed must be yes before finalization")
            expected_additions[sid] = _integer(
                row.get("additional_event_count"),
                field=f"{sid}.additional_event_count", required=True) or 0
            reviewed_sessions.add(sid)
        except VerificationError as exc:
            errors.append(str(exc))

    adjudicated_events = []
    seen_event_keys = set()
    for row in candidates:
        key = _event_key(row)
        sid = str(row.get("session_id") or "")
        if sid not in reviewed_sessions:
            if require_full_session_review:
                errors.append(
                    f"{key}: session has no completed full-track review")
                continue
            nomination_only.add(sid)
        if key in seen_event_keys:
            errors.append(f"duplicate event key: {key}")
            continue
        seen_event_keys.add(key)
        try:
            adjudicated_events.append(_finalize_candidate(row))
        except VerificationError as exc:
            errors.append(str(exc))

    addition_counts: Counter = Counter()
    for row in additions:
        sid = str(row.get("session_id") or "")
        try:
            addition = _finalize_addition(row, reviewed_sessions)
            key = f"{sid}:{addition['episode_id']}"
            if key in seen_event_keys:
                raise VerificationError(f"duplicate event key: {key}")
            seen_event_keys.add(key)
            adjudicated_events.append(addition)
            addition_counts[sid] += 1
        except VerificationError as exc:
            errors.append(str(exc))
    for sid in sorted(reviewed_sessions):
        if addition_counts[sid] != expected_additions[sid]:
            errors.append(
                f"{sid}: additional_event_count={expected_additions[sid]} but "
                f"{addition_counts[sid]} addition row(s) were supplied")

    adjudicated_gaps = []
    seen_gap_keys = set()
    for row in gap_candidates:
        key = f"{row.get('session_id')}:{row.get('gap_id')}"
        sid = str(row.get("session_id") or "")
        if sid not in reviewed_sessions:
            if require_full_session_review:
                errors.append(
                    f"{key}: session has no completed full-track review")
                continue
            nomination_only.add(sid)
        if key in seen_gap_keys:
            errors.append(f"duplicate gap key: {key}")
            continue
        seen_gap_keys.add(key)
        try:
            adjudicated_gaps.append(_finalize_gap(row))
        except VerificationError as exc:
            errors.append(str(exc))

    report = {
        "schema": SCHEMA,
        "status": "invalid_or_incomplete" if errors else "complete",
        "dataset": str(dataset),
        "sessions_reviewed": len(reviewed_sessions),
        "sessions_nomination_only": len(nomination_only),
        "full_session_review_complete": not nomination_only,
        "candidate_events": len(candidates),
        "manual_additions": len(additions),
        "candidate_gaps": len(gap_candidates),
        "errors": errors,
    }
    (dataset / "turn_verification_preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True))
    if errors:
        return report

    verified_events = [row for row in adjudicated_events if any(
        str(row.get(key)) == "1" for key in (
            "final_overlap", "final_participant_barge_in",
            "final_assistant_premature_onset"))]
    verified_gaps = [row for row in adjudicated_gaps
                     if str(row.get("final_positive_gap")) == "1"]
    event_output_fields = event_fields + [
        "manual_added", "verification_status", "final_overlap",
        "final_participant_barge_in", "final_assistant_premature_onset",
        "final_successful_assistant_yielding",
        "final_disruptive_assistant_interruption",
        "final_overlap_duration_ms", "final_assistant_stop_latency_ms",
        "final_participant_stop_latency_ms",
    ]
    gap_output_fields = gap_fields + [
        "verification_status", "final_positive_gap", "final_gap_start_ms",
        "final_gap_end_ms", "final_gap_duration_ms",
    ]
    summary_rows = _summary(verified_events, verified_gaps, session_reviews,
                            reviewed_sessions)
    summary_fields = list(summary_rows[0]) if summary_rows else [
        "session_id", "participant_id", "condition"]
    _write_csv(dataset / "turn_events_adjudicated.csv", adjudicated_events,
               event_output_fields)
    _write_csv(dataset / "turn_events_verified.csv", verified_events,
               event_output_fields)
    _write_csv(dataset / "turn_gaps_adjudicated.csv", adjudicated_gaps,
               gap_output_fields)
    _write_csv(dataset / "turn_gaps_verified.csv", verified_gaps,
               gap_output_fields)
    _write_csv(dataset / "turn_session_summary_verified.csv", summary_rows,
               summary_fields)
    input_names = [
        "turn_verification_queue.csv", "turn_gap_verification_queue.csv",
        "turn_session_review_queue.csv", "turn_event_manual_additions.csv",
    ]
    output_names = [
        "turn_events_adjudicated.csv", "turn_events_verified.csv",
        "turn_gaps_adjudicated.csv", "turn_gaps_verified.csv",
        "turn_session_summary_verified.csv",
    ]
    manifest = {
        **report,
        "status": "complete",
        "finalized_at_unix_s": time.time(),
        "verified_events": len(verified_events),
        "verified_gaps": len(verified_gaps),
        "input_sha256": {name: sha256_file(dataset / name) for name in input_names},
        "outputs": output_names,
        "output_sha256": {
            name: sha256_file(dataset / name) for name in output_names
        },
    }
    atomic_write_json(manifest_path, manifest, exclusive=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m study.turn_verification")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--allow-nomination-only", action="store_true",
                        help="finalize without the full-track session review; "
                             "rates then bound nominated episodes only")
    args = parser.parse_args()
    result = finalize(args.dataset,
                      require_full_session_review=not args.allow_nomination_only)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
