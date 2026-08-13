"""Derived, auditable technical-validity decisions for study sessions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, file_record, sha256_file
from .continuity import run_for_session_dir as run_continuity_check
from .dialogue_transcript import captured_duration_ms
from .transition_analysis import read_events

TECHNICAL_VALIDITY_SCHEMA = "hmo.technical-validity.v10"
BOUNDARY_CONFIRMATION_STATUS = "confirmed_for_candidate_nomination"

DEFAULT_THRESHOLDS = {
    "max_route_activation_lag_ms": 250.0,
    "max_capture_gap_total_ms": 50.0,
    "max_capture_gap_ms": 10.0,
    "max_manual_verification_gap_ms": 50.0,
    "max_capture_time_base_ratio": 1.25,
    "invalid_end_reasons": ["technical_problem", "artifact_initialization_failed"],
}


def _end_reason_reclassification(session: dict, data_root: Path) -> dict | None:
    """A recorded decision that a participant-selected technical end reason
    was not corroborated by the objective record.

    The end-reason button is the participant's report, and a report can be
    wrong in both directions: one session's record showed the assistant dead
    from 241 s, corroborating it, while another ran full length with clean
    playback and a questionnaire that contradicted it. The file holds the
    evidence for each reclassified session, so the decision is auditable and
    never depends on how the recording sounds to a listener."""
    study_id = session.get("study_id")
    path = (Path(data_root) / "review" / f"study{int(study_id or 0)}"
            / "end_reason_reclassifications.json")
    if not path.is_file():
        return None
    try:
        records = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    record = records.get(str(session.get("session_id")))
    return record if isinstance(record, dict) else None


def _configured_thresholds(session: dict) -> dict:
    study = (session.get("config_snapshot") or {}).get("study") or {}
    configured = (study.get("settings") or {}).get("technical_validity") or {}
    return {**DEFAULT_THRESHOLDS, **configured}


def _analysis_checkpoint_s(session: dict) -> float | None:
    study = (session.get("config_snapshot") or {}).get("study") or {}
    value = (study.get("settings") or {}).get("analysis_checkpoint_s")
    try:
        checkpoint_s = float(value)
    except (TypeError, ValueError):
        return None
    return checkpoint_s if checkpoint_s > 0 else None


def _delivery_continuity(session: dict, data_root: Path) -> dict | None:
    events_path = _event_path(session, data_root)
    if not events_path or not events_path.is_file():
        return None
    try:
        return run_continuity_check(str(events_path.parent))
    except Exception:  # noqa: BLE001 - missing evidence must never be fatal
        return None


def _expected_modes(schedule: list[dict]) -> list[str]:
    modes: list[str] = []
    for segment in schedule or [{"mode": "natural"}]:
        mode = str(segment.get("mode") or "natural")
        if not modes or modes[-1] != mode:
            modes.append(mode)
    return modes


def _sequence_gaps(events: list[dict]) -> list[int]:
    sequences = sorted({int(row.get("event_sequence") or 0) for row in events
                        if int(row.get("event_sequence") or 0) > 0})
    if not sequences:
        return []
    present = set(sequences)
    return [value for value in range(1, sequences[-1] + 1) if value not in present]


def _event_path(session: dict, data_root: Path) -> Path | None:
    record = ((session.get("artifact_manifest") or {}).get("artifacts") or {}).get("events")
    if isinstance(record, dict) and record.get("path"):
        return data_root / record["path"]
    files = session.get("files") or {}
    for key in ("participant_raw", "participant"):
        if files.get(key):
            return (data_root / files[key]).parent / "events.jsonl"
    return None


def _artifact_check(manifest: dict, data_root: Path, names: list[str]) -> dict:
    artifacts = manifest.get("artifacts") or {}
    missing: list[str] = []
    hash_mismatches: list[str] = []
    unhashed: list[str] = []
    for name in names:
        record = artifacts.get(name)
        if not isinstance(record, dict) or not record.get("path"):
            missing.append(name)
            continue
        path = data_root / record["path"]
        if not path.is_file():
            missing.append(name)
            continue
        expected_hash = record.get("sha256")
        if expected_hash:
            if sha256_file(path) != expected_hash:
                hash_mismatches.append(name)
        else:
            unhashed.append(name)
    return {"missing": missing, "hash_mismatches": hash_mismatches,
            "unhashed": unhashed}


def evaluate_technical_validity(session: dict, data_root: Path,
                                timing: dict | None = None,
                                stage_errors: dict[str, str] | None = None) -> dict:
    """Evaluate frozen artifacts without modifying them.

    The result separates condition-analysis validity from timing reconstruction.
    Human validation of speech boundaries remains a later, study-level gate.
    """
    manifest = session.get("artifact_manifest") or {}
    schedule = session.get("schedule") or []
    thresholds = _configured_thresholds(session)
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}

    def add(code: str, passed: bool, message: str, *, observed: Any = None,
            expected: Any = None, warning: bool = False) -> None:
        row = {"passed": bool(passed), "message": message}
        if observed is not None:
            row["observed"] = observed
        if expected is not None:
            row["expected"] = expected
        checks[code] = row
        if not passed:
            (warnings if warning else failures).append({"code": code, **row})

    ended = session.get("ended_at") is not None
    add("session_finalized", ended, "The session has an end timestamp.")
    finalization = manifest.get("finalization")
    if isinstance(finalization, dict):
        add(
            "artifact_finalization",
            finalization.get("status") == "complete",
            "The ended session's immutable browser and proxy artifacts were completely sealed.",
            observed=finalization.get("status"),
            expected="complete",
        )
    invalid_end_reasons = set(thresholds.get("invalid_end_reasons") or [])
    end_reason = session.get("end_reason")
    end_reason_ok = bool(end_reason) and end_reason not in invalid_end_reasons
    reclassified = (None if end_reason_ok
                    else _end_reason_reclassification(session, data_root))
    add("end_reason", end_reason_ok or reclassified is not None,
        "The session has a non-technical end reason.",
        observed=end_reason,
        expected=f"not one of {sorted(invalid_end_reasons)}")
    if reclassified is not None:
        add("end_reason_reclassified", False,
            "A participant-selected technical end reason was reclassified as "
            "uncorroborated; the recorded evidence accompanies this warning.",
            observed=reclassified, warning=True)

    browser_names = ["participant", "participant_raw", "model", "merged",
                     "client_timeline", "events"]
    vc_segments = [segment for segment in schedule if segment.get("mode") == "vc"]
    if vc_segments:
        browser_names.append("target")
    browser = _artifact_check(manifest, data_root, browser_names)
    add("browser_artifacts_present", not browser["missing"],
        "All required browser-side artifacts are present.",
        observed=browser["missing"], expected=[])
    add("browser_artifact_hashes", not browser["hash_mismatches"],
        "Browser-side artifact hashes match the immutable manifest.",
        observed=browser["hash_mismatches"], expected=[])
    if browser["unhashed"]:
        add("browser_artifact_hash_coverage", False,
            "Some browser-side artifacts do not yet have frozen hashes.",
            observed=browser["unhashed"], expected=[], warning=True)

    software = manifest.get("software") or {}
    missing_versions = [key for key in ("hmo_commit", "xvc_commit", "personaplex_version")
                        if not software.get(key)]
    if missing_versions:
        add("software_provenance", False,
            "Some frozen software-version identifiers are missing.",
            observed=missing_versions, expected=[], warning=True)

    expected_proxy = (session.get("config_snapshot") or {}).get("engine") == "xvc"
    # Failed end-of-call delivery is infrastructure loss, not an unusable
    # capture: a certified gapless transmitted span covered by the monitor copy
    # downgrades these checks to warnings. Timing stays excluded regardless.
    proxy_delivery_incomplete = False
    delivery_continuity = None
    delivery_verified = False
    if expected_proxy:
        proxy_names = ["proxy_received.wav", "participant_proxy.wav",
                       "personaplex_input.opus", "proxy_timeline"]
        proxy = _artifact_check(manifest, data_root, proxy_names)
        proxy_delivery_incomplete = bool(proxy["missing"])
        if proxy_delivery_incomplete:
            delivery_continuity = _delivery_continuity(session, data_root)
            delivery_verified = bool(
                delivery_continuity
                and delivery_continuity.get("verdict") == "pass")
        add("proxy_artifacts_present", not proxy["missing"],
            "All required XVC proxy artifacts are present.",
            observed=proxy["missing"], expected=[], warning=delivery_verified)
        add("proxy_artifact_hashes", not proxy["hash_mismatches"],
            "XVC proxy artifact hashes match the immutable manifest.",
            observed=proxy["hash_mismatches"], expected=[])
        if proxy["unhashed"]:
            add("proxy_artifact_hash_coverage", False,
                "Some proxy artifacts do not yet have frozen hashes.",
                observed=proxy["unhashed"], expected=[], warning=True)

    events_path = _event_path(session, data_root)
    events = read_events(events_path) if events_path and events_path.is_file() else []
    names = [row.get("event") for row in events]
    add("event_stream_present", bool(events), "The proxy event stream is present.")
    add("stream_start", names.count("stream_start") == 1,
        "Exactly one stream-start event was recorded.",
        observed=names.count("stream_start"), expected=1)
    add("stream_stop", names.count("stream_stop") == 1,
        "Exactly one stream-stop event was recorded.",
        observed=names.count("stream_stop"), expected=1,
        warning=delivery_verified)

    checkpoint_s = _analysis_checkpoint_s(session)
    checkpoint_requested_sample = (
        round(checkpoint_s * 16000) if checkpoint_s is not None else None
    )
    checkpoint_requests = [
        row for row in events if row.get("event") == "analysis_checkpoint_requested"
    ]
    checkpoint_reached_rows = [
        row for row in events if row.get("event") == "analysis_checkpoint_reached"
    ]
    stream_stop_rows = [row for row in events if row.get("event") == "stream_stop"]
    final_input_sample = (
        int(stream_stop_rows[0].get("input_samples") or 0)
        if len(stream_stop_rows) == 1 else None
    )
    if final_input_sample is None and delivery_verified:
        # The same delivery failure truncates the stream before its sealing
        # event, so take the capture's extent from its certified last chunk.
        chunk_ends = [int(row.get("input_end_sample") or 0) for row in events
                      if row.get("event") == "input_chunk"]
        final_input_sample = max(chunk_ends) if chunk_ends else None
    crossed_checkpoint = bool(
        checkpoint_requested_sample is not None
        and final_input_sample is not None
        and final_input_sample >= checkpoint_requested_sample
    )
    add(
        "analysis_checkpoint_configured",
        checkpoint_s is not None,
        "A positive matched analysis checkpoint is frozen in the session configuration.",
        observed=checkpoint_s,
        expected="a positive number of seconds",
        warning=True,
    )
    checkpoint_request_valid = bool(
        checkpoint_requested_sample is not None
        and len(checkpoint_requests) == 1
        and int(checkpoint_requests[0].get("requested_input_sample") or -1)
        == checkpoint_requested_sample
    )
    if checkpoint_s is not None:
        add(
            "analysis_checkpoint_requested",
            checkpoint_request_valid,
            "The proxy recorded exactly one request for the frozen checkpoint.",
            observed=[row.get("requested_input_sample") for row in checkpoint_requests],
            expected=[checkpoint_requested_sample],
            warning=True,
        )

    checkpoint_lag_limit_ms = float(thresholds["max_route_activation_lag_ms"])
    checkpoint_reached_valid = False
    checkpoint_observed: dict[str, Any] = {
        "session_crossed_checkpoint": crossed_checkpoint,
        "final_input_sample": final_input_sample,
        "events": len(checkpoint_reached_rows),
    }
    if len(checkpoint_reached_rows) == 1 and checkpoint_requested_sample is not None:
        reached_row = checkpoint_reached_rows[0]
        reached_sample = int(reached_row.get("input_sample") or -1)
        reached_requested = int(reached_row.get("requested_input_sample") or -1)
        lag_ms = (reached_sample - checkpoint_requested_sample) / 16.0
        checkpoint_observed.update({
            "requested_input_sample": reached_requested,
            "input_sample": reached_sample,
            "lag_ms": lag_ms,
            "route_mode": reached_row.get("route_mode"),
        })
        checkpoint_reached_valid = bool(
            checkpoint_request_valid
            and crossed_checkpoint
            and reached_requested == checkpoint_requested_sample
            and 0 <= lag_ms <= checkpoint_lag_limit_ms
        )
    checks["analysis_checkpoint_reached"] = {
        "passed": checkpoint_reached_valid,
        "message": (
            "The session crossed the matched checkpoint and the proxy recorded it "
            "on the next allowed input-sample boundary."
        ),
        "observed": checkpoint_observed,
        "expected": {
            "requested_input_sample": checkpoint_requested_sample,
            "event_count": 1,
            "lag_ms": f"0 to {checkpoint_lag_limit_ms:g}",
        },
    }
    if crossed_checkpoint and not checkpoint_reached_valid:
        warnings.append({
            "code": "analysis_checkpoint_reached",
            **checks["analysis_checkpoint_reached"],
        })
    gaps = _sequence_gaps(events)
    add("event_sequence", not gaps, "The server event sequence is contiguous.",
        observed=gaps, expected=[])
    invalid_frames = names.count("invalid_input_frame")
    add("input_frames", invalid_frames == 0, "No malformed microphone frames were dropped.",
        observed=invalid_frames, expected=0)
    inference_failures = names.count("inference_failure")
    add("xvc_inference_failures", inference_failures == 0,
        "XVC reported no inference failures.", observed=inference_failures, expected=0)
    proxy_failures = names.count("proxy_artifacts_failed")
    add("proxy_artifact_upload", proxy_failures == 0,
        "The proxy did not report an artifact-upload failure.",
        observed=proxy_failures, expected=0)

    expected_modes = _expected_modes(schedule)
    route_events = [row for row in events if row.get("event") == "route_activated"]
    actual_modes = [str(row.get("to_mode")) for row in route_events]
    add("input_route_sequence", actual_modes == expected_modes,
        "The input route followed the frozen schedule.",
        observed=actual_modes, expected=expected_modes)
    transmitted_events = [row for row in events
                          if row.get("event") == "transmitted_route_activated"]
    transmitted_modes = [str(row.get("to_mode")) for row in transmitted_events]
    add("transmitted_route_sequence", transmitted_modes == expected_modes,
        "The transmitted route followed the frozen schedule.",
        observed=transmitted_modes, expected=expected_modes)

    lag_limit = float(thresholds["max_route_activation_lag_ms"])
    route_lags = []
    for row in route_events[1:]:
        if row.get("requested_start_s") is None:
            continue
        lag_ms = (int(row.get("input_sample") or 0)
                  - round(float(row["requested_start_s"]) * 16000)) / 16.0
        route_lags.append(lag_ms)
    add("route_activation_lag",
        len(route_lags) == max(0, len(expected_modes) - 1)
        and all(0 <= lag <= lag_limit for lag in route_lags),
        "Every requested switch activated on the next allowed input boundary.",
        observed=route_lags, expected=f"0 to {lag_limit:g} ms")

    inference_batches = [row for row in events
                         if row.get("event") == "xvc_inference_batch"
                         and int(row.get("inference_windows") or 0) > 0]
    vc_regions_without_inference = []
    for index, segment in enumerate(vc_segments):
        start = round(float(segment.get("start_s") or 0) * 16000)
        end_s = segment.get("end_s")
        end = round(float(end_s) * 16000) if end_s is not None else None
        if not any(int(row.get("input_end_sample") or 0) > start
                   and (end is None or int(row.get("input_start_sample") or 0) < end)
                   for row in inference_batches):
            vc_regions_without_inference.append(index)
    add("xvc_route_executed", not vc_regions_without_inference,
        "Every scheduled VC region contains actual XVC inference.",
        observed=vc_regions_without_inference, expected=[])

    # Sustained buffer starvation costs the recording whole seconds, so the
    # file holds less time than it spans and plays back fast. The transcript
    # remaps around a shortfall, but nothing recovers a session that lost a
    # third of its time base - and the model heard that audio too, so the
    # condition it ran under is not the condition that was assigned.
    time_base_ratio = None
    if timing is not None:
        assistant_ends = [float(row["end_ms"])
                          for row in (timing.get("assistant_intervals") or [])
                          if row.get("end_ms") is not None]
        raw_relative = (session.get("files") or {}).get("participant_raw")
        held_ms = (captured_duration_ms(Path(data_root) / raw_relative)
                   if raw_relative else 0.0)
        if assistant_ends and held_ms > 0:
            time_base_ratio = max(assistant_ends) / held_ms
    time_base_limit = float(thresholds["max_capture_time_base_ratio"])
    add("capture_time_base", time_base_ratio is None
        or time_base_ratio <= time_base_limit,
        "The participant recording holds the time it spans.",
        observed=time_base_ratio, expected=f"<= {time_base_limit:g}")

    capture = ((timing or {}).get("integrity") or {})
    capture_gap_valid = False
    capture_gap_manual_valid = False
    if timing is not None:
        gaps = capture.get("capture_gaps") or {}
        total_gap_ms = gaps.get("total_gap_ms")
        max_gap_ms = gaps.get("max_gap_ms")
        total_limit = float(thresholds["max_capture_gap_total_ms"])
        single_limit = float(thresholds["max_capture_gap_ms"])
        manual_limit = float(thresholds["max_manual_verification_gap_ms"])
        # Dropped blocks leave silence at the correct place - the aligned WAV
        # places every chunk by its own timeline stamp - so a session total
        # measures accumulated deficit, not displacement. What can hide speech
        # is one long gap, which is why manual verification is gated on the
        # largest gap alone and automatic measurement still carries a budget.
        capture_gap_manual_valid = (
            isinstance(max_gap_ms, (int, float)) and max_gap_ms <= manual_limit)
        capture_gap_valid = (
            isinstance(total_gap_ms, (int, float))
            and isinstance(max_gap_ms, (int, float))
            and total_gap_ms <= total_limit and max_gap_ms <= single_limit
        )
        add("microphone_capture_drops", capture_gap_valid,
            "Reconstructed microphone sample-boundary gaps stayed within the pilot-calibrated timing limits.",
            observed={
                "gap_count": gaps.get("gap_count"),
                "total_gap_ms": total_gap_ms,
                "max_gap_ms": max_gap_ms,
                "browser_reported_estimated_dropped_samples": gaps.get(
                    "browser_reported_estimated_dropped_samples"),
            },
            expected={
                "total_gap_ms": f"<= {total_limit:g}",
                "max_gap_ms": f"<= {single_limit:g}",
            }, warning=True)
        crosswalk_complete = capture.get("crosswalk_complete")
        add("browser_proxy_crosswalk", crosswalk_complete is True,
            "Browser and proxy packet/sample identifiers form a complete crosswalk.",
            observed=crosswalk_complete, expected=True,
            warning=delivery_verified)
    else:
        add("timing_evidence", False,
            "Timing reconstruction was unavailable, so capture drops and the browser/proxy crosswalk could not be evaluated.",
            warning=True)

    playback = capture.get("playback") or {}
    underrun_total = float(playback.get("queue_underrun_total_ms") or 0.0)
    if underrun_total > 0:
        audible = ((playback.get("underrun_diagnostics") or {})
                   .get("audible_boundaries") or {})
        add("playback_underruns", False,
            "Playback underruns were observed; audible-boundary gaps are reported separately for technical review.",
            observed={"count": playback.get("queue_underrun_count"),
                      "total_ms": underrun_total,
                      "max_ms": playback.get("queue_underrun_max_ms"),
                      "audible_boundary_count": audible.get("count"),
                      "audible_boundary_total_ms": audible.get("total_ms"),
                      "audible_boundary_max_ms": audible.get("max_ms")},
            warning=True)

    for stage, error in (stage_errors or {}).items():
        add(f"{stage}_stage", False, f"The {stage} stage could not complete.",
            observed=error, warning=True)

    evaluation_complete = timing is not None
    condition_valid = ended and evaluation_complete and not failures
    # Timing measures need the certified crosswalk itself, so an incomplete
    # proxy delivery excludes them however the checks above were recorded.
    timing_reconstruction_valid = bool(
        condition_valid and capture.get("valid_for_timing") is True
        and capture_gap_manual_valid and not proxy_delivery_incomplete)
    timing_status = (timing or {}).get("status")
    manual_turn_verification_valid = bool(
        timing_reconstruction_valid
        and timing_status == BOUNDARY_CONFIRMATION_STATUS)
    # This is session-level eligibility only. Individual automatic candidates
    # still require the event-level verification recorded by dataset_export.
    # Automatic measurement keeps the frozen budget; a listener setting the
    # boundary by ear is itself the remedy for what an accumulated deficit
    # costs, so manual verification does not.
    confirmatory_timing_valid = bool(
        manual_turn_verification_valid and capture_gap_valid)
    post_checkpoint_valid = bool(condition_valid and checkpoint_reached_valid)
    if not ended or (not evaluation_complete and not failures):
        status = "incomplete"
    else:
        status = "valid" if condition_valid else "invalid"
    return {
        "schema": TECHNICAL_VALIDITY_SCHEMA,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session.get("session_id"),
        "status": status,
        "valid_for_condition_analysis": condition_valid,
        "valid_for_post_checkpoint_analysis": post_checkpoint_valid,
        "valid_for_timing_reconstruction": timing_reconstruction_valid,
        "valid_for_manual_turn_verification": manual_turn_verification_valid,
        "valid_for_confirmatory_timing_analysis": confirmatory_timing_valid,
        "speech_boundary_validation_status": timing_status or "unavailable",
        "delivery_continuity": delivery_continuity,
        "analysis_checkpoint_s": checkpoint_s,
        "thresholds": thresholds,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
    }


def prepare_technical_validity(session: dict, data_root: Path, analysis_id: str,
                               timing: dict | None = None,
                               stage_errors: dict[str, str] | None = None) -> dict:
    result = evaluate_technical_validity(session, data_root, timing, stage_errors)
    events_path = _event_path(session, data_root)
    if events_path is None:
        raise FileNotFoundError("cannot locate the session artifact directory")
    out_dir = events_path.parent / "analysis" / "technical_validity" / analysis_id
    out_dir.mkdir(parents=True, exist_ok=False)
    result_path = out_dir / "technical_validity.json"
    atomic_write_json(result_path, result, exclusive=True)
    result["result_artifact"] = file_record(result_path, relative_to=data_root)
    return result
