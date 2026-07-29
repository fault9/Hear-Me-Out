"""Derived, auditable technical-validity decisions for study sessions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, file_record, sha256_file
from .transition_analysis import read_events

TECHNICAL_VALIDITY_SCHEMA = "hmo.technical-validity.v1"

DEFAULT_THRESHOLDS = {
    "max_route_activation_lag_ms": 250.0,
    "max_estimated_dropped_samples": 0,
    "invalid_end_reasons": ["technical_problem", "artifact_initialization_failed"],
}


def _configured_thresholds(session: dict) -> dict:
    study = (session.get("config_snapshot") or {}).get("study") or {}
    configured = (study.get("settings") or {}).get("technical_validity") or {}
    return {**DEFAULT_THRESHOLDS, **configured}


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
    invalid_end_reasons = set(thresholds.get("invalid_end_reasons") or [])
    end_reason = session.get("end_reason")
    add("end_reason", bool(end_reason) and end_reason not in invalid_end_reasons,
        "The session has a non-technical end reason.",
        observed=end_reason, expected=f"not one of {sorted(invalid_end_reasons)}")

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
    if expected_proxy:
        proxy_names = ["proxy_received.wav", "participant_proxy.wav",
                       "personaplex_input.opus", "proxy_timeline"]
        proxy = _artifact_check(manifest, data_root, proxy_names)
        add("proxy_artifacts_present", not proxy["missing"],
            "All required XVC proxy artifacts are present.",
            observed=proxy["missing"], expected=[])
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
        observed=names.count("stream_stop"), expected=1)
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

    capture = ((timing or {}).get("integrity") or {})
    if timing is not None:
        dropped = capture.get("estimated_dropped_samples")
        max_dropped = int(thresholds["max_estimated_dropped_samples"])
        add("microphone_capture_drops",
            isinstance(dropped, (int, float)) and dropped <= max_dropped,
            "Browser microphone capture stayed within the dropped-sample limit.",
            observed=dropped, expected=f"<= {max_dropped}")
        crosswalk_complete = capture.get("crosswalk_complete")
        add("browser_proxy_crosswalk", crosswalk_complete is True,
            "Browser and proxy packet/sample identifiers form a complete crosswalk.",
            observed=crosswalk_complete, expected=True)
    else:
        add("timing_evidence", False,
            "Timing reconstruction was unavailable, so capture drops and the browser/proxy crosswalk could not be evaluated.",
            warning=True)

    playback = capture.get("playback") or {}
    underrun_total = float(playback.get("queue_underrun_total_ms") or 0.0)
    if underrun_total > 0:
        add("playback_underruns", False,
            "Playback underruns were observed; review the pilot audio before setting an exclusion threshold.",
            observed={"count": playback.get("queue_underrun_count"),
                      "total_ms": underrun_total,
                      "max_ms": playback.get("queue_underrun_max_ms")},
            warning=True)

    for stage, error in (stage_errors or {}).items():
        add(f"{stage}_stage", False, f"The {stage} stage could not complete.",
            observed=error, warning=True)

    evaluation_complete = timing is not None
    condition_valid = ended and evaluation_complete and not failures
    timing_reconstruction_valid = bool(
        condition_valid and capture.get("valid_for_timing") is True)
    timing_status = (timing or {}).get("status")
    confirmatory_timing_valid = bool(
        timing_reconstruction_valid and timing_status == "validated")
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
        "valid_for_timing_reconstruction": timing_reconstruction_valid,
        "valid_for_confirmatory_timing_analysis": confirmatory_timing_valid,
        "speech_boundary_validation_status": timing_status or "unavailable",
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
