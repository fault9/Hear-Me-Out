"""Reconstruct and verify a certified turn-taking prefix after teardown loss.

This module is intentionally separate from ``dataset_export``.  It never
changes the original timing artifact or the primary turn-analysis cohort.  A
successful reconstruction creates a one-session review bundle that can be
finalized with ``study.turn_verification`` and then combined with the primary
verified tables as an explicitly labelled sensitivity analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import (atomic_write_json, file_record, git_revision,
                        resolve_artifact_path, sha256_file)
from .continuity import check as check_continuity
from .continuity import load_events
from .storage import get_backend
from .technical_validity import (BOUNDARY_CONFIRMATION_STATUS,
                                 DEFAULT_THRESHOLDS)
from .turn_taking import build_positive_response_gaps, build_turn_episodes

RECONSTRUCTION_SCHEMA = "hmo.turn-prefix-reconstruction.v1"
REVIEW_SCHEMA = "hmo.turn-reconstruction-review.v1"
SENSITIVITY_SCHEMA = "hmo.turn-sensitivity-dataset.v1"


class ReconstructionError(ValueError):
    pass


EVENT_COLUMNS = [
    "session_id", "participant_id", "condition", "analysis_included",
    "valid_for_confirmatory_timing_analysis",
    "valid_for_manual_turn_verification", "crosswalk_complete",
    "certified_prefix_crosswalk_complete", "analysis_scope",
    "certified_end_ms", "participant_raw_path", "assistant_model_path",
    "episode_id", "participant_interval", "assistant_interval", "initiator",
    "participant_onset_ms", "participant_offset_ms", "assistant_onset_ms",
    "assistant_offset_ms", "overlap_start_ms", "overlap_end_ms",
    "overlap_duration_ms", "overlap_200ms_candidate",
    "participant_barge_in_candidate", "assistant_premature_onset_candidate",
    "assistant_stop_latency_ms_candidate",
    "participant_stop_latency_ms_candidate", "legacy_reconstruction",
    "verified_overlap", "verified_participant_barge_in",
    "verified_assistant_premature_onset", "successful_assistant_yielding",
    "disruptive_assistant_interruption", "verified_assistant_stop_latency_ms",
    "verified_participant_stop_latency_ms", "verifier_initials",
    "verification_note",
]

GAP_COLUMNS = [
    "session_id", "participant_id", "condition", "analysis_included",
    "valid_for_confirmatory_timing_analysis",
    "valid_for_manual_turn_verification", "crosswalk_complete",
    "certified_prefix_crosswalk_complete", "analysis_scope",
    "certified_end_ms", "participant_raw_path", "assistant_model_path",
    "gap_id", "direction", "from_speaker", "to_speaker", "from_interval",
    "to_interval", "gap_start_ms", "gap_end_ms", "gap_duration_ms", "schema",
    "verified_positive_gap", "verified_gap_start_ms", "verified_gap_end_ms",
    "verified_gap_duration_ms", "verifier_initials", "verification_note",
]

SESSION_COLUMNS = [
    "session_id", "participant_id", "condition", "analysis_scope",
    "certified_end_ms", "excluded_tail_ms", "participant_raw_path",
    "assistant_model_path", "full_session_reviewed", "additional_event_count",
    "verifier_initials", "verification_note",
]

ADDITION_COLUMNS = [
    "session_id", "participant_id", "condition", "manual_event_id",
    "participant_onset_ms", "participant_offset_ms", "assistant_onset_ms",
    "assistant_offset_ms", "overlap_start_ms", "overlap_end_ms",
    "overlap_duration_ms", "verified_overlap",
    "verified_participant_barge_in", "verified_assistant_premature_onset",
    "successful_assistant_yielding", "disruptive_assistant_interruption",
    "verified_assistant_stop_latency_ms",
    "verified_participant_stop_latency_ms", "verifier_initials",
    "verification_note",
]


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise ReconstructionError(f"missing verified table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _checked_record(data_root: Path, record: dict | None, label: str) -> Path:
    if not isinstance(record, dict) or not record.get("path"):
        raise ReconstructionError(f"{label} has no recorded artifact path")
    path = resolve_artifact_path(data_root, record["path"])
    if path is None or not path.is_file():
        raise ReconstructionError(f"{label} artifact cannot be resolved")
    expected = record.get("sha256")
    if not expected:
        raise ReconstructionError(f"{label} artifact has no frozen SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise ReconstructionError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return path


def _session_file(data_root: Path, session: dict, key: str) -> Path:
    value = (session.get("files") or {}).get(key)
    path = resolve_artifact_path(data_root, value)
    if path is None or not path.is_file():
        raise ReconstructionError(f"session file {key!r} cannot be resolved")
    record = ((session.get("artifact_manifest") or {}).get("artifacts") or {}).get(key)
    frozen = _checked_record(data_root, record, f"session file {key}")
    if frozen != path:
        raise ReconstructionError(
            f"session file {key!r} differs from its frozen manifest path")
    return path


def _is_trailing(values: set[int], common_max: int) -> bool:
    return all(value > common_max for value in values)


def _capture_cutoff(capture: dict,
                    proxy_sequences: set[int]) -> tuple[float, dict, set[int]]:
    chunks = {
        int(row["chunk_sequence"]): row
        for row in capture.get("chunks") or []
        if row.get("chunk_sequence") is not None
    }
    client_sequences = set(chunks)
    common = client_sequences & proxy_sequences
    if not common:
        raise ReconstructionError("capture crosswalk has no common packet IDs")
    common_min, common_max = min(common), max(common)
    expected_common = set(range(common_min, common_max + 1))
    if common != expected_common:
        raise ReconstructionError("capture crosswalk has an internal common-prefix hole")
    missing_at_proxy = client_sequences - proxy_sequences
    missing_at_client = proxy_sequences - client_sequences
    if common_min != min(client_sequences) or common_min != min(proxy_sequences):
        raise ReconstructionError("capture crosswalk has leading packet loss")
    if not _is_trailing(missing_at_proxy, common_max):
        raise ReconstructionError("capture crosswalk has internal loss at the proxy")
    if not _is_trailing(missing_at_client, common_max):
        raise ReconstructionError("capture crosswalk has internal loss at the client")

    sample_rate = float(capture.get("sample_rate_hz") or 16000)
    ends = []
    for sequence in common:
        row = chunks[sequence]
        start = row.get("timeline_start_ms")
        count = row.get("sample_count")
        if start is None or count is None:
            raise ReconstructionError(
                f"capture chunk {sequence} lacks timeline/sample boundaries")
        ends.append(float(start) + float(count) * 1000.0 / sample_rate)
    return max(ends), {
        "client_sequences": len(client_sequences),
        "proxy_sequences": len(proxy_sequences),
        "common_sequences": len(common),
        "common_first": common_min,
        "common_last": common_max,
        "client_only_trailing": sorted(missing_at_proxy),
        "proxy_only_trailing": sorted(missing_at_client),
    }, common


def _playback_cutoff(playback: dict,
                     proxy_sequences: set[int]) -> tuple[float, dict, set[int]]:
    packets = {
        int(row["packet_sequence"]): row
        for row in playback.get("assistant_packets") or []
        if row.get("packet_sequence") is not None
    }
    client_sequences = set(packets)
    common = client_sequences & proxy_sequences
    if not common:
        raise ReconstructionError("playback crosswalk has no common packet IDs")
    first_client, last_client = min(client_sequences), max(client_sequences)
    common_max = max(common)
    proxy_only = proxy_sequences - client_sequences
    client_only = client_sequences - proxy_sequences
    internal_proxy_only = {
        value for value in proxy_only if first_client <= value <= last_client
    }
    if internal_proxy_only:
        raise ReconstructionError(
            "playback crosswalk has proxy packets missing inside the playable range")
    if not _is_trailing(client_only, common_max):
        raise ReconstructionError("playback crosswalk has internal client-only packets")
    proxy_only_trailing = {value for value in proxy_only if value > common_max}
    if not _is_trailing(proxy_only_trailing, common_max):
        raise ReconstructionError("playback crosswalk has ambiguous trailing packets")

    ends = []
    for sequence in common:
        end = packets[sequence].get("timeline_end_ms")
        if end is None:
            raise ReconstructionError(
                f"playback packet {sequence} lacks a timeline end")
        ends.append(float(end))
    return max(ends), {
        "client_sequences": len(client_sequences),
        "proxy_sequences": len(proxy_sequences),
        "common_sequences": len(common),
        "common_first": min(common),
        "common_last": common_max,
        "proxy_only_leading_priming": sorted(
            value for value in proxy_only if value < first_client),
        "client_only_trailing": sorted(client_only),
        "proxy_only_trailing": sorted(proxy_only_trailing),
    }, common


def _trim_intervals(rows: list[dict], cutoff_ms: float) -> list[dict]:
    retained = []
    for row in rows:
        start = float(row["start_ms"])
        end = min(float(row["end_ms"]), cutoff_ms)
        if start >= cutoff_ms or end <= start:
            continue
        retained.append({**row, "start_ms": start, "end_ms": end})
    return retained


def _wav_samples(path: Path, target_rate: int = 16000) -> int:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    return frames if rate == target_rate else round(frames * target_rate / rate)


def _write_mono_wav(destination: Path, *, rate: int, sample_width: int,
                    frames: bytes) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as dest:
        dest.setnchannels(1)
        dest.setsampwidth(sample_width)
        dest.setframerate(rate)
        dest.writeframes(frames)
    return file_record(destination, relative_to=destination.parents[1])


def _copy_frames(target: bytearray, source: bytes, *, source_frame: int,
                 target_frame: int, frame_count: int, frame_width: int) -> None:
    if frame_count <= 0:
        return
    source_start = source_frame * frame_width
    source_end = source_start + frame_count * frame_width
    target_start = target_frame * frame_width
    target_end = target_start + frame_count * frame_width
    target[target_start:target_end] = source[source_start:source_end]


def _aligned_participant_wav(source: Path, destination: Path, capture: dict,
                             common_sequences: set[int], cutoff_ms: float,
                             latency_correction_ms: float) -> tuple[dict, dict]:
    with wave.open(str(source), "rb") as src:
        if src.getnchannels() != 1:
            raise ReconstructionError("participant raw WAV must be mono")
        rate = src.getframerate()
        sample_width = src.getsampwidth()
        source_frames = src.getnframes()
        encoded = src.readframes(source_frames)
    capture_rate = int(capture.get("sample_rate_hz") or 16000)
    if rate != capture_rate:
        raise ReconstructionError(
            "participant raw WAV rate differs from the capture timeline")
    output_frames = round(cutoff_ms * rate / 1000.0)
    aligned = bytearray(output_frames * sample_width)
    chunks = sorted(
        (row for row in capture.get("chunks") or []
         if int(row.get("chunk_sequence") or -1) in common_sequences),
        key=lambda row: int(row["chunk_sequence"]),
    )
    for row in chunks:
        source_start = int(row.get("capture_start_sample") or 0)
        count = int(row["sample_count"])
        target_start = round(
            (float(row["timeline_start_ms"]) - latency_correction_ms)
            * rate / 1000.0)
        if source_start < 0 or source_start + count > source_frames:
            raise ReconstructionError(
                f"capture chunk {row['chunk_sequence']} exceeds participant WAV")
        if target_start < 0:
            source_start -= target_start
            count += target_start
            target_start = 0
        count = min(count, output_frames - target_start)
        _copy_frames(aligned, encoded, source_frame=source_start,
                     target_frame=target_start, frame_count=count,
                     frame_width=sample_width)
    record = _write_mono_wav(
        destination, rate=rate, sample_width=sample_width, frames=bytes(aligned))
    return record, {
        "schema": "hmo.browser-clock-participant-audio.v1",
        "source_frames": source_frames,
        "placed_chunks": len(chunks),
        "latency_correction_ms": latency_correction_ms,
        "output_frames": output_frames,
        "sample_rate_hz": rate,
    }


def _aligned_assistant_wav(source: Path, destination: Path, playback: dict,
                           common_sequences: set[int],
                           cutoff_ms: float) -> tuple[dict, dict]:
    with wave.open(str(source), "rb") as src:
        if src.getnchannels() != 1:
            raise ReconstructionError("assistant model WAV must be mono")
        rate = src.getframerate()
        sample_width = src.getsampwidth()
        source_frames = src.getnframes()
        encoded = src.readframes(source_frames)
    packets = sorted(
        playback.get("assistant_packets") or [],
        key=lambda row: int(row.get("packet_sequence") or 0),
    )
    if not packets:
        raise ReconstructionError("playback timeline has no decoded packets")
    packet_rates = {
        int(row.get("sample_rate_hz") or rate) for row in packets
    }
    if packet_rates != {rate}:
        raise ReconstructionError(
            "assistant model WAV rate differs from decoded playback packets")
    first_start_ms = float(packets[0].get("timeline_start_ms") or 0.0)
    source_cursor = round(first_start_ms * rate / 1000.0)
    expected_end = source_cursor + sum(int(row["decoded_samples"]) for row in packets)
    if expected_end > source_frames:
        raise ReconstructionError(
            "decoded packet boundaries exceed the assistant model WAV")
    output_latency_ms = float(playback.get("output_latency_ms") or 0.0)
    output_frames = round(cutoff_ms * rate / 1000.0)
    aligned = bytearray(output_frames * sample_width)
    placed = 0
    for row in packets:
        count = int(row["decoded_samples"])
        sequence = int(row["packet_sequence"])
        if sequence in common_sequences:
            target_start = round(
                (float(row["timeline_start_ms"]) + output_latency_ms)
                * rate / 1000.0)
            count_to_place = min(count, output_frames - target_start)
            if target_start >= 0 and count_to_place > 0:
                _copy_frames(aligned, encoded, source_frame=source_cursor,
                             target_frame=target_start,
                             frame_count=count_to_place,
                             frame_width=sample_width)
                placed += 1
        source_cursor += count
    record = _write_mono_wav(
        destination, rate=rate, sample_width=sample_width, frames=bytes(aligned))
    return record, {
        "schema": "hmo.browser-clock-assistant-audio.v1",
        "source_frames": source_frames,
        "source_initial_offset_frames": round(first_start_ms * rate / 1000.0),
        "decoded_packet_frames": sum(
            int(row["decoded_samples"]) for row in packets),
        "placed_packets": placed,
        "output_latency_ms": output_latency_ms,
        "output_frames": output_frames,
        "sample_rate_hz": rate,
    }


def _thresholds(session: dict) -> dict:
    study = (session.get("config_snapshot") or {}).get("study") or {}
    configured = (study.get("settings") or {}).get("technical_validity") or {}
    return {**DEFAULT_THRESHOLDS, **configured}


def _verify_capture_gaps(session: dict, timing: dict) -> dict:
    gaps = (timing.get("integrity") or {}).get("capture_gaps") or {}
    total = gaps.get("total_gap_ms")
    maximum = gaps.get("max_gap_ms")
    limits = _thresholds(session)
    if not isinstance(total, (int, float)) or not isinstance(maximum, (int, float)):
        raise ReconstructionError("capture-gap diagnostics are incomplete")
    # Only a long gap can hide speech from a listener, so only that blocks
    # reconstruction. The budgets still decide confirmatory eligibility, which
    # technical_validity records from the same numbers.
    if maximum > float(limits["max_manual_verification_gap_ms"]):
        raise ReconstructionError("largest capture gap exceeds the frozen threshold")
    return {
        "observed": gaps,
        "limits": {
            "max_capture_gap_total_ms": limits["max_capture_gap_total_ms"],
            "max_capture_gap_ms": limits["max_capture_gap_ms"],
            "max_manual_verification_gap_ms": limits[
                "max_manual_verification_gap_ms"],
        },
        "within_confirmatory_budget": (
            total <= float(limits["max_capture_gap_total_ms"])
            and maximum <= float(limits["max_capture_gap_ms"])),
    }


def _source_record(timing: dict, key: str) -> dict:
    record = (timing.get("sources") or {}).get(key)
    if not isinstance(record, dict):
        raise ReconstructionError(f"timing artifact lacks source record {key!r}")
    return record


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def reconstruct(session: dict, data_root: Path, out_dir: Path | None = None) -> dict:
    """Create an immutable prefix artifact and portable manual-review bundle."""
    data_root = Path(data_root).resolve()
    sid = str(session.get("session_id") or "")
    if not sid:
        raise ReconstructionError("session has no session_id")
    if session.get("ended_at") is None:
        raise ReconstructionError("session is not finalized")

    analysis = (session.get("artifact_manifest") or {}).get("analysis") or {}
    timing_path = _checked_record(data_root, analysis.get("timing_latest"),
                                  "timing")
    timing = json.loads(timing_path.read_text())
    if timing.get("status") != BOUNDARY_CONFIRMATION_STATUS:
        raise ReconstructionError(
            "speech-boundary validation was not frozen as confirmed")
    client_path = _checked_record(
        data_root, _source_record(timing, "client_timeline"), "client timeline")
    events_path = _checked_record(
        data_root, _source_record(timing, "proxy_events"), "proxy events")
    raw_path = _checked_record(
        data_root, _source_record(timing, "participant_raw"), "participant raw")
    model_path = _checked_record(
        data_root, _source_record(timing, "model"), "assistant model")
    transmitted_path = _session_file(data_root, session, "participant")

    client = json.loads(client_path.read_text())
    events = load_events(events_path)
    continuity_failures, continuity = check_continuity(
        events, _wav_samples(transmitted_path))
    continuity_result = {
        "schema": "hmo.continuity-check.v1",
        "verdict": "pass" if not continuity_failures else "fail",
        "failures": continuity_failures,
        **continuity,
    }
    if continuity_failures:
        raise ReconstructionError(
            "transmitted-audio continuity failed: " + "; ".join(continuity_failures))

    capture_proxy = {
        int(row["browser_chunk_sequence"])
        for row in events
        if row.get("event") == "input_chunk"
        and row.get("browser_chunk_sequence") is not None
    }
    playback_proxy = {
        int(row["packet_sequence"])
        for row in events
        if row.get("event") == "personaplex_output_packet"
        and row.get("tag") == 1 and row.get("packet_sequence") is not None
    }
    capture_cutoff, capture_crosswalk, capture_common = _capture_cutoff(
        client.get("capture") or {}, capture_proxy)
    playback_cutoff, playback_crosswalk, playback_common = _playback_cutoff(
        client.get("playback") or {}, playback_proxy)
    certified_cutoff = min(capture_cutoff, playback_cutoff)
    gap_validation = _verify_capture_gaps(session, timing)

    participant_all = timing.get("participant_intervals") or []
    assistant_all = timing.get("assistant_intervals") or []
    participant = _trim_intervals(participant_all, certified_cutoff)
    assistant = _trim_intervals(assistant_all, certified_cutoff)
    if not participant or not assistant:
        raise ReconstructionError("certified prefix lacks one or both speech tracks")
    episodes = build_turn_episodes(participant, assistant)
    gaps = build_positive_response_gaps(participant, assistant)
    full_end = max(
        [float(row["end_ms"]) for row in participant_all + assistant_all] or [0.0])
    excluded_tail = max(0.0, full_end - certified_cutoff)

    analysis_id = _timestamp()
    if out_dir is None:
        study_id = session.get("study_id")
        out_dir = (data_root / "exports" / f"study{study_id}"
                   / f"turn_reconstruction_{sid}_{analysis_id}")
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    artifact_dir = events_path.parent / "analysis" / "turn_reconstruction" / analysis_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = artifact_dir / "turn_reconstruction.json"
    result: dict[str, Any] = {
        "schema": RECONSTRUCTION_SCHEMA,
        "analysis_id": analysis_id,
        "session_id": sid,
        "status": "eligible_for_reconstructed_turn_sensitivity",
        "analysis_scope": "reconstructed_common_prefix_sensitivity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(Path(__file__).resolve().parents[3]),
        "certification": {
            "capture_cutoff_ms": capture_cutoff,
            "playback_cutoff_ms": playback_cutoff,
            "certified_cutoff_ms": certified_cutoff,
            "excluded_tail_ms": excluded_tail,
            "trailing_only_loss": True,
            "original_crosswalk_complete": bool(
                (timing.get("integrity") or {}).get("crosswalk_complete")),
            "certified_prefix_crosswalk_complete": True,
            "capture_crosswalk": capture_crosswalk,
            "playback_crosswalk": playback_crosswalk,
            "capture_gap_validation": gap_validation,
            "continuity": continuity_result,
        },
        "participant_intervals": participant,
        "assistant_intervals": assistant,
        "turn_episodes": episodes,
        "positive_response_gaps": gaps,
        "summary": {
            "participant_intervals_original": len(participant_all),
            "participant_intervals_retained": len(participant),
            "assistant_intervals_original": len(assistant_all),
            "assistant_intervals_retained": len(assistant),
            "overlap_candidates": len(episodes),
            "overlap_200ms_candidates": sum(
                bool(row["overlap_200ms_candidate"]) for row in episodes),
            "participant_barge_in_candidates": sum(
                bool(row["participant_barge_in_candidate"]) for row in episodes),
            "assistant_premature_onset_candidates": sum(
                bool(row["assistant_premature_onset_candidate"]) for row in episodes),
            "positive_response_gaps": len(gaps),
        },
        "sources": {
            "timing": file_record(timing_path, relative_to=data_root),
            "client_timeline": file_record(client_path, relative_to=data_root),
            "proxy_events": file_record(events_path, relative_to=data_root),
            "participant_raw": file_record(raw_path, relative_to=data_root),
            "assistant_model": file_record(model_path, relative_to=data_root),
            "participant_transmitted": file_record(
                transmitted_path, relative_to=data_root),
        },
        "technical_warnings": {
            "playback": (timing.get("integrity") or {}).get("playback") or {},
        },
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    participant_review = out_dir / "audio" / "participant_browser_clock.wav"
    assistant_review = out_dir / "audio" / "assistant_browser_clock.wav"
    capture = client.get("capture") or {}
    playback = client.get("playback") or {}
    latency_correction = float(
        (timing.get("integrity") or {}).get(
            "participant_capture_latency_correction_ms") or 0.0)
    participant_record, participant_alignment = _aligned_participant_wav(
        raw_path, participant_review, capture, capture_common,
        certified_cutoff, latency_correction)
    assistant_record, assistant_alignment = _aligned_assistant_wav(
        model_path, assistant_review, playback, playback_common,
        certified_cutoff)
    atomic_write_json(artifact_path, result, exclusive=True)
    artifact_record = file_record(artifact_path, relative_to=data_root)
    participant_rel = str(participant_review.relative_to(out_dir))
    assistant_rel = str(assistant_review.relative_to(out_dir))

    common = {
        "session_id": sid,
        "participant_id": session.get("participant_id"),
        "condition": session.get("voice_condition"),
        "analysis_included": False,
        "valid_for_confirmatory_timing_analysis": False,
        "valid_for_manual_turn_verification": True,
        "crosswalk_complete": False,
        "certified_prefix_crosswalk_complete": True,
        "analysis_scope": "reconstructed_common_prefix_sensitivity",
        "certified_end_ms": certified_cutoff,
        "participant_raw_path": participant_rel,
        "assistant_model_path": assistant_rel,
    }
    event_rows = []
    for event in episodes:
        if not any((event.get("overlap_200ms_candidate"),
                    event.get("participant_barge_in_candidate"),
                    event.get("assistant_premature_onset_candidate"))):
            continue
        event_rows.append({
            **common, **event,
            "assistant_stop_latency_ms_candidate": event.get(
                "assistant_stop_latency_ms"),
            "participant_stop_latency_ms_candidate": event.get(
                "participant_stop_latency_ms"),
            "legacy_reconstruction": False,
        })
    gap_rows = [{**common, **gap} for gap in gaps]
    session_rows = [{
        "session_id": sid,
        "participant_id": session.get("participant_id"),
        "condition": session.get("voice_condition"),
        "analysis_scope": "reconstructed_common_prefix_sensitivity",
        "certified_end_ms": certified_cutoff,
        "excluded_tail_ms": excluded_tail,
        "participant_raw_path": participant_rel,
        "assistant_model_path": assistant_rel,
    }]
    _write_csv(out_dir / "turn_verification_queue.csv", event_rows, EVENT_COLUMNS)
    _write_csv(out_dir / "turn_gap_verification_queue.csv", gap_rows, GAP_COLUMNS)
    _write_csv(out_dir / "turn_session_review_queue.csv", session_rows,
               SESSION_COLUMNS)
    _write_csv(out_dir / "turn_event_manual_additions.csv", [], ADDITION_COLUMNS)
    source_names = [
        "turn_verification_queue.csv", "turn_gap_verification_queue.csv",
        "turn_session_review_queue.csv", "turn_event_manual_additions.csv",
    ]
    manifest = {
        "schema": REVIEW_SCHEMA,
        "status": "awaiting_manual_verification",
        "analysis_scope": "reconstructed_common_prefix_sensitivity",
        "session_id": sid,
        "reconstruction_artifact": artifact_record,
        "certified_end_ms": certified_cutoff,
        "excluded_tail_ms": excluded_tail,
        "candidate_events": len(event_rows),
        "candidate_gaps": len(gap_rows),
        "audio": {
            "participant_browser_clock": participant_record,
            "assistant_browser_clock": assistant_record,
            "participant_alignment": participant_alignment,
            "assistant_alignment": assistant_alignment,
        },
        "input_sha256": {
            name: sha256_file(out_dir / name) for name in source_names
        },
    }
    atomic_write_json(out_dir / "reconstruction_manifest.json", manifest,
                      exclusive=True)
    return {**manifest, "out_dir": str(out_dir)}


def _complete_manifest(dataset: Path) -> dict:
    path = dataset / "turn_verification_manifest.json"
    if not path.is_file():
        raise ReconstructionError(f"manual verification is not finalized: {dataset}")
    result = json.loads(path.read_text())
    if result.get("status") != "complete":
        raise ReconstructionError(f"manual verification is incomplete: {dataset}")
    return result


def combine_verified(primary: Path, reconstructed: Path, out_dir: Path) -> dict:
    """Combine finalized primary and reconstructed tables for sensitivity only."""
    primary = Path(primary).resolve()
    reconstructed = Path(reconstructed).resolve()
    out_dir = Path(out_dir).resolve()
    primary_manifest = _complete_manifest(primary)
    reconstructed_manifest = _complete_manifest(reconstructed)
    reconstruction_record = json.loads(
        (reconstructed / "reconstruction_manifest.json").read_text())
    session_id = str(reconstruction_record.get("session_id") or "")
    if reconstruction_record.get("analysis_scope") != \
            "reconstructed_common_prefix_sensitivity":
        raise ReconstructionError("reconstruction bundle has the wrong analysis scope")

    table_names = [
        "turn_events_adjudicated.csv", "turn_events_verified.csv",
        "turn_gaps_adjudicated.csv", "turn_gaps_verified.csv",
        "turn_session_summary_verified.csv",
    ]
    out_dir.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    counts = {}
    for name in table_names:
        primary_fields, primary_rows = _read_csv(primary / name)
        reconstructed_fields, reconstructed_rows = _read_csv(reconstructed / name)
        fields = list(primary_fields)
        for field in reconstructed_fields:
            if field not in fields:
                fields.append(field)
        if "analysis_scope" not in fields:
            fields.append("analysis_scope")
        for row in primary_rows:
            row["analysis_scope"] = "primary_prespecified"
        for row in reconstructed_rows:
            row["analysis_scope"] = "reconstructed_common_prefix_sensitivity"
        if name == "turn_session_summary_verified.csv":
            primary_ids = {row.get("session_id") for row in primary_rows}
            reconstructed_ids = {row.get("session_id") for row in reconstructed_rows}
            if reconstructed_ids != {session_id}:
                raise ReconstructionError(
                    "reconstructed summary does not contain exactly its declared session")
            if primary_ids & reconstructed_ids:
                raise ReconstructionError("reconstructed session already exists in primary")
        rows = primary_rows + reconstructed_rows
        _write_csv(out_dir / name, rows, fields)
        output_hashes[name] = sha256_file(out_dir / name)
        counts[name] = len(rows)

    manifest = {
        "schema": SENSITIVITY_SCHEMA,
        "status": "complete",
        "analysis_scope": "primary_plus_reconstructed_sensitivity",
        "created_at_unix_s": time.time(),
        "primary_dataset": str(primary),
        "reconstruction_dataset": str(reconstructed),
        "reconstructed_session_id": session_id,
        "primary_verification_manifest_sha256": sha256_file(
            primary / "turn_verification_manifest.json"),
        "reconstruction_verification_manifest_sha256": sha256_file(
            reconstructed / "turn_verification_manifest.json"),
        "primary_sessions": primary_manifest.get("sessions_reviewed"),
        "reconstructed_sessions": reconstructed_manifest.get("sessions_reviewed"),
        "counts": counts,
        "output_sha256": output_hashes,
    }
    atomic_write_json(out_dir / "turn_sensitivity_manifest.json", manifest,
                      exclusive=True)
    return {**manifest, "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m study.turn_reconstruction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    reconstruct_parser = subparsers.add_parser("reconstruct")
    reconstruct_parser.add_argument("--session", required=True)
    reconstruct_parser.add_argument("--data-root", type=Path)
    reconstruct_parser.add_argument("--out", type=Path)
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--primary-dataset", required=True, type=Path)
    combine_parser.add_argument("--reconstruction-dataset", required=True,
                                type=Path)
    combine_parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "reconstruct":
        data_root = (args.data_root or Path(os.path.expanduser(
            os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))).resolve()
        session = get_backend().get_session(args.session)
        if session is None:
            raise SystemExit(f"unknown session: {args.session}")
        result = reconstruct(session, data_root, args.out)
    else:
        result = combine_verified(
            args.primary_dataset, args.reconstruction_dataset, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
