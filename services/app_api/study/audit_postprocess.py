"""Post-process soundboard-audit runs into analysis-ready tables.

Run as:  python -m study.audit_postprocess [--run NAME] [--force]
(cwd = services/app_api, under the app-api venv; STUDY_DATA_ROOT may point at
a copied data root.)

For every run directory under STUDY_DATA_ROOT/audit/ this:
  1. separately verifies the frozen stimulus WAV and exact tagged-audio
     transport from browser send through relay forwarding;
  2. verifies matched pairing (presentations grouped by raw-source hash;
     each item must appear in a natural and a converted manipulation);
  3. transcribes each presentation's PersonaPlex response WAV (Whisper via
     the existing metrics stack) into transcripts/NNN_slot.json;
  4. combines source-sample input annotations with scheduled assistant packet
     timing to derive onset, gap, overlap, pause, yield, and resumption metrics;
  5. writes item/turn and source-route summaries next to results.zip, including
     conversation-clustered uncertainty intervals over technically valid runs.

Everything derives from the frozen manifest + immutable run log; reruns with
--force only ever ADD files, never mutate the originals.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import statistics
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Transcriber = Callable[[str], dict]


def _default_transcriber(path: str) -> dict:
    from metrics import get_transcript_result

    return get_transcript_result(path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _load_run_inputs(run_dir: Path) -> tuple[dict, dict] | None:
    """Manifest + run log, from the unpacked copies or from the zip."""
    manifest = _read_json(run_dir / "manifest.json")
    log = _read_json(run_dir / "run_log.json")
    if manifest is None or log is None:
        zip_path = run_dir / "results.zip"
        if not zip_path.exists():
            return None
        try:
            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
                if manifest is None and "manifest.json" in names:
                    manifest = json.loads(archive.read("manifest.json"))
                if log is None and "run_log.json" in names:
                    log = json.loads(archive.read("run_log.json"))
        except zipfile.BadZipFile:
            return None
    if manifest is None or log is None:
        return None
    return manifest, log


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(mean, 1), round(sd, 1)


# Assistant timing is reconstructed from the browser's scheduled playback
# timeline. Each packet carries its exact start/end on a clock relative to the
# PersonaPlex handshake; t_handshake_ms places it on the run-log clock.
# Legacy event reconstruction remains available for old pilot archives, and is
# labelled in every output row so it cannot be mistaken for packet timing.
DEFAULT_ENERGY_RUN_GAP_MS = 350.0
DEFAULT_ENERGY_THRESHOLD_RMS = 0.02
LEGACY_PACKET_MS = 20.0
ADJACENT_PACKET_TOLERANCE_MS = 1.0
EXPECTED_PP_SAMPLE_RATE_HZ = 24_000
MAX_DURATION_DRIFT_MS = 5.0
DEFAULT_OVERLAP_EVENT_MIN_MS = 20.0
DEFAULT_INTERRUPT_FIRE_DELAY_MS = 800.0
DEFAULT_INTERRUPT_FIRE_TOLERANCE_MS = 300.0


def _merge_intervals(
    intervals: list[tuple[float, float]], max_gap_ms: float
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= max_gap_ms:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _legacy_event_timeline(record: dict, gap_ms: float) -> dict:
    """Best-effort timing for archives created before playback timeline v2."""
    events = record.get("pp_speech_events") or []
    packet_ends = sorted(
        float(event.get("timestampMs") or 0)
        for event in events
        if event.get("type") == "pp_energy"
    )
    if packet_ends:
        intervals = _merge_intervals(
            [(end - LEGACY_PACKET_MS, end) for end in packet_ends],
            ADJACENT_PACKET_TOLERANCE_MS,
        )
        return {
            "speech_intervals": intervals,
            "speech_runs": _merge_intervals(intervals, gap_ms),
            "source": "legacy_energy_events",
            "energetic_packets": len(packet_ends),
            "playback_diagnostics": None,
        }

    intervals: list[tuple[float, float]] = []
    open_start: float | None = None
    fallback_end = float(record.get("t_play_end_ms") or 0)
    for event in events:
        timestamp = float(event.get("timestampMs") or 0)
        if event.get("type") == "pp_speech_start":
            open_start = timestamp
        elif event.get("type") == "pp_speech_end" and open_start is not None:
            intervals.append((open_start, timestamp))
            open_start = None
    if open_start is not None:
        intervals.append((open_start, fallback_end))
    intervals = _merge_intervals(intervals, ADJACENT_PACKET_TOLERANCE_MS)
    return {
        "speech_intervals": intervals,
        "speech_runs": _merge_intervals(intervals, gap_ms),
        "source": "legacy_packet_arrival_events",
        "energetic_packets": None,
        "playback_diagnostics": None,
    }


def _playback_timeline(
    archive: zipfile.ZipFile,
    timeline_name: str | None,
    record: dict,
    timing: dict,
) -> dict:
    gap_ms = float(timing.get("ppSpeakingGapMs") or DEFAULT_ENERGY_RUN_GAP_MS)
    threshold = float(
        timing.get("ppEnergyThresholdRms") or DEFAULT_ENERGY_THRESHOLD_RMS
    )
    handshake_ms = record.get("t_handshake_ms")
    if timeline_name and handshake_ms is not None:
        try:
            timeline = json.loads(archive.read(timeline_name))
        except (KeyError, ValueError):
            timeline = None
        if (
            isinstance(timeline, dict)
            and timeline.get("schema") == "hmo.client-playback-timeline.v2"
            and timeline.get("epoch") == "personaplex_handshake_performance_now"
        ):
            epoch = float(handshake_ms)
            packets = timeline.get("assistant_packets") or []
            intervals = []
            energetic_packets = 0
            for packet in packets:
                try:
                    rms = float(packet["rms"])
                    start = epoch + float(packet["timeline_start_ms"])
                    end = epoch + float(packet["timeline_end_ms"])
                except (KeyError, TypeError, ValueError):
                    continue
                if rms >= threshold and end > start:
                    energetic_packets += 1
                    intervals.append((start, end))
            intervals = _merge_intervals(
                intervals, ADJACENT_PACKET_TOLERANCE_MS
            )
            return {
                "speech_intervals": intervals,
                "speech_runs": _merge_intervals(intervals, gap_ms),
                "source": "playback_timeline_v2",
                "energetic_packets": energetic_packets,
                "playback_diagnostics": {
                    "queue_underrun_count": timeline.get("queue_underrun_count"),
                    "queue_underrun_total_ms": timeline.get(
                        "queue_underrun_total_ms"
                    ),
                    "queue_underrun_max_ms": timeline.get(
                        "queue_underrun_max_ms"
                    ),
                },
            }
    return _legacy_event_timeline(record, gap_ms)


def _response_latency_ms(
    speech_runs: list[tuple[float, float]], clip_end_ms: float | None
) -> float | None:
    if clip_end_ms is None:
        return None
    clip_end = float(clip_end_ms)
    starts = [start for start, _ in speech_runs if start >= clip_end]
    return round(min(starts) - clip_end, 1) if starts else None


def _participant_timeline(record: dict, item: dict) -> dict:
    """Place frozen source-sample annotations on the audit performance clock."""
    play_start = record.get("t_play_start_ms")
    play_end = record.get("t_play_end_ms")
    annotation = item.get("input_timing") or {}
    sample_rate = annotation.get("sample_rate_hz")
    if play_start is None or play_end is None:
        return {
            "speech_intervals": [],
            "pause_intervals": [],
            "speech_start_ms": None,
            "speech_end_ms": None,
            "source": "unavailable",
        }
    if not sample_rate:
        # Manifest v2 did not carry source-sample annotations. Preserve support
        # for old pilots, but label the whole-clip approximation explicitly.
        return {
            "speech_intervals": [(float(play_start), float(play_end))],
            "pause_intervals": [],
            "speech_start_ms": float(play_start),
            "speech_end_ms": float(play_end),
            "source": "legacy_clip_window",
        }

    epoch = float(play_start)
    samples_per_ms = float(sample_rate) / 1000.0

    def place(interval: dict) -> tuple[float, float]:
        return (
            epoch + float(interval["start_sample"]) / samples_per_ms,
            epoch + float(interval["end_sample"]) / samples_per_ms,
        )

    speech_intervals = [
        place(interval) for interval in annotation.get("speech_intervals") or []
    ]
    pause_intervals = [
        place(interval) for interval in annotation.get("pause_intervals") or []
    ]
    return {
        "speech_intervals": speech_intervals,
        "pause_intervals": pause_intervals,
        "speech_start_ms": speech_intervals[0][0] if speech_intervals else None,
        "speech_end_ms": speech_intervals[-1][1] if speech_intervals else None,
        "source": "manifest_input_samples",
    }


def _intersection_ms(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    return sum(
        max(0.0, min(left_end, right_end) - max(left_start, right_start))
        for left_start, left_end in left
        for right_start, right_end in right
    )


def _turn_timing_metrics(
    record: dict,
    item: dict,
    assistant_intervals: list[tuple[float, float]],
    assistant_runs: list[tuple[float, float]],
) -> dict:
    participant = _participant_timeline(record, item)
    participant_intervals = participant["speech_intervals"]
    pause_intervals = participant["pause_intervals"]
    speech_start = participant["speech_start_ms"]
    speech_end = participant["speech_end_ms"]

    overlap = _intersection_ms(participant_intervals, assistant_intervals)
    onsets_during_speech = sum(
        1
        for onset, _ in assistant_runs
        if any(start <= onset < end for start, end in participant_intervals)
    )
    pause_overlap = _intersection_ms(pause_intervals, assistant_intervals)
    onsets_during_pause = sum(
        1
        for onset, _ in assistant_runs
        if any(start <= onset < end for start, end in pause_intervals)
    )

    response_onset = None
    if speech_start is not None:
        candidates = [
            onset
            for onset, offset in assistant_runs
            if offset > float(speech_start)
        ]
        if candidates:
            response_onset = min(candidates)
    onset_offset = None
    if response_onset is not None and speech_end is not None:
        onset_offset = round(response_onset - float(speech_end), 1)
    assistant_speaking_at_input_start = (
        any(
            start <= float(speech_start) < end
            for start, end in assistant_runs
        )
        if speech_start is not None
        else None
    )

    return {
        "participant_timing_source": participant["source"],
        "participant_speech_start_ms": (
            round(float(speech_start), 1) if speech_start is not None else None
        ),
        "participant_speech_end_ms": (
            round(float(speech_end), 1) if speech_end is not None else None
        ),
        "response_onset_offset_ms": onset_offset,
        "assistant_speaking_at_input_start": (
            assistant_speaking_at_input_start
        ),
        "positive_response_gap_ms": (
            onset_offset if onset_offset is not None and onset_offset >= 0 else None
        ),
        "premature_assistant_onset": (
            onset_offset is not None and onset_offset < 0
        ),
        "overlap_ms": round(overlap, 1),
        "pp_onsets_during_participant_speech": onsets_during_speech,
        "prespecified_pause_count": len(pause_intervals),
        "assistant_speech_during_prespecified_pause_ms": round(
            pause_overlap, 1
        ),
        "assistant_onsets_during_prespecified_pause": onsets_during_pause,
    }


def _interruption_metrics(
    record: dict,
    item: dict,
    assistant_intervals: list[tuple[float, float]],
    assistant_runs: list[tuple[float, float]],
) -> dict:
    empty = {
        "assistant_speaking_at_interrupt": None,
        "fire_offset_into_assistant_turn_ms": None,
        "assistant_stop_latency_ms": None,
        "assistant_yielded_before_input_offset": None,
        "post_interruption_overlap_ms": None,
        "response_resumed": None,
        "response_resumption_latency_ms": None,
    }
    if item.get("presentation_mode") != "during_pp_speech":
        return empty

    participant = _participant_timeline(record, item)
    input_start = participant["speech_start_ms"]
    input_end = participant["speech_end_ms"]
    if input_start is None or input_end is None:
        return {**empty, "assistant_speaking_at_interrupt": False}

    ongoing = next(
        (
            (start, end)
            for start, end in assistant_runs
            if start <= float(input_start) < end
        ),
        None,
    )
    participant_intervals = participant["speech_intervals"]
    overlap = _intersection_ms(participant_intervals, assistant_intervals)
    if ongoing is None:
        return {
            **empty,
            "assistant_speaking_at_interrupt": False,
            "post_interruption_overlap_ms": round(overlap, 1),
        }

    ongoing_start, ongoing_end = ongoing
    stop_latency = round(max(0.0, ongoing_end - float(input_start)), 1)
    resumed_runs = [
        (start, end)
        for start, end in assistant_runs
        if start > ongoing_end and start >= float(input_end)
    ]
    resumed_at = min((start for start, _ in resumed_runs), default=None)
    return {
        "assistant_speaking_at_interrupt": True,
        "fire_offset_into_assistant_turn_ms": round(
            float(input_start) - ongoing_start, 1
        ),
        "assistant_stop_latency_ms": stop_latency,
        "assistant_yielded_before_input_offset": ongoing_end <= float(input_end),
        "post_interruption_overlap_ms": round(overlap, 1),
        "response_resumed": resumed_at is not None,
        "response_resumption_latency_ms": (
            round(resumed_at - float(input_end), 1)
            if resumed_at is not None
            else None
        ),
    }


def _timing_source_summary(sources: list[str]) -> str:
    unique = sorted(set(sources))
    if not unique:
        return "unavailable"
    return unique[0] if len(unique) == 1 else "mixed"


def _delivery_metrics(record: dict, frozen_clip_sha256: str | None) -> dict:
    sent_clip_sha256 = record.get("sent_clip_sha256")
    stimulus_integrity = (
        sent_clip_sha256 is not None
        and frozen_clip_sha256 is not None
        and sent_clip_sha256 == frozen_clip_sha256
    )
    audit = record.get("delivery_audit") or {}
    browser = audit.get("browser_sent")
    received = audit.get("relay_received")
    forwarded = audit.get("relay_forwarded")
    if not all(isinstance(value, dict) for value in (browser, received, forwarded)):
        return {
            "stimulus_integrity_verified": stimulus_integrity,
            "delivery_verified": None,
            "delivery_status": "unavailable",
            "delivery_audit": audit or None,
        }

    def signature(value: dict) -> tuple[int, int, str]:
        return (
            int(value.get("frames") or 0),
            int(value.get("bytes") or 0),
            str(value.get("sha256") or ""),
        )

    browser_signature = signature(browser)
    received_signature = signature(received)
    forwarded_signature = signature(forwarded)
    nonempty = browser_signature[0] > 0 and browser_signature[1] > 0
    verified = (
        audit.get("status") == "complete"
        and nonempty
        and browser_signature == received_signature == forwarded_signature
    )
    return {
        "stimulus_integrity_verified": stimulus_integrity,
        "delivery_verified": verified,
        "delivery_status": (
            "verified" if verified else "empty" if not nonempty else "mismatch"
        ),
        "delivery_audit": audit,
    }


def _technical_validity(row: dict) -> dict:
    reasons = []
    warnings = []
    if row.get("status") not in ("ok", "no_response"):
        reasons.append(f"run_status:{row.get('status') or 'missing'}")
    if row.get("timing_source") != "playback_timeline_v2":
        reasons.append("playback_timeline_unavailable")
    if row.get("participant_timing_source") != "manifest_input_samples":
        reasons.append("input_sample_annotations_unavailable")
    if row.get("participant_speech_start_ms") is None:
        reasons.append("participant_speech_missing")
    if not row.get("source_speaker"):
        reasons.append("source_speaker_missing")
    if not row.get("route"):
        reasons.append("route_missing")
    if not row.get("raw_sha256"):
        reasons.append("raw_source_artifact_missing")
    if row.get("input_timing_source_sha256") != row.get("raw_sha256"):
        reasons.append("input_timing_source_mismatch")
    if row.get("route") == "converted" and not row.get("target_sha256"):
        reasons.append("converted_target_artifact_missing")
    clip_audio = row.get("clip_audio") or {}
    if clip_audio.get("sample_rate_hz") != EXPECTED_PP_SAMPLE_RATE_HZ:
        reasons.append("clip_sample_rate_invalid")
    if (
        clip_audio.get("normalized") is not True
        or clip_audio.get("measured_lufs") is None
    ):
        reasons.append("clip_loudness_unverified")
    try:
        excessive_drift = abs(float(clip_audio["duration_drift_ms"])) > (
            MAX_DURATION_DRIFT_MS
        )
    except (KeyError, TypeError, ValueError):
        reasons.append("clip_duration_drift_unavailable")
    else:
        if excessive_drift:
            reasons.append("clip_duration_drift_excessive")
    if row.get("stimulus_integrity_verified") is not True:
        reasons.append("stimulus_integrity_unverified")
    if row.get("delivery_verified") is not True:
        reasons.append("transport_delivery_unverified")
    if row.get("personaplex_wav_present") is not True:
        reasons.append("personaplex_wav_missing")
    if row.get("sent_wav_present") is not True:
        reasons.append("sent_wav_missing")
    playback = row.get("playback_diagnostics") or {}
    if playback.get("queue_underrun_count") is None:
        reasons.append("assistant_playback_underrun_diagnostics_missing")
    elif int(playback["queue_underrun_count"]) > 0:
        warnings.append("assistant_playback_underrun")
    if (
        row.get("presentation_mode") == "during_pp_speech"
        and row.get("assistant_speaking_at_interrupt") is not True
    ):
        reasons.append("assistant_not_speaking_at_interrupt")
    if row.get("presentation_mode") == "during_pp_speech":
        fire_error = row.get("interruption_fire_error_ms")
        tolerance = row.get("interruption_fire_tolerance_ms")
        if fire_error is None or tolerance is None:
            reasons.append("interruption_fire_offset_unavailable")
        elif abs(float(fire_error)) > float(tolerance):
            reasons.append("interruption_fire_offset_out_of_tolerance")
    if (
        row.get("presentation_mode") == "after_silence"
        and row.get("assistant_speaking_at_input_start") is True
    ):
        reasons.append("assistant_not_silent_at_input_start")
    return {
        "technical_evaluable": not reasons,
        "technical_invalid_reasons": reasons,
        "technical_warnings": warnings,
    }


def _mark_technical_invalid(row: dict, reason: str) -> None:
    reasons = row.setdefault("technical_invalid_reasons", [])
    if reason not in reasons:
        reasons.append(reason)
    row["technical_evaluable"] = False


def _apply_run_level_validity(rows: list[dict]) -> None:
    target_levels = {
        float(audio["target_lufs"])
        for row in rows
        if isinstance((audio := row.get("clip_audio")), dict)
        and audio.get("target_lufs") is not None
    }
    if len(target_levels) > 1:
        for row in rows:
            _mark_technical_invalid(row, "loudness_target_mismatch")


def _invalid_reason_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(
        reason
        for row in rows
        for reason in row.get("technical_invalid_reasons") or []
    ))


def _warning_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(
        warning
        for row in rows
        for warning in row.get("technical_warnings") or []
    ))


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _cluster_bootstrap_mean(
    rows: list[dict],
    cluster_key: str,
    value,
    seed: int,
    repetitions: int = 2_000,
) -> dict:
    clusters: dict[str, list[dict]] = {}
    for row in rows:
        clusters.setdefault(str(row.get(cluster_key)), []).append(row)
    cluster_ids = sorted(clusters)
    observed = [
        float(result)
        for row in rows
        if (result := value(row)) is not None
    ]
    if not observed or not cluster_ids:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    if len(cluster_ids) == 1:
        mean = statistics.fmean(observed)
        return {
            "mean": round(mean, 3),
            "ci95_low": round(mean, 3),
            "ci95_high": round(mean, 3),
            "n": len(observed),
        }

    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sampled_rows = [
            row
            for _ in cluster_ids
            for row in clusters[rng.choice(cluster_ids)]
        ]
        sampled_values = [
            float(result)
            for row in sampled_rows
            if (result := value(row)) is not None
        ]
        if sampled_values:
            estimates.append(statistics.fmean(sampled_values))
    return {
        "mean": round(statistics.fmean(observed), 3),
        "ci95_low": round(float(_percentile(estimates, 0.025)), 3),
        "ci95_high": round(float(_percentile(estimates, 0.975)), 3),
        "n": len(observed),
    }


def _clustered_cell_summary(
    rows: list[dict], cluster_key: str, seed: int
) -> dict:
    evaluable = [row for row in rows if row.get("technical_evaluable")]
    valid_interruptions = [
        row for row in evaluable
        if row.get("assistant_speaking_at_interrupt") is True
    ]

    def metric(name: str, source: list[dict], getter, offset: int) -> dict:
        result = _cluster_bootstrap_mean(
            source, cluster_key, getter, seed + offset
        )
        return {
            f"{name}_mean": result["mean"],
            f"{name}_ci95_low": result["ci95_low"],
            f"{name}_ci95_high": result["ci95_high"],
            f"{name}_n": result["n"],
        }

    clusters: dict[str, list[dict]] = {}
    for row in rows:
        clusters.setdefault(str(row.get(cluster_key)), []).append(row)
    contributing_clusters = {
        str(row.get(cluster_key)) for row in evaluable
    }
    fully_evaluable_clusters = {
        cluster_id
        for cluster_id, cluster_rows in clusters.items()
        if cluster_rows and all(
            row.get("technical_evaluable") for row in cluster_rows
        )
    }
    result = {
        "n_attempted": len(rows),
        "n_evaluable": len(evaluable),
        "n_technical_invalid": len(rows) - len(evaluable),
        "n_attempted_conversation_clusters": len(clusters),
        "n_contributing_conversation_clusters": len(contributing_clusters),
        "n_fully_evaluable_conversation_clusters": len(
            fully_evaluable_clusters
        ),
        "n_replacement_conversations_needed": (
            len(clusters) - len(fully_evaluable_clusters)
        ),
    }
    result.update(metric(
        "complete_audio_delivery_rate", rows,
        lambda row: float(row.get("delivery_verified") is True), 8,
    ))
    result.update(metric(
        "stimulus_integrity_rate", rows,
        lambda row: float(row.get("stimulus_integrity_verified") is True), 9,
    ))
    result.update(metric(
        "technical_evaluable_rate", rows,
        lambda row: float(row.get("technical_evaluable") is True), 10,
    ))
    result.update(metric(
        "response_onset_offset_ms", evaluable,
        lambda row: row.get("response_onset_offset_ms"), 11,
    ))
    result.update(metric(
        "positive_response_gap_ms", evaluable,
        lambda row: row.get("positive_response_gap_ms"), 1,
    ))
    result.update(metric(
        "overlap_ms", evaluable, lambda row: row.get("overlap_ms"), 2,
    ))
    result.update(metric(
        "overlap_occurrence_rate", evaluable,
        lambda row: float(bool(row.get("overlap_event"))), 12,
    ))
    result.update(metric(
        "premature_onset_rate", evaluable,
        lambda row: float(bool(row.get("premature_assistant_onset"))), 3,
    ))
    result.update(metric(
        "no_response_rate", evaluable,
        lambda row: float(row.get("status") == "no_response"), 4,
    ))
    result.update(metric(
        "fire_offset_into_assistant_turn_ms", valid_interruptions,
        lambda row: row.get("fire_offset_into_assistant_turn_ms"), 15,
    ))
    result.update(metric(
        "assistant_stop_latency_ms", valid_interruptions,
        lambda row: row.get("assistant_stop_latency_ms"), 5,
    ))
    result.update(metric(
        "yield_before_input_offset_rate", valid_interruptions,
        lambda row: float(bool(row.get("assistant_yielded_before_input_offset"))),
        6,
    ))
    result.update(metric(
        "post_interruption_overlap_ms", valid_interruptions,
        lambda row: row.get("post_interruption_overlap_ms"), 13,
    ))
    result.update(metric(
        "response_resumption_rate", valid_interruptions,
        lambda row: float(bool(row.get("response_resumed"))), 7,
    ))
    result.update(metric(
        "response_resumption_latency_ms", valid_interruptions,
        lambda row: row.get("response_resumption_latency_ms"), 14,
    ))
    return result


def _cell_summaries(
    rows: list[dict], cluster_key: str, seed: int
) -> list[dict]:
    """Summarize source-route cells while resampling whole conversations."""
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (
            str(row.get("source_speaker") or ""),
            str(row.get("route") or ""),
            str(row.get("target_id") or ""),
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for offset, (key, items) in enumerate(sorted(grouped.items())):
        source_speaker, route, target_id = key
        cell = {
            "source_speaker": source_speaker or None,
            "route": route or None,
            "target_id": target_id or None,
            "target_label": next(
                (
                    row.get("target_label")
                    for row in items
                    if row.get("target_label")
                ),
                None,
            ),
            "engine": next(
                (row.get("engine") for row in items if row.get("engine")),
                None,
            ),
            "target_sha256": next(
                (
                    row.get("target_sha256")
                    for row in items
                    if row.get("target_sha256")
                ),
                None,
            ),
            "technical_invalid_reason_counts": _invalid_reason_counts(items),
            "technical_warning_counts": _warning_counts(items),
        }
        cell.update(
            _clustered_cell_summary(
                items,
                cluster_key,
                seed + offset * 100,
            )
        )
        summaries.append(cell)
    return summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    serializable = [
        {
            key: (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(serializable[0].keys()))
    writer.writeheader()
    writer.writerows(serializable)
    path.write_text(buffer.getvalue())


def _process_script_run(run_dir: Path, manifest: dict, log: dict,
                        transcribe: Transcriber, force: bool) -> dict:
    # Scripts keyed by condition: interleaved manifests carry one per
    # condition tag; single-script manifests use the None key.
    scripts_by_condition: dict[str | None, dict[int, dict]] = {}
    if manifest.get("interleaved") and manifest.get("scripts"):
        for entry in manifest["scripts"]:
            scripts_by_condition[entry.get("condition")] = {
                int(t["turn"]): t for t in entry.get("turns") or []}
    else:
        scripts_by_condition[None] = {
            int(t["turn"]): t for t in manifest.get("script") or []}
    timing = manifest.get("timing") or {}
    gap_ms = float(timing.get("ppSpeakingGapMs") or DEFAULT_ENERGY_RUN_GAP_MS)
    sessions = list(log.get("records") or [])
    transcripts_dir = run_dir / "transcripts"
    zip_path = run_dir / "results.zip"
    turn_rows: list[dict] = []
    session_rows: list[dict] = []
    timing_sources: list[str] = []
    with (zipfile.ZipFile(zip_path) if zip_path.exists()
          else zipfile.ZipFile(io.BytesIO(), "w")) as archive:
        names = set(archive.namelist())
        for session in sessions:
            rep = int(session.get("rep") or 0)
            condition = session.get("condition")
            script = (scripts_by_condition.get(condition)
                      or scripts_by_condition.get(None) or {})
            # Interleaved runs suffix the folder with the condition slug;
            # match by prefix so both layouts resolve.
            rep_prefix = f"runs/rep_{rep:03d}"
            wav_name = next(
                (n for n in sorted(names)
                 if n.startswith(rep_prefix) and n.endswith("personaplex.wav")),
                f"{rep_prefix}/personaplex.wav")
            sent_name = next(
                (n for n in sorted(names)
                 if n.startswith(rep_prefix) and n.endswith("sent.wav")),
                f"{rep_prefix}/sent.wav",
            )
            timeline_name = next(
                (n for n in sorted(names)
                 if n.startswith(rep_prefix)
                 and n.endswith("playback_timeline.json")),
                None,
            )
            speech = _playback_timeline(
                archive, timeline_name, session, timing
            )
            speech_intervals = speech["speech_intervals"]
            speech_runs = speech["speech_runs"]
            timing_sources.append(speech["source"])
            transcript_path = transcripts_dir / f"rep_{rep:03d}.json"
            transcript_text = None
            if transcript_path.exists() and not force:
                transcript_text = (_read_json(transcript_path) or {}).get("text")
            elif wav_name in names:
                transcripts_dir.mkdir(parents=True, exist_ok=True)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
                    handle.write(archive.read(wav_name))
                    handle.flush()
                    try:
                        asr = transcribe(handle.name)
                        transcript_text = str(asr.get("text") or "").strip()
                        transcript_path.write_text(json.dumps(
                            {"text": transcript_text, "status": asr.get("status"),
                             "source": wav_name}, indent=2))
                    except Exception as exc:  # noqa: BLE001
                        transcript_path.write_text(json.dumps(
                            {"text": None, "status": "failed", "error": str(exc),
                             "source": wav_name}, indent=2))
            session_rows.append({
                "rep": rep, "condition": condition,
                "status": session.get("status"),
                "greeted": session.get("greeted"),
                "turns_ok": sum(1 for t in session.get("turns") or []
                                if t.get("status") == "ok"),
                "turns_total": len(session.get("turns") or []),
                "pp_transcript_rows": len(session.get("pp_transcript") or []),
                "pp_full_transcript": transcript_text,
                "personaplex_wav_present": wav_name in names,
                "sent_wav_present": sent_name in names,
                "playback_timeline_present": timeline_name in names
                if timeline_name else False,
            })
            for turn in session.get("turns") or []:
                index = int(turn.get("turn") or 0)
                turn_spec = script.get(index) or {}
                frozen = turn_spec.get("clip_sha256")
                end = turn.get("t_play_end_ms")
                delivery_metrics = _delivery_metrics(turn, frozen)
                post_clip_latency = _response_latency_ms(speech_runs, end)
                if speech["source"] != "playback_timeline_v2":
                    post_clip_latency = turn.get("response_latency_ms")
                timing_metrics = _turn_timing_metrics(
                    turn, turn_spec, speech_intervals, speech_runs
                )
                overlap_minimum_ms = float(
                    timing.get("overlapEventMinimumMs")
                    or DEFAULT_OVERLAP_EVENT_MIN_MS
                )
                response_latency = timing_metrics["positive_response_gap_ms"]
                if timing_metrics["participant_timing_source"] == "legacy_clip_window":
                    response_latency = post_clip_latency
                row = {
                    "rep": rep, "condition": condition, "turn": index,
                    "replay_condition": condition,
                    "item_condition": turn_spec.get("condition"),
                    "label": turn.get("label"),
                    "manipulation": turn_spec.get("manipulation"),
                    "route": turn_spec.get("route"),
                    "source_speaker": turn_spec.get("source_speaker"),
                    "engine": turn_spec.get("engine"),
                    "target_id": turn_spec.get("target_id"),
                    "target_label": turn_spec.get("target_label"),
                    "target_sha256": turn_spec.get("target_sha256"),
                    "raw_sha256": turn_spec.get("raw_sha256"),
                    "input_timing_source_sha256": turn_spec.get(
                        "input_timing_source_sha256"
                    ),
                    "clip_audio": turn_spec.get("clip_audio"),
                    "status": turn.get("status"),
                    "presentation_mode": "after_silence",
                    **delivery_metrics,
                    "response_latency_ms": response_latency,
                    "post_clip_response_latency_ms": post_clip_latency,
                    "runner_response_latency_ms": turn.get("response_latency_ms"),
                    "timing_source": speech["source"],
                    "energetic_packets": speech["energetic_packets"],
                    "playback_diagnostics": speech["playback_diagnostics"],
                    "personaplex_wav_present": wav_name in names,
                    "sent_wav_present": sent_name in names,
                    "pp_spoke_during_clip": turn.get("pp_spoke_during_clip"),
                    "overlap_event": (
                        timing_metrics["overlap_ms"] >= overlap_minimum_ms
                    ),
                    "pp_onsets_during_clip": timing_metrics[
                        "pp_onsets_during_participant_speech"
                    ],
                    **timing_metrics,
                }
                row.update(_technical_validity(row))
                turn_rows.append(row)

    _apply_run_level_validity(turn_rows)

    # Per-turn summary across replays, split by condition when interleaved
    # (condition is None for single-script runs and sorts first).
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in turn_rows:
        grouped.setdefault((str(row.get("condition") or ""), int(row["turn"])),
                           []).append(row)
    summary_rows = []
    for (condition_key, index) in sorted(grouped):
        items = grouped[(condition_key, index)]
        condition = items[0].get("condition")
        turn_spec = ((scripts_by_condition.get(condition)
                      or scripts_by_condition.get(None) or {}).get(index) or {})
        all_latencies = [r["response_latency_ms"] for r in items
                         if r.get("response_latency_ms") is not None]
        evaluable_latencies = [r["response_latency_ms"] for r in items
                               if r.get("technical_evaluable")
                               and r.get("response_latency_ms") is not None]
        mean, sd = _mean_sd(all_latencies)
        evaluable_mean, evaluable_sd = _mean_sd(evaluable_latencies)
        n_evaluable = sum(1 for row in items if row["technical_evaluable"])
        target_n = int(manifest.get("reps") or len(items))
        summary_rows.append({
            "condition": condition,
            "turn": index,
            "label": turn_spec.get("label"),
            "manipulation": turn_spec.get("manipulation"),
            "n": len(items),
            "n_target_evaluable": target_n,
            "n_evaluable": n_evaluable,
            "n_technical_invalid": len(items) - n_evaluable,
            "n_replacements_needed": max(0, target_n - n_evaluable),
            "n_ok": sum(1 for r in items if r["status"] == "ok"),
            "n_no_response": sum(1 for r in items if r["status"] == "no_response"),
            "n_evaluable_no_response": sum(
                1 for r in items
                if r["technical_evaluable"] and r["status"] == "no_response"
            ),
            "n_delivery_verified": sum(1 for r in items if r["delivery_verified"]),
            "n_stimulus_integrity_verified": sum(
                1 for r in items if r["stimulus_integrity_verified"]
            ),
            "n_runner_assistant_overlap_candidate": sum(
                1 for r in items if r.get("pp_spoke_during_clip")
            ),
            "n_premature_assistant_onset": sum(
                1 for r in items if r.get("premature_assistant_onset")
            ),
            "n_overlap_events": sum(
                1 for r in items if r.get("overlap_event")
            ),
            "response_latency_mean_ms": mean,
            "response_latency_sd_ms": sd,
            "response_latency_evaluable_mean_ms": evaluable_mean,
            "response_latency_evaluable_sd_ms": evaluable_sd,
            "overlap_ms_total": round(sum(r["overlap_ms"] or 0 for r in items), 1),
            "assistant_pause_overlap_ms_total": round(sum(
                r.get("assistant_speech_during_prespecified_pause_ms") or 0
                for r in items
            ), 1),
            "assistant_onsets_during_prespecified_pauses": sum(
                r.get("assistant_onsets_during_prespecified_pause") or 0
                for r in items
            ),
            "technical_invalid_reason_counts": _invalid_reason_counts(items),
            "technical_warning_counts": _warning_counts(items),
        })

    bootstrap_seed = int(manifest.get("seed") or 0)
    cell_summary = _cell_summaries(turn_rows, "rep", bootstrap_seed)
    summary = {
        "schema": "hmo.soundboard-audit-summary.v2",
        "mode": "script",
        "interleaved": bool(manifest.get("interleaved")),
        "run": run_dir.name,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_schema": manifest.get("schema"),
        "assistant": manifest.get("assistant"),
        "audio_format": manifest.get("audio_format"),
        "generated_at_unix_s": time.time(),
        # Self-describing measurement definitions (from the frozen manifest
        # when present; defaults otherwise).
        "detection": {
            "energy_run_gap_ms": gap_ms,
            "pp_energy_threshold_rms": timing.get("ppEnergyThresholdRms"),
            "overlap_event_minimum_ms": float(
                timing.get("overlapEventMinimumMs")
                or DEFAULT_OVERLAP_EVENT_MIN_MS
            ),
            "timing_source": _timing_source_summary(timing_sources),
            "timing_source_counts": dict(Counter(timing_sources)),
            "legacy_packet_ms": (
                LEGACY_PACKET_MS
                if any(source == "legacy_energy_events"
                       for source in timing_sources)
                else None
            ),
            "threshold_source": (
                "manifest.timing"
                if timing.get("ppEnergyThresholdRms") is not None
                else "postprocess_default"
            ),
            "run_gap_source": (
                "manifest.timing"
                if timing.get("ppSpeakingGapMs") is not None
                else "postprocess_default"
            ),
        },
        "sessions": session_rows,
        "turns": turn_rows,
        "turn_summary": summary_rows,
        "cell_summary": cell_summary,
        "clustered_uncertainty": {
            "method": "percentile_cluster_bootstrap",
            "confidence": 0.95,
            "repetitions": 2_000,
            "cluster": "fresh_personaplex_conversation",
            "cluster_key": "rep",
            "seed": bootstrap_seed,
        },
        "counts": {
            "replays": len(session_rows),
            "replays_ok": sum(1 for s in session_rows if s["status"] == "ok"),
            "turns": len(turn_rows),
            "turns_ok": sum(1 for r in turn_rows if r["status"] == "ok"),
            "turns_evaluable": sum(
                1 for r in turn_rows if r["technical_evaluable"]
            ),
            "turns_technical_invalid": sum(
                1 for r in turn_rows if not r["technical_evaluable"]
            ),
            "delivery_verified": sum(1 for r in turn_rows if r["delivery_verified"]),
            "stimulus_integrity_verified": sum(
                1 for r in turn_rows if r["stimulus_integrity_verified"]
            ),
            "technical_invalid_reason_counts": _invalid_reason_counts(turn_rows),
            "technical_warning_counts": _warning_counts(turn_rows),
        },
    }
    (run_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    _write_csv(run_dir / "audit_summary.csv", summary_rows)
    _write_csv(run_dir / "audit_cell_summary.csv", cell_summary)
    _write_csv(run_dir / "audit_turns.csv", turn_rows)
    _write_csv(run_dir / "audit_sessions.csv", session_rows)
    return summary


def process_run(run_dir: Path, transcriber: Transcriber | None = None,
                force: bool = False) -> dict | None:
    summary_path = run_dir / "audit_summary.json"
    if summary_path.exists() and not force:
        return None
    loaded = _load_run_inputs(run_dir)
    if loaded is None:
        return {"run": run_dir.name, "error": "manifest or run log unavailable"}
    manifest, log = loaded
    if manifest.get("mode") == "script":
        return _process_script_run(run_dir, manifest, log,
                                   transcriber or _default_transcriber, force)
    presentations = {int(p["index"]): p for p in manifest.get("presentations") or []}
    records = list(log.get("records") or [])
    transcribe = transcriber or _default_transcriber

    # ---- per-presentation table -------------------------------------------
    zip_path = run_dir / "results.zip"
    transcripts_dir = run_dir / "transcripts"
    timing = manifest.get("timing") or {}
    rows: list[dict] = []
    timing_sources: list[str] = []
    archive_cm = (zipfile.ZipFile(zip_path) if zip_path.exists()
                  else zipfile.ZipFile(io.BytesIO(), "w"))
    with archive_cm as archive:
        names = set(archive.namelist())
        for record in records:
            presentation = presentations.get(int(record.get("index") or 0), {})
            prefix = (f"runs/{int(record['index']):03d}_"
                      f"{record.get('slot_id')}")
            timeline_name = f"{prefix}/playback_timeline.json"
            speech = _playback_timeline(
                archive,
                timeline_name if timeline_name in names else None,
                record,
                timing,
            )
            timing_sources.append(speech["source"])
            frozen = presentation.get("clip_sha256")
            delivery_metrics = _delivery_metrics(record, frozen)
            # 3. PP response transcript.
            transcript_text = None
            wav_name = f"{prefix}/personaplex.wav"
            sent_name = f"{prefix}/sent.wav"
            transcript_path = transcripts_dir / f"{prefix.split('/', 1)[1]}.json"
            if transcript_path.exists() and not force:
                cached = _read_json(transcript_path)
                transcript_text = (cached or {}).get("text")
            elif wav_name in names:
                transcripts_dir.mkdir(parents=True, exist_ok=True)
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
                    handle.write(archive.read(wav_name))
                    handle.flush()
                    try:
                        asr = transcribe(handle.name)
                        transcript_text = str(asr.get("text") or "").strip()
                        transcript_path.write_text(json.dumps({
                            "text": transcript_text,
                            "status": asr.get("status"),
                            "source": wav_name,
                        }, indent=2))
                    except Exception as exc:  # noqa: BLE001 - keep processing
                        transcript_text = None
                        transcript_path.write_text(json.dumps({
                            "text": None, "status": "failed",
                            "error": str(exc), "source": wav_name,
                        }, indent=2))
            post_clip_latency = _response_latency_ms(
                speech["speech_runs"], record.get("t_play_end_ms")
            )
            if speech["source"] != "playback_timeline_v2":
                post_clip_latency = record.get("response_latency_ms")
            timing_metrics = _turn_timing_metrics(
                record,
                presentation,
                speech["speech_intervals"],
                speech["speech_runs"],
            )
            overlap_minimum_ms = float(
                timing.get("overlapEventMinimumMs")
                or DEFAULT_OVERLAP_EVENT_MIN_MS
            )
            interruption_metrics = _interruption_metrics(
                record,
                presentation,
                speech["speech_intervals"],
                speech["speech_runs"],
            )
            fire_target_ms = float(
                timing.get("interruptFireDelayMs")
                or DEFAULT_INTERRUPT_FIRE_DELAY_MS
            )
            fire_tolerance_ms = float(
                timing.get("interruptFireToleranceMs")
                or DEFAULT_INTERRUPT_FIRE_TOLERANCE_MS
            )
            fire_offset_ms = interruption_metrics[
                "fire_offset_into_assistant_turn_ms"
            ]
            response_latency = timing_metrics["positive_response_gap_ms"]
            if timing_metrics["participant_timing_source"] == "legacy_clip_window":
                response_latency = post_clip_latency
            yield_latency = interruption_metrics["assistant_stop_latency_ms"]
            if speech["source"] != "playback_timeline_v2":
                yield_latency = record.get("pp_yield_latency_ms")
            row = {
                "index": record.get("index"),
                "label": record.get("label"),
                "manipulation": presentation.get("manipulation"),
                "route": presentation.get("route"),
                "source_speaker": presentation.get("source_speaker"),
                "engine": presentation.get("engine"),
                "target_id": presentation.get("target_id"),
                "target_label": presentation.get("target_label"),
                "target_sha256": presentation.get("target_sha256"),
                "condition": presentation.get("condition"),
                "clip_audio": presentation.get("clip_audio"),
                "raw_sha256": presentation.get("raw_sha256"),
                "input_timing_source_sha256": presentation.get(
                    "input_timing_source_sha256"
                ),
                "presentation_mode": presentation.get("presentation_mode"),
                "status": record.get("status"),
                **delivery_metrics,
                "response_latency_ms": response_latency,
                "post_clip_response_latency_ms": post_clip_latency,
                "runner_response_latency_ms": record.get("response_latency_ms"),
                "pp_yield_latency_ms": yield_latency,
                "runner_pp_yield_latency_ms": record.get("pp_yield_latency_ms"),
                "timing_source": speech["source"],
                "energetic_packets": speech["energetic_packets"],
                "playback_diagnostics": speech["playback_diagnostics"],
                "personaplex_wav_present": wav_name in names,
                "sent_wav_present": sent_name in names,
                "pp_onsets_during_clip": timing_metrics[
                    "pp_onsets_during_participant_speech"
                ],
                "overlap_event": (
                    timing_metrics["overlap_ms"] >= overlap_minimum_ms
                ),
                **timing_metrics,
                **interruption_metrics,
                "interruption_fire_target_ms": fire_target_ms,
                "interruption_fire_tolerance_ms": fire_tolerance_ms,
                "interruption_fire_error_ms": (
                    round(float(fire_offset_ms) - fire_target_ms, 1)
                    if fire_offset_ms is not None
                    else None
                ),
                "pp_response_transcript": transcript_text,
            }
            row.update(_technical_validity(row))
            rows.append(row)

    # ---- 2. matched-pair verification --------------------------------------
    by_raw: dict[str, dict[str, set[str]]] = {}
    for presentation in presentations.values():
        raw = presentation.get("raw_sha256")
        if raw:
            entry = by_raw.setdefault(raw, {
                "manipulations": set(),
                "source_speakers": set(),
            })
            entry["manipulations"].add(str(presentation.get("manipulation")))
            source = presentation.get("source_speaker")
            if source:
                entry["source_speakers"].add(str(source))
    pairing_failures = []
    require_source_metadata = manifest.get("schema") == (
        "hmo.soundboard-audit-manifest.v3"
    )
    for raw, entry in sorted(by_raw.items()):
        reasons = []
        manipulations = entry["manipulations"]
        source_speakers = entry["source_speakers"]
        if not ({"unconverted", "vc"} <= manipulations):
            reasons.append("natural_converted_pair_missing")
        if require_source_metadata and len(source_speakers) != 1:
            reasons.append("source_speaker_mismatch")
        if reasons:
            pairing_failures.append({
                "raw_sha256": raw,
                "manipulations": sorted(manipulations),
                "source_speakers": sorted(source_speakers),
                "reasons": reasons,
            })
    failed_raw_hashes = {
        failure["raw_sha256"] for failure in pairing_failures
    }
    for row in rows:
        if row.get("raw_sha256") in failed_raw_hashes:
            _mark_technical_invalid(row, "matched_pair_invalid")
    _apply_run_level_validity(rows)

    # ---- item x condition summary -------------------------------------------
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("raw_sha256") or row["label"]),
               str(row.get("manipulation")))
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for (item, manipulation), items in sorted(grouped.items()):
        all_latencies = [r["response_latency_ms"] for r in items
                         if r.get("response_latency_ms") is not None]
        all_yields = [r["pp_yield_latency_ms"] for r in items
                      if r.get("pp_yield_latency_ms") is not None]
        all_resumptions = [r["response_resumption_latency_ms"] for r in items
                           if r.get("response_resumption_latency_ms") is not None]
        evaluable_latencies = [r["response_latency_ms"] for r in items
                               if r.get("technical_evaluable")
                               and r.get("response_latency_ms") is not None]
        evaluable_yields = [r["pp_yield_latency_ms"] for r in items
                            if r.get("technical_evaluable")
                            and r.get("pp_yield_latency_ms") is not None]
        evaluable_resumptions = [
            r["response_resumption_latency_ms"] for r in items
            if r.get("technical_evaluable")
            and r.get("response_resumption_latency_ms") is not None
        ]
        latency_mean, latency_sd = _mean_sd(all_latencies)
        yield_mean, yield_sd = _mean_sd(all_yields)
        resumption_mean, resumption_sd = _mean_sd(all_resumptions)
        evaluable_latency_mean, evaluable_latency_sd = _mean_sd(
            evaluable_latencies
        )
        evaluable_yield_mean, evaluable_yield_sd = _mean_sd(evaluable_yields)
        evaluable_resumption_mean, evaluable_resumption_sd = _mean_sd(
            evaluable_resumptions
        )
        n_evaluable = sum(1 for row in items if row["technical_evaluable"])
        target_n = int(manifest.get("reps") or len(items))
        summary_rows.append({
            "item_raw_sha256": item[:12],
            "labels": "; ".join(sorted({r["label"] for r in items})),
            "manipulation": manipulation,
            "n": len(items),
            "n_target_evaluable": target_n,
            "n_evaluable": n_evaluable,
            "n_technical_invalid": len(items) - n_evaluable,
            "n_replacements_needed": max(0, target_n - n_evaluable),
            "n_ok": sum(1 for r in items if r["status"] == "ok"),
            "n_no_response": sum(1 for r in items if r["status"] == "no_response"),
            "n_evaluable_no_response": sum(
                1 for r in items
                if r["technical_evaluable"] and r["status"] == "no_response"
            ),
            "n_delivery_verified": sum(1 for r in items if r["delivery_verified"]),
            "n_stimulus_integrity_verified": sum(
                1 for r in items if r["stimulus_integrity_verified"]
            ),
            "n_premature_assistant_onset": sum(
                1 for r in items if r.get("premature_assistant_onset")
            ),
            "n_overlap_events": sum(
                1 for r in items if r.get("overlap_event")
            ),
            "response_latency_mean_ms": latency_mean,
            "response_latency_sd_ms": latency_sd,
            "response_latency_evaluable_mean_ms": evaluable_latency_mean,
            "response_latency_evaluable_sd_ms": evaluable_latency_sd,
            "yield_latency_mean_ms": yield_mean,
            "yield_latency_sd_ms": yield_sd,
            "yield_latency_evaluable_mean_ms": evaluable_yield_mean,
            "yield_latency_evaluable_sd_ms": evaluable_yield_sd,
            "n_valid_interruption_onset": sum(
                1 for r in items if r.get("assistant_speaking_at_interrupt") is True
            ),
            "n_yielded_before_input_offset": sum(
                1 for r in items
                if r.get("assistant_yielded_before_input_offset") is True
            ),
            "n_response_resumed": sum(
                1 for r in items if r.get("response_resumed") is True
            ),
            "response_resumption_latency_mean_ms": resumption_mean,
            "response_resumption_latency_sd_ms": resumption_sd,
            "response_resumption_latency_evaluable_mean_ms": (
                evaluable_resumption_mean
            ),
            "response_resumption_latency_evaluable_sd_ms": (
                evaluable_resumption_sd
            ),
            "overlap_ms_total": round(sum(r["overlap_ms"] or 0 for r in items), 1),
            "pp_onsets_during_clip": sum(r["pp_onsets_during_clip"] or 0 for r in items),
            "assistant_pause_overlap_ms_total": round(sum(
                r.get("assistant_speech_during_prespecified_pause_ms") or 0
                for r in items
            ), 1),
            "assistant_onsets_during_prespecified_pauses": sum(
                r.get("assistant_onsets_during_prespecified_pause") or 0
                for r in items
            ),
            "technical_invalid_reason_counts": _invalid_reason_counts(items),
            "technical_warning_counts": _warning_counts(items),
        })

    bootstrap_seed = int(manifest.get("seed") or 0)
    cell_summary = _cell_summaries(rows, "index", bootstrap_seed)
    summary = {
        "schema": "hmo.soundboard-audit-summary.v2",
        "run": run_dir.name,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_schema": manifest.get("schema"),
        "assistant": manifest.get("assistant"),
        "audio_format": manifest.get("audio_format"),
        "generated_at_unix_s": time.time(),
        "detection": {
            "energy_run_gap_ms": float(
                timing.get("ppSpeakingGapMs")
                or DEFAULT_ENERGY_RUN_GAP_MS
            ),
            "pp_energy_threshold_rms": float(
                timing.get("ppEnergyThresholdRms")
                or DEFAULT_ENERGY_THRESHOLD_RMS
            ),
            "overlap_event_minimum_ms": float(
                timing.get("overlapEventMinimumMs")
                or DEFAULT_OVERLAP_EVENT_MIN_MS
            ),
            "interrupt_fire_delay_ms": float(
                timing.get("interruptFireDelayMs")
                or DEFAULT_INTERRUPT_FIRE_DELAY_MS
            ),
            "interrupt_fire_tolerance_ms": float(
                timing.get("interruptFireToleranceMs")
                or DEFAULT_INTERRUPT_FIRE_TOLERANCE_MS
            ),
            "timing_source": _timing_source_summary(timing_sources),
            "timing_source_counts": dict(Counter(timing_sources)),
            "legacy_packet_ms": (
                LEGACY_PACKET_MS
                if any(source == "legacy_energy_events"
                       for source in timing_sources)
                else None
            ),
        },
        "presentations": rows,
        "pairing_failures": pairing_failures,
        "item_condition_summary": summary_rows,
        "cell_summary": cell_summary,
        "clustered_uncertainty": {
            "method": "percentile_cluster_bootstrap",
            "confidence": 0.95,
            "repetitions": 2_000,
            "cluster": "fresh_personaplex_conversation",
            "cluster_key": "index",
            "seed": bootstrap_seed,
        },
        "counts": {
            "presentations": len(rows),
            "ok": sum(1 for r in rows if r["status"] == "ok"),
            "evaluable": sum(1 for r in rows if r["technical_evaluable"]),
            "technical_invalid": sum(
                1 for r in rows if not r["technical_evaluable"]
            ),
            "delivery_verified": sum(1 for r in rows if r["delivery_verified"]),
            "stimulus_integrity_verified": sum(
                1 for r in rows if r["stimulus_integrity_verified"]
            ),
            "technical_invalid_reason_counts": _invalid_reason_counts(rows),
            "technical_warning_counts": _warning_counts(rows),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_csv(run_dir / "audit_summary.csv", summary_rows)
    _write_csv(run_dir / "audit_cell_summary.csv", cell_summary)
    _write_csv(run_dir / "audit_presentations.csv", rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m study.audit_postprocess")
    parser.add_argument("--run", default=None,
                        help="process only this run directory name")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    data_root = Path(os.path.expanduser(
        os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))
    audit_root = data_root / "audit"
    if not audit_root.exists():
        print(json.dumps({"error": f"no audit directory at {audit_root}"}))
        return
    processed, skipped = [], []
    for run_dir in sorted(audit_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if args.run and run_dir.name != args.run:
            continue
        result = process_run(run_dir, force=args.force)
        if result is None:
            skipped.append(run_dir.name)
        else:
            processed.append({"run": run_dir.name,
                              "counts": result.get("counts"),
                              "pairing_failures": len(result.get("pairing_failures") or []),
                              "error": result.get("error")})
    print(json.dumps({"processed": processed, "skipped_existing": skipped},
                     indent=2))


if __name__ == "__main__":
    main()
