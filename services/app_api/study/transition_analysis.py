"""Build reproducible, route-aware derived clips from immutable session audio."""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json, file_record


def read_events(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def route_regions(events: list[dict]) -> list[dict]:
    """Coalesce authoritative transmitted windows into contiguous route regions."""
    windows = sorted(
        (row for row in events if row.get("event") == "transmitted_window"),
        key=lambda row: (row.get("transmitted_start_sample", 0), row.get("event_sequence", 0)),
    )
    regions: list[dict] = []
    for row in windows:
        item = {
            "mode": row.get("route_mode", "natural"),
            "input_start_sample": int(row.get("input_start_sample", 0)),
            "input_end_sample": int(row.get("input_end_sample", 0)),
            "transmitted_start_sample": int(row.get("transmitted_start_sample", 0)),
            "transmitted_end_sample": int(row.get("transmitted_end_sample", 0)),
            "windows": 1,
        }
        previous = regions[-1] if regions else None
        if (previous and previous["mode"] == item["mode"]
                and previous["transmitted_end_sample"] == item["transmitted_start_sample"]):
            previous["input_end_sample"] = item["input_end_sample"]
            previous["transmitted_end_sample"] = item["transmitted_end_sample"]
            previous["windows"] += 1
        else:
            regions.append(item)

    # XVC emits a current window only after look-ahead audio arrives. Therefore
    # a transmitted_window's callback offsets are not the source interval for
    # that output. Route activations are the authoritative raw-input boundaries;
    # transmitted windows remain authoritative for the output boundaries.
    activations = sorted(
        (row for row in events if row.get("event") == "route_activated"),
        key=lambda row: row.get("event_sequence", 0),
    )
    stop = next((row for row in reversed(events) if row.get("event") == "stream_stop"), {})
    input_routes = []
    for index, row in enumerate(activations):
        end = (activations[index + 1].get("input_sample")
               if index + 1 < len(activations) else stop.get("input_samples"))
        input_routes.append({"mode": row.get("to_mode", "natural"),
                             "input_start_sample": int(row.get("input_sample", 0)),
                             "input_end_sample": int(end) if end is not None else None})
    cursor = 0
    for region in regions:
        for index in range(cursor, len(input_routes)):
            candidate = input_routes[index]
            if candidate["mode"] == region["mode"]:
                region["input_start_sample"] = candidate["input_start_sample"]
                if candidate["input_end_sample"] is not None:
                    region["input_end_sample"] = candidate["input_end_sample"]
                cursor = index + 1
                break
    return regions


def _read_wav(path: Path) -> tuple[wave._wave_params, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        params = wav.getparams()
        raw = wav.readframes(wav.getnframes())
    dtype = {1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}.get(params.sampwidth)
    if dtype is None:
        raise ValueError(f"unsupported WAV sample width: {params.sampwidth}")
    samples = np.frombuffer(raw, dtype=dtype)
    if params.nchannels > 1:
        samples = samples.reshape(-1, params.nchannels)
    return params, samples


def _write_slice(source: Path, destination: Path, start_s: float, end_s: float,
                 relative_to: Path) -> dict:
    params, samples = _read_wav(source)
    start = max(0, min(len(samples), round(start_s * params.framerate)))
    end = max(start, min(len(samples), round(end_s * params.framerate)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as wav:
        wav.setparams(params)
        wav.writeframes(samples[start:end].tobytes())
    return file_record(destination, relative_to=relative_to)


def _participant_speech_intervals(events: list[dict]) -> list[tuple[int, int]]:
    """Return ordered raw-input speech intervals from the recorded RMS events."""
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for row in sorted(events, key=lambda item: item.get("event_sequence", 0)):
        if row.get("event") == "participant_speech_start":
            start = int(row.get("input_sample", 0))
        elif row.get("event") == "participant_speech_end" and start is not None:
            end = int(row.get("input_sample", start))
            if end > start:
                intervals.append((start, end))
            start = None
    if start is not None:
        stop = next((row for row in reversed(events)
                     if row.get("event") == "stream_stop"), {})
        end = int(stop.get("input_samples", start))
        if end > start:
            intervals.append((start, end))
    return intervals


def _write_concatenated_slices(source: Path, destination: Path,
                               intervals_s: list[tuple[float, float]],
                               relative_to: Path,
                               separator_s: float = 0.15) -> dict:
    """Write ordered speech slices with a short silence between utterances."""
    params, samples = _read_wav(source)
    pieces: list[np.ndarray] = []
    separator_frames = max(0, round(separator_s * params.framerate))
    separator_shape = ((separator_frames, params.nchannels)
                       if samples.ndim == 2 else (separator_frames,))
    separator = np.zeros(separator_shape, dtype=samples.dtype)
    for start_s, end_s in intervals_s:
        start = max(0, min(len(samples), round(start_s * params.framerate)))
        end = max(start, min(len(samples), round(end_s * params.framerate)))
        if end <= start:
            continue
        if pieces and separator_frames:
            pieces.append(separator)
        pieces.append(samples[start:end])
    output = (np.concatenate(pieces) if pieces
              else np.empty((0,) + samples.shape[1:], dtype=samples.dtype))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as wav:
        wav.setparams(params)
        wav.writeframes(output.tobytes())
    record = file_record(destination, relative_to=relative_to)
    record.update({
        "selection": "participant_rms_speech_concatenation",
        "source_intervals": len(intervals_s),
        "separator_s": separator_s,
    })
    return record


def _mono_float(path: Path) -> tuple[int, np.ndarray]:
    params, samples = _read_wav(path)
    values = samples.astype(np.float64)
    if params.sampwidth == 1:
        values = values - 128.0
        scale = 128.0
    else:
        scale = float(2 ** (8 * params.sampwidth - 1))
    if values.ndim == 2:
        values = values.mean(axis=1)
    return params.framerate, values / scale


def _boundary_metrics(audio: np.ndarray, sample: int, sample_rate: int) -> dict:
    radius = max(1, round(0.02 * sample_rate))
    before = audio[max(0, sample - radius):sample]
    after = audio[sample:min(len(audio), sample + radius)]

    def rms(values: np.ndarray) -> float | None:
        return float(np.sqrt(np.mean(values * values))) if len(values) else None

    pre, post = rms(before), rms(after)
    delta_db = None
    if pre is not None and post is not None:
        delta_db = 20 * math.log10((post + 1e-9) / (pre + 1e-9))
    jump = (float(abs(audio[sample] - audio[sample - 1]))
            if 0 < sample < len(audio) else None)
    return {"window_ms": 20, "pre_rms": pre, "post_rms": post,
            "level_delta_db": delta_db, "boundary_jump_abs": jump}


def _sequence_gaps(events: list[dict], event: str, key: str) -> int:
    seq = [int(row[key]) for row in events if row.get("event") == event and row.get(key) is not None]
    return sum(max(0, current - previous - 1) for previous, current in zip(seq, seq[1:]))


def prepare_session_analysis(session: dict, data_root: Path, analysis_id: str,
                             guard_s: float = 0.5, transition_window_s: float = 1.0) -> dict:
    files = session.get("files") or {}
    transmitted = data_root / files["participant"]
    source = data_root / files["participant_raw"]
    target = transmitted.parent / "target.wav"
    event_path = transmitted.parent / "events.jsonl"
    if not target.exists():
        raise FileNotFoundError("the frozen target.wav is missing")
    if not event_path.exists():
        raise FileNotFoundError("events.jsonl is missing")

    events = read_events(event_path)
    regions = route_regions(events)
    speech_intervals = _participant_speech_intervals(events)
    out_dir = transmitted.parent / "analysis" / "vc_quality" / analysis_id
    out_dir.mkdir(parents=True, exist_ok=False)
    tx_rate, tx_audio = _mono_float(transmitted)

    derived = []
    score_jobs = []
    for index, region in enumerate(regions, start=1):
        input_start_s = region["input_start_sample"] / 16000.0
        input_end_s = region["input_end_sample"] / 16000.0
        tx_start_s = region["transmitted_start_sample"] / 16000.0
        tx_end_s = region["transmitted_end_sample"] / 16000.0
        trim = guard_s if len(regions) > 1 else 0.0
        stable_input = (input_start_s + trim, input_end_s - trim)
        stable_tx = (tx_start_s + trim, tx_end_s - trim)
        item = {**region, "index": index, "stable_guard_s": trim}
        if stable_input[1] > stable_input[0] and stable_tx[1] > stable_tx[0]:
            src_path = out_dir / f"region_{index:02d}_{region['mode']}_source.wav"
            tx_path = out_dir / f"region_{index:02d}_{region['mode']}_transmitted.wav"
            item["source"] = _write_slice(source, src_path, *stable_input, data_root)
            item["transmitted"] = _write_slice(transmitted, tx_path, *stable_tx, data_root)
            if region["mode"] == "vc":
                selected = [
                    (max(start, round(stable_input[0] * 16000)),
                     min(end, round(stable_input[1] * 16000)))
                    for start, end in speech_intervals
                    if end > stable_input[0] * 16000 and start < stable_input[1] * 16000
                ]
                selected = [(start, end) for start, end in selected if end > start]
                score_source = src_path
                score_transmitted = tx_path
                if selected:
                    padding_samples = round(0.2 * 16000)
                    input_start = round(stable_input[0] * 16000)
                    input_end = round(stable_input[1] * 16000)
                    tx_start = round(stable_tx[0] * 16000)
                    tx_end = round(stable_tx[1] * 16000)
                    input_span = max(1, input_end - input_start)
                    tx_span = max(1, tx_end - tx_start)
                    source_slices: list[tuple[float, float]] = []
                    transmitted_slices: list[tuple[float, float]] = []
                    for start, end in selected:
                        padded_start = max(input_start, start - padding_samples)
                        padded_end = min(input_end, end + padding_samples)
                        source_slices.append((padded_start / 16000, padded_end / 16000))
                        mapped_start = tx_start + round(
                            (padded_start - input_start) * tx_span / input_span)
                        mapped_end = tx_start + round(
                            (padded_end - input_start) * tx_span / input_span)
                        transmitted_slices.append(
                            (mapped_start / 16000, mapped_end / 16000))
                    score_source = out_dir / f"region_{index:02d}_vc_source_speech.wav"
                    score_transmitted = out_dir / f"region_{index:02d}_vc_transmitted_speech.wav"
                    item["score_source"] = _write_concatenated_slices(
                        source, score_source, source_slices, data_root)
                    item["score_transmitted"] = _write_concatenated_slices(
                        transmitted, score_transmitted, transmitted_slices, data_root)
                    item["score_selection"] = {
                        "mode": "participant_rms_speech_concatenation",
                        "speech_intervals": len(selected),
                        "boundary_padding_s": 0.2,
                        "mapping": "linear_within_route_region",
                    }
                else:
                    item["score_selection"] = {
                        "mode": "whole_region_fallback",
                        "reason": "no_participant_speech_events_in_region",
                    }
                score_jobs.append({
                    "region": index,
                    "source": str(score_source.relative_to(data_root)),
                    "converted": str(score_transmitted.relative_to(data_root)),
                    "target": str(target.relative_to(data_root)),
                })
        derived.append(item)

    activations = [row for row in events if row.get("event") == "route_activated"]
    transmitted_activations = [row for row in events
                               if row.get("event") == "transmitted_route_activated"]
    stream_start = next((row for row in events if row.get("event") == "stream_start"), {})
    transitions = []
    for index, row in enumerate(activations[1:], start=1):
        input_sample = int(row.get("input_sample", 0))
        tx_sample = int(row.get("transmitted_sample", 0))
        requested = row.get("requested_start_s")
        details: dict[str, Any] = {
            "index": index, "from_mode": row.get("from_mode"), "to_mode": row.get("to_mode"),
            "requested_start_s": requested, "actual_input_sample": input_sample,
            "actual_input_s": input_sample / 16000.0, "actual_transmitted_sample": tx_sample,
            "actual_transmitted_s": tx_sample / 16000.0,
            "activation_lag_input_samples": (input_sample - round(float(requested) * 16000))
            if requested is not None else None,
        }
        if details["activation_lag_input_samples"] is not None:
            details["activation_lag_ms"] = details["activation_lag_input_samples"] / 16.0
        if requested is not None and stream_start.get("monotonic_ns") and row.get("monotonic_ns"):
            checkpoint_ns = int(stream_start["monotonic_ns"]) + float(requested) * 1_000_000_000
            details["checkpoint_to_input_activation_ms"] = (
                int(row["monotonic_ns"]) - checkpoint_ns) / 1_000_000
        output_event = next((candidate for candidate in transmitted_activations
                             if candidate.get("to_mode") == row.get("to_mode")
                             and candidate.get("event_sequence", 0) >= row.get("event_sequence", 0)), None)
        if output_event:
            details["output_activation_monotonic_ns"] = output_event.get("monotonic_ns")
            details["actual_transmitted_sample"] = int(
                output_event.get("transmitted_sample", tx_sample))
            details["actual_transmitted_s"] = details["actual_transmitted_sample"] / 16000.0
            if row.get("monotonic_ns") and output_event.get("monotonic_ns"):
                details["input_to_output_activation_ms"] = (
                    int(output_event["monotonic_ns"]) - int(row["monotonic_ns"])) / 1_000_000
            tx_sample = details["actual_transmitted_sample"]
        details.update(_boundary_metrics(tx_audio, round(tx_sample * tx_rate / 16000), tx_rate))
        half = transition_window_s
        tx_clip = out_dir / f"transition_{index:02d}_{row.get('from_mode')}_to_{row.get('to_mode')}_transmitted.wav"
        src_clip = out_dir / f"transition_{index:02d}_{row.get('from_mode')}_to_{row.get('to_mode')}_source.wav"
        details["transmitted_clip"] = _write_slice(
            transmitted, tx_clip, max(0.0, tx_sample / 16000.0 - half),
            tx_sample / 16000.0 + half, data_root)
        details["source_clip"] = _write_slice(
            source, src_clip, max(0.0, input_sample / 16000.0 - half),
            input_sample / 16000.0 + half, data_root)
        transitions.append(details)

    manifest = {
        "schema": "hmo.vc-quality-inputs.v1", "analysis_id": analysis_id,
        "session_id": session["session_id"], "guard_s": guard_s,
        "transition_window_s": transition_window_s,
        "source_events": file_record(event_path, relative_to=data_root),
        "source_audio": file_record(source, relative_to=data_root),
        "transmitted_audio": file_record(transmitted, relative_to=data_root),
        "target_audio": file_record(target, relative_to=data_root),
        "regions": derived, "transitions": transitions, "score_jobs": score_jobs,
        "timeline_quality": {
            "input_chunk_sequence_gaps": _sequence_gaps(events, "input_chunk", "chunk_sequence"),
            "model_output_packet_sequence_gaps": _sequence_gaps(
                events, "personaplex_output_packet", "packet_sequence"),
            "inference_failures": sum(row.get("event") == "inference_failure" for row in events),
            "client_capture": next((row for row in reversed(events)
                                    if row.get("event") == "client_capture_summary"), None),
        },
    }
    atomic_write_json(out_dir / "inputs.manifest.json", manifest, exclusive=True)
    return manifest
