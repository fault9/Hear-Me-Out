"""Deterministic participant-only playback clips derived from study artifacts."""

from __future__ import annotations

import io
import json
import math
import os
import wave
from pathlib import Path

import numpy as np

from .artifacts import atomic_write_bytes, atomic_write_json, file_record
from .transition_analysis import read_events, route_regions


def _read_pcm16(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("playback derivation requires 16-bit PCM WAV audio")
        rate = source.getframerate()
        channels = source.getnchannels()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    if channels > 1:
        samples = np.rint(samples.reshape(-1, channels).mean(axis=1)).astype("<i2")
    return rate, samples


def _wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return output.getvalue()


def _speech_pieces(samples: np.ndarray, start: int, end: int, rate: int,
                   threshold: float) -> list[np.ndarray]:
    """Return complete RMS-active runs with a small amount of acoustic context."""
    start = max(0, min(len(samples), start))
    end = max(start, min(len(samples), end))
    frame = max(1, round(rate * 0.02))
    active_frames: list[int] = []
    for frame_index, offset in enumerate(range(start, end, frame)):
        chunk = samples[offset:min(end, offset + frame)].astype(np.float64) / 32768.0
        rms = math.sqrt(float(np.mean(chunk * chunk))) if len(chunk) else 0.0
        if rms >= threshold:
            active_frames.append(frame_index)
    if not active_frames:
        return []

    # Join activity separated by at most 200 ms and retain 100 ms of context.
    max_gap = max(1, round(0.2 / 0.02))
    pad = round(rate * 0.1)
    runs: list[tuple[int, int]] = []
    first = previous = active_frames[0]
    for current in active_frames[1:]:
        if current - previous > max_gap:
            runs.append((first, previous + 1))
            first = current
        previous = current
    runs.append((first, previous + 1))

    pieces = []
    for first_frame, last_frame in runs:
        run_start = max(start, start + first_frame * frame - pad)
        run_end = min(end, start + last_frame * frame + pad)
        pieces.append(samples[run_start:run_end])
    return pieces


def _condense_speech(samples: np.ndarray, start: int, end: int, rate: int,
                     threshold: float, join_silence_s: float = 0.08) -> np.ndarray:
    """Keep RMS-active speech runs while removing long conversational silence."""
    runs = _speech_pieces(samples, start, end, rate, threshold)
    if not runs:
        return np.zeros(0, dtype="<i2")
    join_silence = np.zeros(round(rate * join_silence_s), dtype="<i2")
    pieces = []
    for index, run in enumerate(runs):
        if index:
            pieces.append(join_silence)
        pieces.append(run)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype="<i2")


def ensure_transition_playback(session: dict, data_root: Path,
                               max_duration_s: int = 30) -> tuple[Path, dict]:
    """Create converted-then-natural speech around a VC-deactivation boundary."""
    files = session.get("files") or {}
    relative_source = files.get("participant")
    if not relative_source:
        raise FileNotFoundError("the transmitted participant recording is missing")
    source = data_root / relative_source
    events_path = source.parent / "events.jsonl"
    if not source.exists() or not events_path.exists():
        raise FileNotFoundError("playback source audio or route events are missing")

    duration_limit = max(1, min(int(max_duration_s or 30), 30))
    output_dir = source.parent / "derived" / "playback"
    output = output_dir / f"vc_deactivation_{duration_limit}s.wav"
    manifest_path = output.with_suffix(".json")
    if output.exists() and manifest_path.exists():
        return output, json.loads(manifest_path.read_text())

    events = read_events(events_path)
    regions = route_regions(events)
    pair = next(((left, right) for left, right in zip(regions, regions[1:])
                 if left["mode"] == "vc" and right["mode"] == "natural"), None)
    if pair is None:
        raise ValueError("no VC-deactivation route boundary was recorded")
    converted_region, natural_region = pair

    rate, samples = _read_pcm16(source)
    scale = rate / 16000.0
    threshold = float(os.environ.get(
        "STUDY_PLAYBACK_RMS_THRESHOLD",
        os.environ.get("STUDY_VAD_RMS_THRESHOLD", "0.012"),
    ))
    converted = _condense_speech(
        samples,
        round(converted_region["transmitted_start_sample"] * scale),
        round(converted_region["transmitted_end_sample"] * scale),
        rate, threshold,
    )
    natural = _condense_speech(
        samples,
        round(natural_region["transmitted_start_sample"] * scale),
        round(natural_region["transmitted_end_sample"] * scale),
        rate, threshold,
    )
    if not len(converted) or not len(natural):
        raise ValueError("eligible participant speech was not found on both sides of the switch")

    separator = np.zeros(round(rate * 0.2), dtype="<i2")
    available = max(2, round(duration_limit * rate) - len(separator))
    half = available // 2
    converted_count = min(len(converted), half)
    natural_count = min(len(natural), half)
    remaining = available - converted_count - natural_count
    converted_extra = min(len(converted) - converted_count, (remaining + 1) // 2)
    converted_count += converted_extra
    remaining -= converted_extra
    natural_extra = min(len(natural) - natural_count, remaining)
    natural_count += natural_extra
    remaining -= natural_extra
    converted_count += min(len(converted) - converted_count, remaining)

    clip = np.concatenate([
        converted[-converted_count:], separator, natural[:natural_count],
    ])
    if not output.exists():
        atomic_write_bytes(output, _wav_bytes(clip, rate), exclusive=True)
    manifest = {
        "schema": "hmo.playback-clip.v1",
        "session_id": session.get("session_id"),
        "selection": "rms_speech_around_vc_deactivation",
        "max_duration_s": duration_limit,
        "rms_threshold": threshold,
        "source_audio": file_record(source, relative_to=data_root),
        "source_events": file_record(events_path, relative_to=data_root),
        "converted_speech_s": converted_count / rate,
        "natural_speech_s": natural_count / rate,
        "output": file_record(output, relative_to=data_root),
    }
    atomic_write_json(manifest_path, manifest, exclusive=True)
    return output, manifest


def _playback_utterance_pool(utterances: list[np.ndarray], clip_index: int
                             ) -> tuple[list[tuple[int, np.ndarray]], str | None]:
    """Return a deterministic, non-overlapping utterance pool for one of two clips."""
    if clip_index not in (1, 2):
        raise ValueError("stable-converted playback clip_index must be 1 or 2")
    indexed = list(enumerate(utterances, start=1))
    if len(indexed) >= 2:
        split = max(1, len(indexed) // 2)
        return (indexed[:split] if clip_index == 1 else indexed[split:]), None

    # A one-turn interaction cannot provide two distinct utterances. Keep the
    # questionnaire usable by exposing disjoint halves and make the deviation
    # explicit in the immutable derived-clip manifest.
    utterance = indexed[0][1]
    split = max(1, len(utterance) // 2)
    if clip_index == 1:
        return [(1, utterance[:split])], "single_utterance_split_nonoverlapping"
    return [(2, utterance[split:])], "single_utterance_split_nonoverlapping"


def _fixed_duration_utterance_clip(
        pool: list[tuple[int, np.ndarray]], duration_samples: int,
        separator: np.ndarray) -> tuple[np.ndarray, list[int], int, bool]:
    """Pack complete utterances where possible, then pad to a fixed duration."""
    selected: list[np.ndarray] = []
    selected_indices: list[int] = []
    speech_samples = 0
    total_samples = 0
    truncated = False
    for utterance_index, utterance in pool:
        separator_samples = len(separator) if selected else 0
        available = duration_samples - total_samples - separator_samples
        if available <= 0:
            break
        if len(utterance) > available:
            if selected:
                break
            utterance = utterance[:available]
            truncated = True
        if selected:
            selected.append(separator)
            total_samples += len(separator)
        selected.append(utterance)
        selected_indices.append(utterance_index)
        speech_samples += len(utterance)
        total_samples += len(utterance)
    if not selected_indices:
        raise ValueError("eligible converted participant speech was not found for this clip")
    if total_samples < duration_samples:
        selected.append(np.zeros(duration_samples - total_samples, dtype="<i2"))
    return np.concatenate(selected), selected_indices, speech_samples, truncated


def ensure_stable_converted_playback(session: dict, data_root: Path,
                                     max_duration_s: int = 30,
                                     clip_index: int = 1) -> tuple[Path, dict]:
    """Create one of two participant-only excerpts from an all-VC condition."""
    files = session.get("files") or {}
    relative_source = files.get("participant")
    if not relative_source:
        raise FileNotFoundError("the transmitted participant recording is missing")
    source = data_root / relative_source
    events_path = source.parent / "events.jsonl"
    if not source.exists() or not events_path.exists():
        raise FileNotFoundError("playback source audio or route events are missing")

    duration_limit = max(1, min(int(max_duration_s or 30), 30))
    output_dir = source.parent / "derived" / "playback"
    output = output_dir / (
        f"stable_converted_participant_v3_clip{clip_index}_{duration_limit}s.wav")
    manifest_path = output.with_suffix(".json")
    if output.exists() and manifest_path.exists():
        return output, json.loads(manifest_path.read_text())

    regions = route_regions(read_events(events_path))
    if not regions or any(region.get("mode") != "vc" for region in regions):
        raise ValueError("the selected session is not a stable-converted condition")

    rate, samples = _read_pcm16(source)
    threshold = float(os.environ.get(
        "STUDY_PLAYBACK_RMS_THRESHOLD",
        os.environ.get("STUDY_VAD_RMS_THRESHOLD", "0.012"),
    ))
    utterances = _speech_pieces(samples, 0, len(samples), rate, threshold)
    if not utterances:
        raise ValueError("eligible converted participant speech was not found")

    pool, selection_fallback = _playback_utterance_pool(utterances, clip_index)
    max_samples = round(duration_limit * rate)
    join_silence_ms = 300
    separator = np.zeros(round(rate * join_silence_ms / 1000), dtype="<i2")
    clip, selected_indices, selected_speech_samples, truncated = (
        _fixed_duration_utterance_clip(pool, max_samples, separator))

    if not output.exists():
        atomic_write_bytes(output, _wav_bytes(clip, rate), exclusive=True)
    manifest = {
        "schema": "hmo.playback-clip.v1",
        "session_id": session.get("session_id"),
        "selection": "distinct_rms_utterances_from_stable_converted",
        "clip_index": clip_index,
        "max_duration_s": duration_limit,
        "rms_threshold": threshold,
        "detected_utterance_count": len(utterances),
        "selected_utterance_indices": selected_indices,
        "selected_utterance_count": len(selected_indices),
        "selection_fallback": selection_fallback,
        "truncated_utterance": truncated,
        "join_silence_ms": join_silence_ms,
        "source_audio": file_record(source, relative_to=data_root),
        "source_events": file_record(events_path, relative_to=data_root),
        "converted_speech_s": selected_speech_samples / rate,
        "output_duration_s": len(clip) / rate,
        "output": file_record(output, relative_to=data_root),
    }
    atomic_write_json(manifest_path, manifest, exclusive=True)
    return output, manifest


def ensure_stable_converted_interaction_playback(
        session: dict, data_root: Path, max_duration_s: int = 30) -> tuple[Path, dict]:
    """Create a contiguous user-and-assistant excerpt from an all-VC condition."""
    files = session.get("files") or {}
    relative_source = files.get("merged")
    if not relative_source:
        raise FileNotFoundError("the merged interaction recording is missing")
    source = data_root / relative_source
    events_path = source.parent / "events.jsonl"
    if not source.exists() or not events_path.exists():
        raise FileNotFoundError("playback source audio or route events are missing")

    duration_limit = max(1, min(int(max_duration_s or 30), 30))
    output_dir = source.parent / "derived" / "playback"
    output = output_dir / f"stable_converted_interaction_{duration_limit}s.wav"
    manifest_path = output.with_suffix(".json")
    if output.exists() and manifest_path.exists():
        return output, json.loads(manifest_path.read_text())

    regions = route_regions(read_events(events_path))
    if not regions or any(region.get("mode") != "vc" for region in regions):
        raise ValueError("the selected session is not a stable-converted condition")

    rate, samples = _read_pcm16(source)
    threshold = float(os.environ.get(
        "STUDY_PLAYBACK_RMS_THRESHOLD",
        os.environ.get("STUDY_VAD_RMS_THRESHOLD", "0.012"),
    ))
    frame = max(1, round(rate * 0.02))
    first_active = None
    for offset in range(0, len(samples), frame):
        chunk = samples[offset:offset + frame].astype(np.float64) / 32768.0
        rms = math.sqrt(float(np.mean(chunk * chunk))) if len(chunk) else 0.0
        if rms >= threshold:
            first_active = offset
            break
    if first_active is None:
        raise ValueError("eligible interaction audio was not found")

    start = max(0, first_active - round(0.5 * rate))
    end = min(len(samples), start + round(duration_limit * rate))
    clip = samples[start:end]
    if not len(clip):
        raise ValueError("the selected interaction excerpt is empty")

    if not output.exists():
        atomic_write_bytes(output, _wav_bytes(clip, rate), exclusive=True)
    manifest = {
        "schema": "hmo.playback-clip.v1",
        "session_id": session.get("session_id"),
        "selection": "contiguous_stable_converted_interaction",
        "max_duration_s": duration_limit,
        "rms_threshold": threshold,
        "source_start_s": start / rate,
        "source_end_s": end / rate,
        "source_audio": file_record(source, relative_to=data_root),
        "source_events": file_record(events_path, relative_to=data_root),
        "output": file_record(output, relative_to=data_root),
    }
    atomic_write_json(manifest_path, manifest, exclusive=True)
    return output, manifest
