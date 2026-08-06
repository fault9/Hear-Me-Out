"""Derive a browser-clock dialogue transcript from immutable study artifacts."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable

from .artifacts import atomic_write_json, file_record

DIALOGUE_TRANSCRIPT_SCHEMA = "hmo.dialogue-transcript.v6"
# Silence that ends an assistant turn, measured on the same packet-RMS speech
# detection as the timing analysis. See _merge_assistant_runs.
ASSISTANT_TURN_SILENCE_MS = float(
    os.environ.get("ASSISTANT_TURN_SILENCE_MS", "1500"))
_ASR_PADDING_MS = 200.0

Transcriber = Callable[[str], dict]


def _default_transcriber(path: str) -> dict:
    from metrics import get_transcript_result

    return get_transcript_result(path)


def _capture_origin_ms(session_dir: Path, timing: dict) -> float:
    """Return the browser-clock time represented by raw WAV sample zero."""
    correction = float(
        ((timing.get("integrity") or {}).get(
            "participant_capture_latency_correction_ms")) or 0.0
    )
    try:
        client = json.loads((session_dir / "client_timeline.json").read_text())
        chunks = ((client.get("capture") or {}).get("chunks") or [])
        first_offset = next(
            float(row["timeline_start_ms"])
            for row in chunks
            if row.get("timeline_start_ms") is not None
        )
    except (OSError, ValueError, TypeError, StopIteration):
        first_offset = 0.0
    return first_offset - correction


def _wav_duration_ms(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        return wav.getnframes() * 1000.0 / rate if rate else 0.0


def _write_wav_slice(source: Path, destination: Path,
                     start_ms: float, end_ms: float) -> tuple[float, float]:
    with wave.open(str(source), "rb") as wav:
        params = wav.getparams()
        total_frames = wav.getnframes()
        start_frame = max(0, min(total_frames, round(start_ms * params.framerate / 1000)))
        end_frame = max(start_frame, min(
            total_frames, round(end_ms * params.framerate / 1000)))
        wav.setpos(start_frame)
        frames = wav.readframes(end_frame - start_frame)
    with wave.open(str(destination), "wb") as wav:
        wav.setparams(params)
        wav.writeframes(frames)
    return (
        start_frame * 1000.0 / params.framerate,
        end_frame * 1000.0 / params.framerate,
    )


def _clip_window(intervals: list[dict], index: int, origin_ms: float,
                 duration_ms: float) -> tuple[float, float]:
    item = intervals[index]
    start = float(item["start_ms"])
    end = float(item["end_ms"])
    padded_start = start - _ASR_PADDING_MS
    padded_end = end + _ASR_PADDING_MS
    if index:
        previous_end = float(intervals[index - 1]["end_ms"])
        padded_start = max(padded_start, (previous_end + start) / 2)
    if index + 1 < len(intervals):
        next_start = float(intervals[index + 1]["start_ms"])
        padded_end = min(padded_end, (end + next_start) / 2)
    return (
        max(0.0, min(duration_ms, padded_start - origin_ms)),
        max(0.0, min(duration_ms, padded_end - origin_ms)),
    )


def _initial_voice_mode(session: dict, timing: dict) -> str:
    switches = timing.get("route_switches") or []
    if switches and switches[0].get("from_mode"):
        return str(switches[0]["from_mode"])
    schedule = session.get("schedule") or []
    if schedule and schedule[0].get("mode"):
        return str(schedule[0]["mode"])
    return "vc" if session.get("voice_condition") in {
        "stable_converted", "vc_deactivation",
    } else "natural"


def _route_regions(session: dict, timing: dict, end_ms: float) -> list[dict]:
    switches = sorted(
        (row for row in timing.get("route_switches") or []
         if row.get("participant_timeline_ms") is not None),
        key=lambda row: float(row["participant_timeline_ms"]),
    )
    mode = _initial_voice_mode(session, timing)
    cursor = 0.0
    regions: list[dict] = []
    for switch in switches:
        boundary = max(cursor, float(switch["participant_timeline_ms"]))
        if boundary > cursor:
            regions.append({"mode": mode, "start_ms": cursor, "end_ms": boundary})
        mode = str(switch.get("to_mode") or mode)
        cursor = boundary
    regions.append({"mode": mode, "start_ms": cursor, "end_ms": max(cursor, end_ms)})
    return regions


def _routes_for_interval(start_ms: float, end_ms: float,
                         regions: list[dict]) -> tuple[str | None, list[dict]]:
    covered = []
    for region in regions:
        start = max(start_ms, float(region["start_ms"]))
        end = min(end_ms, float(region["end_ms"]))
        if end > start:
            covered.append({
                "mode": region["mode"],
                "start_ms": round(start, 3),
                "end_ms": round(end, 3),
            })
    modes = list(dict.fromkeys(row["mode"] for row in covered))
    voice_mode = modes[0] if len(modes) == 1 else ("mixed" if modes else None)
    return voice_mode, covered


def _fragment_times(fragment: dict) -> tuple[float, float] | None:
    try:
        start = float(fragment["start"]) * 1000.0
        end = float(fragment["end"]) * 1000.0
    except (KeyError, TypeError, ValueError):
        return None
    return start, max(start, end)


def _speech_runs(intervals: list[dict]) -> list[list[float]]:
    runs: list[list[float]] = []
    for row in sorted(intervals, key=lambda item: float(item["start_ms"])):
        start, end = float(row["start_ms"]), float(row["end_ms"])
        if runs and start - runs[-1][1] <= ASSISTANT_TURN_SILENCE_MS:
            runs[-1][1] = max(runs[-1][1], end)
        else:
            runs.append([start, end])
    return runs


def _onset_for(end_ms: float, runs: list[list[float]]) -> float | None:
    """Start of the audible run the model's text belongs to.

    A fragment's end is anchored to the arrival of its text; its start is
    back-computed from word count and lands the turn seconds early, which
    ordered a participant's answer after the reply it prompted. The audible
    run carrying that arrival gives the onset the participant heard.
    """
    for start, end in runs:
        if start <= end_ms <= end:
            return start
    previous = [row for row in runs if row[1] <= end_ms]
    return previous[-1][0] if previous else None


def _participant_utterances(raw_path: Path, intervals: list[dict],
                            origin_ms: float, route_regions: list[dict],
                            transcriber: Transcriber, temporary: Path) -> list[dict]:
    duration_ms = _wav_duration_ms(raw_path)
    utterances = []
    for index, interval in enumerate(intervals):
        start_ms = float(interval["start_ms"])
        end_ms = float(interval["end_ms"])
        clip_start, clip_end = _clip_window(intervals, index, origin_ms, duration_ms)
        asr = {"text": "", "segments": [], "status": "failed",
               "error": "empty interval after browser-to-WAV clock mapping"}
        actual_start, actual_end = clip_start, clip_end
        if clip_end > clip_start:
            clip_path = temporary / f"participant_{index + 1:03d}.wav"
            actual_start, actual_end = _write_wav_slice(
                raw_path, clip_path, clip_start, clip_end)
            asr = transcriber(str(clip_path))
        voice_mode, route_segments = _routes_for_interval(
            start_ms, end_ms, route_regions)
        utterances.append({
            "id": f"participant_{index + 1:03d}",
            "speaker": "participant",
            "start_ms": round(start_ms, 3),
            "end_ms": round(end_ms, 3),
            "text": str(asr.get("text") or "").strip(),
            "voice_mode": voice_mode,
            "route_segments": route_segments,
            "timing": {
                "timeline": "browser_audio_clock",
                "source": "timing.participant_intervals",
                "detector": interval.get("detector"),
            },
            "text_provenance": {
                "source": "participant_raw.wav",
                "method": "whisper-small_interval_asr",
                "asr_status": asr.get("status"),
                "asr_error": asr.get("error"),
                "wav_start_ms": round(actual_start, 3),
                "wav_end_ms": round(actual_end, 3),
                "browser_clock_origin_ms": round(origin_ms, 3),
                "context_padding_ms": _ASR_PADDING_MS,
            },
        })
    return utterances


def _assistant_utterances(fragments: list[dict],
                          intervals: list[dict]) -> tuple[list[dict], list[dict]]:
    # The model emits one fragment per utterance, so the fragments already are
    # the assistant's turns in the order they were spoken. Mapping them onto the
    # detected speech intervals merged and emptied turns whenever the model
    # paused mid-utterance; the intervals are used only to place each turn's
    # onset, which the back-computed start put seconds early.
    runs = _speech_runs(intervals)
    utterances = []
    unassigned = []
    floor = 0.0
    for fragment in fragments:
        times = _fragment_times(fragment)
        text = str(fragment.get("text") or "").strip()
        if not times or not text:
            unassigned.append(copy.deepcopy(fragment))
            continue
        start, end = times
        onset = _onset_for(end, runs)
        anchor = "audible_run_onset" if onset is not None else "back_computed_from_word_count"
        start = max(onset if onset is not None else start, floor)
        end = max(end, start)
        floor = start
        utterances.append({
            "id": f"assistant_{len(utterances) + 1:03d}",
            "speaker": "assistant",
            "start_ms": round(start, 3),
            "end_ms": round(end, 3),
            "text": text,
            "timing": {
                "timeline": "browser_audio_clock",
                "source": "session.transcript.model",
                "end_anchor": "text_arrival",
                "start_anchor": anchor,
            },
            "text_provenance": {
                "source": "model_transcript.json",
                "method": "verbatim_model_fragment",
                "fragment_count": 1,
            },
        })
    return utterances, unassigned


def _assistant_silence_ms(start_ms: float, end_ms: float,
                          intervals: list[dict]) -> float:
    speech = sum(
        max(0.0, min(end_ms, float(row["end_ms"])) - max(start_ms, float(row["start_ms"])))
        for row in intervals)
    return max(0.0, end_ms - start_ms - speech)


def _merge_assistant_runs(utterances: list[dict],
                          intervals: list[dict]) -> list[dict]:
    # The model emits text roughly per sentence, so one spoken turn arrives as
    # several fragments. A run is closed by the participant speaking, which
    # keeps a barge-in visible at the point it happened, or by the assistant
    # falling silent: over the pooled sessions 55% of fragment junctions carry
    # under 0.5s of silence (continuous speech), and past ~1.5s the
    # distribution is a featureless tail out to 56s.
    merged: list[dict] = []
    for row in utterances:
        previous = merged[-1] if merged else None
        if (row["speaker"] != "assistant" or previous is None
                or previous["speaker"] != "assistant"
                or _assistant_silence_ms(previous["end_ms"], row["end_ms"],
                                         intervals) > ASSISTANT_TURN_SILENCE_MS):
            merged.append(copy.deepcopy(row))
            continue
        previous["text"] = " ".join(
            part for part in (previous["text"], row["text"]) if part)
        previous["end_ms"] = row["end_ms"]
        previous["text_provenance"]["fragment_count"] += 1
    index = 0
    for row in merged:
        if row["speaker"] == "assistant":
            index += 1
            row["id"] = f"assistant_{index:03d}"
    return merged


def prepare_dialogue_transcript(session: dict, data_root: Path,
                                analysis_id: str, timing: dict,
                                transcriber: Transcriber | None = None) -> dict:
    """Create a versioned, time-aligned transcript on the browser audio clock."""
    files = session.get("files") or {}
    if not files.get("participant_raw"):
        raise FileNotFoundError("participant_raw audio is missing")
    raw_path = data_root / files["participant_raw"]
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    session_dir = raw_path.parent
    participant_intervals = timing.get("participant_intervals") or []
    assistant_intervals = timing.get("assistant_intervals") or []
    if not participant_intervals or not assistant_intervals:
        raise ValueError("timing analysis has no participant or assistant speech intervals")

    transcript = session.get("transcript") or {}
    model_fragments = transcript.get("model") or []
    end_ms = max(
        [float(row["end_ms"]) for row in participant_intervals + assistant_intervals],
        default=0.0,
    )
    origin_ms = _capture_origin_ms(session_dir, timing)
    regions = _route_regions(session, timing, end_ms)
    with tempfile.TemporaryDirectory(prefix="hmo-dialogue-") as temporary:
        participant = _participant_utterances(
            raw_path, participant_intervals, origin_ms, regions,
            transcriber or _default_transcriber, Path(temporary),
        )
    assistant, unassigned = _assistant_utterances(
        model_fragments, assistant_intervals)
    utterances = _merge_assistant_runs(sorted(
        participant + assistant,
        key=lambda row: (row["start_ms"], 0 if row["speaker"] == "assistant" else 1),
    ), assistant_intervals)
    failed_asr = sum(
        row["text_provenance"].get("asr_status") != "complete"
        for row in participant
    )
    result: dict[str, Any] = {
        "schema": DIALOGUE_TRANSCRIPT_SCHEMA,
        "analysis_id": analysis_id,
        "session_id": session.get("session_id"),
        "created_at_unix_s": time.time(),
        "status": "complete" if not failed_asr else "partial",
        "timeline": {
            "name": "participant_experienced_browser_audio_clock",
            "timing_schema": timing.get("schema"),
            "timing_status": timing.get("status"),
            "speech_boundaries": "copied_from_timing_analysis_not_inferred_from_asr",
            "assistant_times": "onset_from_audible_run_end_from_text_arrival",
            "assistant_turn_silence_ms": ASSISTANT_TURN_SILENCE_MS,
        },
        "utterances": utterances,
        "overlaps": copy.deepcopy(timing.get("overlaps") or []),
        "barge_ins": copy.deepcopy(timing.get("barge_ins") or []),
        "route_switches": copy.deepcopy(timing.get("route_switches") or []),
        "unassigned_model_fragments": unassigned,
        "summary": {
            "participant_utterances": len(participant),
            "assistant_utterances": sum(row["speaker"] == "assistant"
                                        for row in utterances),
            "model_fragments": len(assistant),
            "participant_asr_failures": failed_asr,
            "unassigned_model_fragments": len(unassigned),
        },
        "sources": {
            "participant_raw": file_record(raw_path, relative_to=data_root),
            "timing": timing.get("result_artifact"),
            "model_transcript": (
                (((session.get("artifact_manifest") or {}).get("artifacts") or {}).get(
                    "model_transcript"))
                or {"source": "session.transcript.model"}
            ),
        },
    }
    out_dir = session_dir / "analysis" / "dialogue" / analysis_id
    out_dir.mkdir(parents=True, exist_ok=False)
    result_path = out_dir / "dialogue_transcript.json"
    atomic_write_json(result_path, result, exclusive=True)
    result["result_artifact"] = file_record(result_path, relative_to=data_root)
    return result
