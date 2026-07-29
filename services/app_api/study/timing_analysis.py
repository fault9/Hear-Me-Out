"""Post-hoc participant/proxy timeline reconstruction and timing measures."""

from __future__ import annotations

import json
import math
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json, file_record
from .transition_analysis import read_events

TIMING_SCHEMA = "hmo.timing-analysis.v2"


def _mono_float(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        if width not in (1, 2, 4):
            raise ValueError(f"unsupported PCM sample width: {width}")
        values = np.frombuffer(wav.readframes(wav.getnframes()), dtype={
            1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4"),
        }[width])
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    if width == 1:
        return rate, (values.astype(np.float64) - 128.0) / 128.0
    return rate, values.astype(np.float64) / float(2 ** (8 * width - 1))


def speech_intervals(audio: np.ndarray, sample_rate: int, *, offset_ms: float = 0.0,
                     threshold: float = 0.012, frame_ms: int = 20,
                     hangover_ms: int = 250,
                     minimum_speech_ms: int = 120) -> list[dict]:
    """Deterministic RMS intervals used as estimates pending pilot validation."""
    frame = max(1, round(sample_rate * frame_ms / 1000))
    active = []
    for start in range(0, len(audio), frame):
        chunk = audio[start:start + frame]
        rms = math.sqrt(float(np.mean(chunk * chunk))) if len(chunk) else 0.0
        active.append(rms >= threshold)
    max_gap = max(0, round(hangover_ms / frame_ms))
    intervals = []
    first = last = None
    silence = 0
    for index, is_active in enumerate(active):
        if is_active:
            if first is None:
                first = index
            last = index
            silence = 0
        elif first is not None:
            silence += 1
            if silence > max_gap:
                start_ms = offset_ms + first * frame_ms
                end_ms = offset_ms + (last + 1) * frame_ms
                if end_ms - start_ms >= minimum_speech_ms:
                    intervals.append({
                        "start_ms": start_ms, "end_ms": end_ms,
                        "detector": "rms",
                    })
                first = last = None
                silence = 0
    if first is not None and last is not None:
        start_ms = offset_ms + first * frame_ms
        end_ms = offset_ms + (last + 1) * frame_ms
        if end_ms - start_ms >= minimum_speech_ms:
            intervals.append({
                "start_ms": start_ms, "end_ms": end_ms,
                "detector": "rms",
            })
    return intervals


def assistant_intervals(playback: dict, *, model_audio: np.ndarray | None = None,
                        model_sample_rate: int | None = None,
                        threshold: float = 0.008,
                        hangover_ms: float = 250.0) -> list[dict]:
    """Detect audible assistant speech on the browser playback timeline.

    New captures persist decoded-packet RMS on the AudioContext schedule. Older
    captures fall back to RMS over model.wav, whose silence-preserving offsets
    begin at the PersonaPlex handshake. Packet cadence alone is not speech: PP
    sends Opus packets containing silence between turns.
    """
    packets = sorted(playback.get("assistant_packets") or [],
                     key=lambda row: row.get("timeline_start_ms", 0))
    output_latency = float(playback.get("output_latency_ms") or 0.0)
    packets_with_energy = [packet for packet in packets
                           if isinstance(packet.get("rms"), (int, float))]
    if not packets_with_energy and model_audio is not None and model_sample_rate:
        intervals = speech_intervals(
            model_audio, model_sample_rate, offset_ms=output_latency,
            threshold=threshold, hangover_ms=round(hangover_ms),
        )
        for interval in intervals:
            interval["detector"] = "model_wav_rms_playback_fallback"
        return intervals

    active_packets = [packet for packet in packets_with_energy
                      if float(packet["rms"]) >= threshold]
    intervals = []
    for packet in active_packets:
        start = float(packet.get("timeline_start_ms", 0)) + output_latency
        end = float(packet.get("timeline_end_ms", start)) + output_latency
        if intervals and start - intervals[-1]["end_ms"] <= hangover_ms:
            intervals[-1]["end_ms"] = max(intervals[-1]["end_ms"], end)
            intervals[-1]["last_packet_sequence"] = packet.get("packet_sequence")
        else:
            intervals.append({
                "start_ms": start, "end_ms": end,
                "first_packet_sequence": packet.get("packet_sequence"),
                "last_packet_sequence": packet.get("packet_sequence"),
                "detector": "decoded_packet_rms_audio_context",
            })
    return intervals


def _overlaps(participant: list[dict], assistant: list[dict], minimum_ms: float) -> list[dict]:
    rows = []
    for p_index, p in enumerate(participant):
        for a_index, a in enumerate(assistant):
            start = max(p["start_ms"], a["start_ms"])
            end = min(p["end_ms"], a["end_ms"])
            if end - start >= minimum_ms:
                rows.append({"participant_interval": p_index, "assistant_interval": a_index,
                             "start_ms": start, "end_ms": end,
                             "duration_ms": end - start})
    return rows


def _barge_ins(participant: list[dict], assistant: list[dict]) -> list[dict]:
    rows = []
    for p_index, p in enumerate(participant):
        active = next(((index, item) for index, item in enumerate(assistant)
                       if item["start_ms"] <= p["start_ms"] < item["end_ms"]), None)
        if active:
            a_index, item = active
            rows.append({
                "participant_interval": p_index,
                "assistant_interval": a_index,
                "participant_onset_ms": p["start_ms"],
                "assistant_stop_ms": item["end_ms"],
                "stop_latency_ms": item["end_ms"] - p["start_ms"],
            })
    return rows


def _route_switches(events: list[dict], chunks: dict[int, dict]) -> list[dict]:
    input_chunks = [row for row in events if row.get("event") == "input_chunk"]
    switches = []
    for event in events:
        if event.get("event") != "route_activated" or event.get("from_mode") is None:
            continue
        sample = int(event.get("input_sample", 0))
        source = next((row for row in input_chunks
                       if int(row.get("input_start_sample", 0)) <= sample
                       < int(row.get("input_end_sample", 0))), None)
        sequence = int((source or {}).get("browser_chunk_sequence") or 0)
        client = chunks.get(sequence)
        timeline_ms = None
        if source and client and client.get("timeline_start_ms") is not None:
            sample_rate = float(source.get("input_sample_rate_hz") or 16000)
            timeline_ms = float(client["timeline_start_ms"]) + (
                sample - int(source.get("input_start_sample", 0))) * 1000.0 / sample_rate
        switches.append({
            "from_mode": event.get("from_mode"), "to_mode": event.get("to_mode"),
            "proxy_input_sample": sample, "browser_chunk_sequence": sequence or None,
            "participant_timeline_ms": timeline_ms,
            "requested_start_s": event.get("requested_start_s"),
        })
    return switches


def validate_intervals(automatic: list[dict], human: list[dict],
                       tolerance_ms: float = 100.0) -> dict:
    """Compare interval boundaries with a manually annotated pilot reference."""
    remaining = set(range(len(human)))
    matches = []
    for automatic_index, item in enumerate(automatic):
        candidates = [(index, abs(item["start_ms"] - human[index]["start_ms"]))
                      for index in remaining]
        if not candidates:
            continue
        human_index, onset_error = min(candidates, key=lambda value: value[1])
        offset_error = abs(item["end_ms"] - human[human_index]["end_ms"])
        if onset_error <= tolerance_ms and offset_error <= tolerance_ms:
            remaining.remove(human_index)
            matches.append({"automatic": automatic_index, "human": human_index,
                            "onset_error_ms": onset_error, "offset_error_ms": offset_error})
    tp = len(matches)
    precision = tp / len(automatic) if automatic else (1.0 if not human else 0.0)
    recall = tp / len(human) if human else (1.0 if not automatic else 0.0)
    return {
        "tolerance_ms": tolerance_ms, "matches": matches,
        "precision": precision, "recall": recall,
        "mean_onset_error_ms": (sum(row["onset_error_ms"] for row in matches) / tp
                                if tp else None),
        "mean_offset_error_ms": (sum(row["offset_error_ms"] for row in matches) / tp
                                 if tp else None),
        "validated": False,
        "note": "Set validated only after prespecified pilot acceptance criteria pass.",
    }


def prepare_timing_analysis(session: dict, data_root: Path, analysis_id: str) -> dict:
    files = session.get("files") or {}
    raw_path = data_root / files["participant_raw"]
    session_dir = raw_path.parent
    timeline_path = session_dir / "client_timeline.json"
    events_path = session_dir / "events.jsonl"
    if not raw_path.exists() or not timeline_path.exists() or not events_path.exists():
        raise FileNotFoundError("raw audio, client timeline, or proxy events are missing")

    client = json.loads(timeline_path.read_text())
    events = read_events(events_path)
    capture = client.get("capture") or {}
    playback = client.get("playback") or {}
    chunks = {int(row["chunk_sequence"]): row for row in capture.get("chunks") or []}
    first_offset = next((float(row["timeline_start_ms"]) for row in capture.get("chunks") or []
                         if row.get("timeline_start_ms") is not None), 0.0)
    sample_rate, raw = _mono_float(raw_path)
    participant = speech_intervals(
        raw, sample_rate, offset_ms=first_offset,
        threshold=float(os.environ.get("STUDY_VAD_RMS_THRESHOLD", "0.012")),
    )
    model_path = data_root / files["model"] if files.get("model") else None
    model_rate = None
    model_audio = None
    if model_path is not None and model_path.exists():
        model_rate, model_audio = _mono_float(model_path)
    assistant_threshold = float(os.environ.get(
        "STUDY_ASSISTANT_VAD_RMS_THRESHOLD", "0.008"))
    assistant = assistant_intervals(
        playback, model_audio=model_audio, model_sample_rate=model_rate,
        threshold=assistant_threshold,
    )
    proxy_sequences = {int(row.get("browser_chunk_sequence")) for row in events
                       if row.get("event") == "input_chunk"
                       and row.get("browser_chunk_sequence") is not None}
    client_sequences = set(chunks)
    missing_at_proxy = sorted(client_sequences - proxy_sequences)
    missing_at_client = sorted(proxy_sequences - client_sequences)
    client_assistant_sequences = {
        int(row["packet_sequence"]) for row in playback.get("assistant_packets") or []
        if row.get("packet_sequence") is not None
    }
    proxy_assistant_sequences = {
        int(row["packet_sequence"]) for row in events
        if row.get("event") == "personaplex_output_packet"
        and row.get("tag") == 1 and row.get("packet_sequence") is not None
    }
    assistant_missing_at_client = sorted(
        proxy_assistant_sequences - client_assistant_sequences)
    assistant_missing_at_proxy = sorted(
        client_assistant_sequences - proxy_assistant_sequences)
    capture_crosswalk_complete = not missing_at_proxy and not missing_at_client
    # Decoder priming/trailing packets can legitimately produce zero samples and
    # therefore have no browser playback row. Only gaps inside the browser's
    # playable sequence range indicate a broken crosswalk.
    if client_assistant_sequences:
        first_playable = min(client_assistant_sequences)
        last_playable = max(client_assistant_sequences)
        internal_missing_at_client = sorted(
            sequence for sequence in assistant_missing_at_client
            if first_playable <= sequence <= last_playable)
    else:
        internal_missing_at_client = assistant_missing_at_client
    playback_crosswalk_complete = (not internal_missing_at_client
                                   and not assistant_missing_at_proxy)
    overlaps = _overlaps(participant, assistant, 200.0)
    barge_ins = _barge_ins(participant, assistant)
    result: dict[str, Any] = {
        "schema": TIMING_SCHEMA, "analysis_id": analysis_id,
        "session_id": session.get("session_id"),
        "status": "estimated_pending_validation",
        "participant_intervals": participant, "assistant_intervals": assistant,
        "overlaps": overlaps, "barge_ins": barge_ins,
        "route_switches": _route_switches(events, chunks),
        "summary": {
            "participant_speech_intervals": len(participant),
            "assistant_speech_intervals": len(assistant),
            "overlap_events_200ms": len(overlaps),
            "barge_in_attempts": len(barge_ins),
            "mean_stop_latency_ms": (sum(row["stop_latency_ms"] for row in barge_ins)
                                     / len(barge_ins) if barge_ins else None),
        },
        "integrity": {
            "client_chunks": len(client_sequences), "proxy_chunks": len(proxy_sequences),
            "missing_at_proxy": missing_at_proxy, "missing_at_client": missing_at_client,
            "client_assistant_packets": len(client_assistant_sequences),
            "proxy_assistant_packets": len(proxy_assistant_sequences),
            "assistant_missing_at_client": assistant_missing_at_client,
            "assistant_internal_missing_at_client": internal_missing_at_client,
            "assistant_missing_at_proxy": assistant_missing_at_proxy,
            "capture_crosswalk_complete": capture_crosswalk_complete,
            "playback_crosswalk_complete": playback_crosswalk_complete,
            "crosswalk_complete": (capture_crosswalk_complete
                                   and playback_crosswalk_complete),
            "estimated_dropped_samples": capture.get("estimated_dropped_samples"),
            "assistant_vad_rms_threshold": assistant_threshold,
            "valid_for_timing": bool(participant and assistant
                                     and capture_crosswalk_complete
                                     and playback_crosswalk_complete),
        },
        "sources": {
            "participant_raw": file_record(raw_path, relative_to=data_root),
            "model": (file_record(model_path, relative_to=data_root)
                      if model_path is not None and model_path.exists() else None),
            "client_timeline": file_record(timeline_path, relative_to=data_root),
            "proxy_events": file_record(events_path, relative_to=data_root),
        },
    }
    out_dir = session_dir / "analysis" / "timing" / analysis_id
    out_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(out_dir / "timing.json", result, exclusive=True)
    result["result_artifact"] = file_record(
        out_dir / "timing.json", relative_to=data_root)
    return result
