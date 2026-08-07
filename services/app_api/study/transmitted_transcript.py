"""Interval-linked ASR of the TRANSMITTED participant track.

The dialogue transcript transcribes `participant_raw.wav` (what the
participant said); this module transcribes `participant.wav` (what the model
actually received after the voice-engine proxy) over the SAME participant
speech intervals, producing utterances with the SAME ids
(`participant_001`, ...). Raw and transmitted transcripts therefore stay
separately linked per interval, which is what the per-unit
"complete presence in the transmitted recording" delivery code consumes.

Alignment assumption (recorded in the artifact): the transmitted track is
assembled by the proxy from the same capture-chunk stream that produces the
raw track, chunk for chunk and duration-preserving, so the raw track's
browser-clock origin maps transmitted samples too. The capture crosswalk in
the timing integrity block is copied into the artifact; sessions with chunks
missing at the proxy are marked `alignment: degraded` so downstream coding
can treat their transmitted-presence codes with caution.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, file_record
from .dialogue_transcript import (Transcriber, _capture_origin_ms,
                                  participant_units, transcribe_words,
                                  words_by_unit)

TRANSMITTED_TRANSCRIPT_SCHEMA = "hmo.transmitted-transcript.v2"


def prepare_transmitted_transcript(session: dict, data_root: Path,
                                   analysis_id: str, timing: dict,
                                   transcriber: Transcriber | None = None) -> dict:
    """Create the transmitted-track transcript on the same intervals (and with
    the same utterance ids) as the dialogue transcript."""
    files = session.get("files") or {}
    if not files.get("participant"):
        raise FileNotFoundError("participant (transmitted) audio is missing")
    transmitted_path = data_root / files["participant"]
    if not transmitted_path.exists():
        raise FileNotFoundError(transmitted_path)
    session_dir = transmitted_path.parent
    intervals = timing.get("participant_intervals") or []
    if not intervals:
        raise ValueError("timing analysis has no participant speech intervals")

    integrity = timing.get("integrity") or {}
    missing_at_proxy = integrity.get("missing_at_proxy")
    alignment = ("degraded" if missing_at_proxy not in (0, None) else "assumed_chunk_aligned")

    origin_ms = _capture_origin_ms(session_dir, timing)
    units = participant_units(intervals)
    words = (transcriber or transcribe_words)(str(transmitted_path)).get("words") or []
    grouped, dropped = words_by_unit(words, units, origin_ms)
    utterances = []
    for index, (unit, own) in enumerate(zip(units, grouped)):
        utterances.append({
            "id": f"participant_{index + 1:03d}",
            "speaker": "participant",
            "start_ms": round(float(unit["start_ms"]), 3),
            "end_ms": round(float(unit["end_ms"]), 3),
            "text": " ".join(row["word"] for row in own).strip(),
            "text_provenance": {
                "source": "participant.wav (transmitted track)",
                "method": "whisper_word_timestamps_whole_file",
                "asr_status": "complete",
                "asr_error": None,
                "word_count": len(own),
                "words_outside_any_unit": dropped,
                "browser_clock_origin_ms": round(origin_ms, 3),
            },
        })

    result: dict[str, Any] = {
        "schema": TRANSMITTED_TRANSCRIPT_SCHEMA,
        "analysis_id": analysis_id,
        "session_id": session.get("session_id"),
        "created_at_unix_s": time.time(),
        "status": "complete",
        "alignment": alignment,
        "alignment_basis": {
            "assumption": "transmitted track chunk-aligned with raw capture",
            "capture_crosswalk_complete": integrity.get("capture_crosswalk_complete"),
            "missing_at_proxy": missing_at_proxy,
        },
        "timeline": {
            "name": "participant_experienced_browser_audio_clock",
            "timing_schema": timing.get("schema"),
            "speech_boundaries": "copied_from_timing_analysis_not_inferred_from_asr",
        },
        "utterances": utterances,
        "summary": {
            "participant_utterances": len(utterances),
            "words_outside_any_unit": dropped,
        },
        "sources": {
            "participant_transmitted": file_record(transmitted_path, relative_to=data_root),
            "timing": timing.get("result_artifact"),
        },
    }
    out_dir = session_dir / "analysis" / "transmitted" / analysis_id
    out_dir.mkdir(parents=True, exist_ok=False)
    result_path = out_dir / "transmitted_transcript.json"
    atomic_write_json(result_path, result, exclusive=True)
    result["result_artifact"] = file_record(result_path, relative_to=data_root)
    return result


def load_latest(session: dict, data_root: Path) -> dict | None:
    analysis = (session.get("artifact_manifest") or {}).get("analysis") or {}
    record = analysis.get("transmitted_transcript_latest") or {}
    path = record.get("path") if isinstance(record, dict) else None
    if not path:
        return None
    try:
        result = json.loads((Path(data_root) / path).read_text())
    except (OSError, ValueError):
        return None
    return result if isinstance(result, dict) else None
