"""Derive a browser-clock dialogue transcript from immutable study artifacts."""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from .artifacts import atomic_write_json, file_record
from .turn_taking import group_turns

DIALOGUE_TRANSCRIPT_SCHEMA = "hmo.dialogue-transcript.v9"
# Transient starvation in the browser drops quiet capture buffers, so the
# participant recording holds fewer seconds than it spans and every later
# participant event is placed earlier than it happened. The assistant track is
# timed from playback packets and is unaffected, so the ratio between the two
# recovers the participant clock. Applied only past this shortfall: healthy
# sessions sit within 1% and must not be rescaled by estimator noise.
CAPTURE_SHORTFALL_TOLERANCE = 0.02
# Silence that ends an assistant turn, measured on the same packet-RMS speech
# detection as the timing analysis. See _merge_assistant_runs.
ASSISTANT_TURN_SILENCE_MS = float(
    os.environ.get("ASSISTANT_TURN_SILENCE_MS", "1500"))
_ASR_PADDING_MS = 200.0
# No spoken word lasts this long. Without VAD the recognizer occasionally
# stretches one across a silence, and an overlap rule takes such a word into
# any unit it brushes - one reached 9.5 s and pulled its text into a turn it
# was never spoken in. Placement uses at most this much of a word.
_MAX_WORD_MS = 2000.0

Transcriber = Callable[[str], dict]


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

    Text arriving in the silence after a run cannot take that run's start: an
    earlier fragment already holds it, and two turns sharing an onset order
    arbitrarily against the participant. Such arrivals anchor to where the
    assistant was last audible instead.
    """
    for start, end in runs:
        if start <= end_ms <= end:
            return start
    previous = [row for row in runs if row[1] <= end_ms]
    return previous[-1][1] if previous else None


def participant_units(intervals: list[dict]) -> list[dict]:
    """The participant's speech intervals grouped into the turns they were
    spoken as, on the same silence rule the turn measures use. One utterance
    per detected interval split single sentences across four lines whenever
    the speaker drew breath."""
    return [{"start_ms": turn["start_ms"], "end_ms": turn["end_ms"],
             "intervals": [member[0] for member in turn["members"]],
             "detector": intervals[turn["members"][0][0]].get("detector")}
            for turn in group_turns(intervals)]


_WORD_MODEL = None


def _word_model():
    """One model per process. Constructing it per call pinned GPU memory that
    was not released between sessions and exhausted the device mid-batch."""
    global _WORD_MODEL
    if _WORD_MODEL is None:
        from faster_whisper import WhisperModel

        device = os.environ.get("WHISPER_DEVICE", "cuda")
        compute = "int8_float16" if device == "cuda" else "int8"
        _WORD_MODEL = WhisperModel(os.environ.get("WHISPER_MODEL", "small"),
                                   device=device, compute_type=compute)
    return _WORD_MODEL


def transcribe_words(path: str) -> dict:
    """Whole-file transcription with word timings (the tier the HMO UI uses)."""
    model = _word_model()
    # No VAD: it decodes silence-stripped audio and maps word times back, and
    # that remapping drifts across long pauses - a word after a five-second
    # silence came back timed inside the previous speech interval. The RMS
    # intervals already say where speech is, and anything invented in a silence
    # region falls outside them all and is dropped rather than becoming text.
    segments, _ = model.transcribe(
        path, beam_size=5, language="en", word_timestamps=True,
        vad_filter=False, condition_on_previous_text=False)
    words = []
    for segment in segments:
        for word in getattr(segment, "words", None) or []:
            text = str(getattr(word, "word", "")).strip()
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if text and start is not None and end is not None:
                words.append({"word": text, "start": float(start),
                              "end": float(end)})
    return {"words": words, "status": "complete", "error": None}


def words_by_unit(words: list[dict], units: list[dict],
                  origin_ms: float) -> tuple[list[list[dict]], int]:
    """Group whole-file words onto the speech units they were spoken in.

    Slicing each unit and transcribing the slice gave the recognizer a few
    hundred milliseconds of breath or playback bleed with no context, and it
    answered with stock politeness ("thank you." on 127 of the 577 intervals
    under 600 ms). Transcribing the file once and keeping each unit's own words
    leaves such intervals empty, which is what they are. A word is placed by
    overlap rather than midpoint: one straddling an onset belongs to that unit,
    and dropping it costs content words - the reference number lost its "one"
    when the boundary fell mid-word.
    """
    grouped: list[list[dict]] = [[] for _ in units]
    dropped = 0
    for word in words:
        # Word times are file positions; the units are on the browser clock.
        start = origin_ms + float(word["start"]) * 1000.0
        end = min(origin_ms + float(word["end"]) * 1000.0, start + _MAX_WORD_MS)
        overlaps = [
            (min(end, float(unit["end_ms"])) - max(start, float(unit["start_ms"])),
             position)
            for position, unit in enumerate(units)]
        best = max(overlaps, default=None)
        if best is not None and best[0] > 0:
            index = best[1]
        else:

            distances = [
                (float(unit["start_ms"]) - end
                 if end < float(unit["start_ms"])
                 else start - float(unit["end_ms"]), position)
                for position, unit in enumerate(units)]
            nearest = min(distances, default=None)
            if nearest is None or nearest[0] > _ASR_PADDING_MS:
                dropped += 1
                continue
            index = nearest[1]
        grouped[index].append(word)
    return grouped, dropped


def _participant_utterances(units: list[dict], origin_ms: float,
                            route_regions: list[dict],
                            words: list[dict]) -> list[dict]:
    grouped, dropped = words_by_unit(words, units, origin_ms)
    utterances = []
    for index, (unit, own) in enumerate(zip(units, grouped)):
        start_ms = float(unit["start_ms"])
        end_ms = float(unit["end_ms"])
        voice_mode, route_segments = _routes_for_interval(
            start_ms, end_ms, route_regions)
        utterances.append({
            "id": f"participant_{index + 1:03d}",
            "speaker": "participant",
            "start_ms": round(start_ms, 3),
            "end_ms": round(end_ms, 3),
            "text": " ".join(row["word"] for row in own).strip(),
            "voice_mode": voice_mode,
            "route_segments": route_segments,
            "timing": {
                "timeline": "browser_audio_clock",
                "source": "timing.participant_intervals",
                "detector": unit.get("detector"),
                "intervals": unit.get("intervals"),
            },
            "text_provenance": {
                "source": "participant_raw.wav",
                "method": "whisper_word_timestamps_whole_file",
                "asr_status": "complete",
                "asr_error": None,
                "word_count": len(own),
                "words_outside_any_unit": dropped,
                "speech_start_ms": (round(origin_ms + own[0]["start"] * 1000.0, 3)
                                    if own else None),
                "speech_end_ms": (round(origin_ms + own[-1]["end"] * 1000.0, 3)
                                  if own else None),
                "browser_clock_origin_ms": round(origin_ms, 3),
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
        # The floor is the previous fragment's end: the model speaks its
        # fragments in sequence, so two of them can never share an onset.
        start = max(onset if onset is not None else start, floor)
        end = max(end, start)
        floor = end
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


def captured_duration_ms(raw_path: Path) -> float:
    import wave

    try:
        with wave.open(str(raw_path)) as handle:
            return handle.getnframes() / handle.getframerate() * 1000.0
    except (OSError, wave.Error):
        return 0.0


def capture_scale(raw_path: Path, assistant_intervals: list[dict]) -> float:
    """Wall-clock seconds per recorded second of participant audio.

    The denominator is what the file holds and the numerator is how long the
    assistant was still talking, which is the only clock in the session known
    to be intact. Returns 1.0 whenever the recording covers the session, so a
    healthy transcript is byte-identical to one built without this.
    """
    captured_ms = captured_duration_ms(raw_path)
    span_ms = max((float(row["end_ms"]) for row in assistant_intervals), default=0.0)
    if captured_ms <= 0 or span_ms <= 0:
        return 1.0
    scale = span_ms / captured_ms
    return scale if scale - 1.0 > CAPTURE_SHORTFALL_TOLERANCE else 1.0


def capture_time_map(intervals: list[dict], scale: float, origin: float,
                     captured_ms: float) -> Callable[[float], float]:
    """Map recorded time to wall-clock time for a session that lost buffers.

    What went missing is silence, so speech keeps the duration it was recorded
    with and the shortfall is shared out across the quiet stretches in
    proportion to the quiet each already holds. Stretching everything by one
    factor instead inflates each utterance by the same ratio, which places
    early turns late whenever the participant fell quiet more later on.
    """
    if scale == 1.0:
        return lambda ms: ms
    speech = sorted(((float(row["start_ms"]), float(row["end_ms"]))
                     for row in intervals), key=lambda pair: pair[0])
    end = max(captured_ms, speech[-1][1] if speech else 0.0)
    gaps, cursor = [], origin
    for start, stop in speech:
        gaps.append((cursor, max(start, cursor)))
        cursor = max(stop, cursor)
    gaps.append((cursor, max(end, cursor)))
    quiet = sum(stop - start for start, stop in gaps)
    missing = (end - origin) * (scale - 1.0)
    if quiet <= 0:
        return lambda ms: origin + (ms - origin) * scale
    stretch = 1.0 + missing / quiet

    def mapped(ms: float) -> float:
        shift = sum(min(max(ms - start, 0.0), stop - start) * (stretch - 1.0)
                    for start, stop in gaps)
        return ms + shift

    return mapped


def _remapped(rows: list[dict], mapped: Callable[[float], float],
              keys: tuple[str, ...]) -> list[dict]:
    return [{**row, **{key: mapped(float(row[key]))
                       for key in keys if row.get(key) is not None}}
            for row in rows]


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
    origin_ms = _capture_origin_ms(session_dir, timing)
    scale = capture_scale(raw_path, assistant_intervals)
    mapped = capture_time_map(participant_intervals, scale, origin_ms,
                              captured_duration_ms(raw_path))
    participant_intervals = _remapped(
        participant_intervals, mapped, ("start_ms", "end_ms"))
    end_ms = max(
        [float(row["end_ms"]) for row in participant_intervals + assistant_intervals],
        default=0.0,
    )
    regions = _route_regions(session, timing, end_ms)
    units = participant_units(participant_intervals)
    # Word times are file-relative seconds; the map works on the browser clock.
    words = [{**word, **{key: (mapped(origin_ms + float(word[key]) * 1000.0)
                               - origin_ms) / 1000.0
                         for key in ("start", "end") if word.get(key) is not None}}
             for word in (transcriber or transcribe_words)(
                 str(raw_path)).get("words") or []]
    participant = _participant_utterances(units, origin_ms, regions, words)
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
            "participant_capture_scale": scale,
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
