"""Did the capture clock keep real time?

Each chunk carries how many samples it holds and when the browser saw it. If
the device produced fewer samples per second than it declared, the WAV is
shorter than the conversation it recorded and plays back sped up - and every
onset, offset and gap measured from it is wrong by that factor.

Reported per session: wall-clock milliseconds elapsed per millisecond of
audio. 1.00 is a clock that kept time; above 1.00 the audio is compressed
(plays fast); the first/last figures show whether it drifted during the
session or was wrong throughout.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/Hear-Me-Out/services/app_api"))
from study.artifacts import (load_manifest_artifact,  # noqa: E402
                             resolve_artifact_path)
from study.dialogue_transcript import (CAPTURE_SHORTFALL_TOLERANCE,  # noqa: E402
                                       captured_duration_ms)
from study.session_scope import annotate_analysis_scopes  # noqa: E402
from study.storage import get_backend  # noqa: E402

ROOT = Path(os.environ.get("STUDY_DATA_ROOT") or os.path.expanduser("~/study-data"))


def ratios(session: dict) -> tuple[list[float], float, float]:
    raw = (session.get("files") or {}).get("participant_raw")
    resolved = resolve_artifact_path(ROOT, raw)
    if resolved is None:
        return [], 0.0, 0.0
    timeline = resolved.parent / "client_timeline.json"
    if not timeline.is_file():
        return [], 0.0, 0.0
    capture = json.loads(timeline.read_text()).get("capture") or {}
    rate = float(capture.get("sample_rate_hz") or 16000)
    chunks = sorted((c for c in capture.get("chunks") or []),
                    key=lambda c: int(c.get("chunk_sequence") or 0))
    values = []
    for previous, current in zip(chunks, chunks[1:]):
        audio_ms = float(previous.get("sample_count") or 0) / rate * 1000.0
        wall_ms = (float(current.get("timeline_start_ms") or 0)
                   - float(previous.get("timeline_start_ms") or 0))
        if audio_ms > 0 and wall_ms > 0:
            values.append(wall_ms / audio_ms)
    audio_total = sum(float(c.get("sample_count") or 0) for c in chunks) / rate
    wall_total = ((float(chunks[-1]["timeline_start_ms"])
                   - float(chunks[0]["timeline_start_ms"])) / 1000.0
                  if len(chunks) > 1 else 0.0)
    return values, audio_total, wall_total


def shortfall(session: dict) -> tuple[float, float, float]:
    """The scale the transcript pipeline would apply, by its own measure.

    Wall-clock span comes from the assistant track, timed from playback
    packets and the only clock in the session known to be intact.
    """
    raw = (session.get("files") or {}).get("participant_raw")
    resolved = resolve_artifact_path(ROOT, raw)
    timing = load_manifest_artifact(session, ROOT, "timing_latest") or {}
    assistant = timing.get("assistant_intervals") or []
    if resolved is None or not assistant:
        return 0.0, 0.0, 0.0
    held_ms = captured_duration_ms(resolved)
    span_ms = max(float(row["end_ms"]) for row in assistant)
    if held_ms <= 0 or span_ms <= 0:
        return 0.0, 0.0, 0.0
    return span_ms / held_ms, held_ms / 1000.0, span_ms / 1000.0


backend = get_backend()
# Inclusion is derived, not stored: raw sessions carry no analysis scope.
sessions = annotate_analysis_scopes(backend.list_sessions(1), backend.list_runs(1))
print(f"{'session':26s} {'in':>2s} {'scale':>6s} {'held_s':>7s} {'span_s':>7s} "
      f"{'chunk':>6s}   note")
for session in sorted(sessions, key=lambda s: str(s.get("session_id") or "")):
    sid = str(session.get("session_id") or "")
    if "_S01_" in sid:
        continue
    mark = "1" if session.get("analysis_included") else "-"
    scale, held_s, span_s = shortfall(session)
    values, _, _ = ratios(session)
    chunk = sorted(values)[len(values) // 2] if values else 0.0
    if not scale:
        print(f"{sid:26s} {mark:>2s}   no timing artifact")
        continue
    lost = span_s - held_s
    note = ""
    if scale - 1.0 > CAPTURE_SHORTFALL_TOLERANCE:
        note = f"<-- compressed, {lost:.0f}s missing; transcript rescaled, timing NOT"
    print(f"{sid:26s} {mark:>2s} {scale:6.3f} {held_s:7.1f} {span_s:7.1f} "
          f"{chunk:6.3f}   {note}")
