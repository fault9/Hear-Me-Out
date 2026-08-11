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
from study.artifacts import resolve_artifact_path  # noqa: E402
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


backend = get_backend()
# Inclusion is derived, not stored: raw sessions carry no analysis scope.
sessions = annotate_analysis_scopes(backend.list_sessions(1), backend.list_runs(1))
print(f"{'session':26s} {'in':>2s} {'median':>7s} {'first10%':>9s} {'last10%':>8s} "
      f"{'audio_s':>8s} {'wall_s':>7s}")
for session in sorted(sessions, key=lambda s: str(s.get("session_id") or "")):
    sid = str(session.get("session_id") or "")
    if "_S01_" in sid:
        continue
    mark = "1" if session.get("analysis_included") else "-"
    values, audio_s, wall_s = ratios(session)
    if not values:
        print(f"{sid:26s} {mark:>2s}  no capture timeline")
        continue
    ordered = sorted(values)
    edge = max(1, len(values) // 10)
    flag = "  <-- audio is compressed" if audio_s and wall_s / audio_s > 1.02 else ""
    print(f"{sid:26s} {mark:>2s} {ordered[len(ordered) // 2]:7.3f} "
          f"{sum(values[:edge]) / edge:9.3f} {sum(values[-edge:]) / edge:8.3f} "
          f"{audio_s:8.1f} {wall_s:7.1f}{flag}")
