"""Does the capture clock drift depend on whether voice conversion is running?

If it does, the recording defect is confounded with the manipulation: natural
stretches would be measured on a different time base from converted ones, and
in the transition conditions the split falls exactly on the boundary the study
is about.

Transition sessions carry the test within themselves - same participant, same
device, same recording, VC on for one stretch and off for the other - so a
difference across that boundary is the route and not the hardware.

Reported per route region: wall-clock milliseconds elapsed per millisecond of
audio. 1.00 is a clock that kept time; above it the recording is losing audio.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/Hear-Me-Out/services/app_api"))
from study.artifacts import (load_manifest_artifact,  # noqa: E402
                             resolve_artifact_path)
from study.dialogue_transcript import _route_regions  # noqa: E402
from study.session_scope import annotate_analysis_scopes  # noqa: E402
from study.storage import get_backend  # noqa: E402

ROOT = Path(os.environ.get("STUDY_DATA_ROOT") or os.path.expanduser("~/study-data"))


def drift_by_region(session: dict, timing: dict) -> list[tuple[str, float, int]]:
    raw = (session.get("files") or {}).get("participant_raw")
    resolved = resolve_artifact_path(ROOT, raw)
    if resolved is None:
        return []
    timeline = resolved.parent / "client_timeline.json"
    if not timeline.is_file():
        return []
    capture = json.loads(timeline.read_text()).get("capture") or {}
    rate = float(capture.get("sample_rate_hz") or 16000)
    chunks = sorted((c for c in capture.get("chunks") or []),
                    key=lambda c: int(c.get("chunk_sequence") or 0))
    if len(chunks) < 2:
        return []
    ends = [float(row["end_ms"]) for row in
            (timing.get("participant_intervals") or [])
            + (timing.get("assistant_intervals") or [])
            if row.get("end_ms") is not None]
    regions = _route_regions(session, timing, max(ends, default=0.0))
    # Each ratio is charged to the region holding the chunk that produced it.
    buckets: dict[str, list[float]] = {}
    for previous, current in zip(chunks, chunks[1:]):
        audio_ms = float(previous.get("sample_count") or 0) / rate * 1000.0
        at_ms = float(previous.get("timeline_start_ms") or 0)
        wall_ms = float(current.get("timeline_start_ms") or 0) - at_ms
        if audio_ms <= 0 or wall_ms <= 0:
            continue
        mode = next((str(r["mode"]) for r in regions
                     if r["start_ms"] <= at_ms < r["end_ms"]), None)
        if mode:
            buckets.setdefault(mode, []).append(wall_ms / audio_ms)
    return [(mode, sorted(values)[len(values) // 2], len(values))
            for mode, values in buckets.items()]


backend = get_backend()
sessions = annotate_analysis_scopes(backend.list_sessions(1), backend.list_runs(1))
print(f"{'session':26s} {'condition':18s}  per-route median wall-ms per audio-ms")
for session in sorted(sessions, key=lambda s: str(s.get("session_id") or "")):
    sid = str(session.get("session_id") or "")
    if "_S01_" in sid or not session.get("analysis_included"):
        continue
    timing = load_manifest_artifact(session, ROOT, "timing_latest")
    regions = drift_by_region(session, timing) if timing else []
    if not regions:
        continue
    shown = "  ".join(f"{mode}={median:.3f} (n={count})"
                      for mode, median, count in sorted(regions))
    spread = (max(r[1] for r in regions) - min(r[1] for r in regions)
              if len(regions) > 1 else 0.0)
    flag = "   <-- differs by route" if spread > 0.02 else ""
    print(f"{sid:26s} {str(session.get('voice_condition') or '?'):18s}  "
          f"{shown}{flag}")
