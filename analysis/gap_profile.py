"""Where do the dropped capture blocks land? Spread out is harmless; a burst
inside one utterance is not.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/Hear-Me-Out/services/app_api"))
from study.artifacts import load_manifest_artifact, resolve_artifact_path  # noqa: E402
from study.storage import get_backend  # noqa: E402

ROOT = Path(os.environ.get("STUDY_DATA_DIR", os.path.expanduser("~/study-data")))
WINDOW_S = 0.5


def chunks_for(session: dict) -> dict:
    """Chunk sample boundaries stay in the client timeline; the timing
    artifact keeps only the gap summary derived from them."""
    raw = (session.get("files") or {}).get("participant_raw")
    resolved = resolve_artifact_path(ROOT, raw)
    if resolved is None:
        return {}
    timeline = resolved.parent / "client_timeline.json"
    if not timeline.is_file():
        return {}
    capture = (json.loads(timeline.read_text()).get("capture") or {})
    return {int(row["chunk_sequence"]): row for row in capture.get("chunks") or []}


def profile(timing: dict, chunks: dict) -> str:
    caps = (timing.get("integrity") or {}).get("capture_gaps") or {}
    rows = caps.get("gaps") or []
    if not rows:
        return f"clean  (total={caps.get('total_gap_ms') or 0:.0f}ms)"
    rate = float(caps.get("sample_rate_hz") or 16000)
    # A gap sits between two chunks; the chunk timeline says when in the
    # session it happened, which is what matters for a nominated window.
    drops = sorted(
        (float(chunks[int(r["after_chunk_sequence"])]["timeline_start_ms"]) / 1000.0,
         float(r["samples"]) * 1000.0 / rate)
        for r in rows if int(r["after_chunk_sequence"]) in chunks)
    if not drops:
        return f"n={len(rows)}  (no chunk timeline)"
    # The worst half-second: the burst that could actually swallow a word.
    burst_ms, burst_at = 0.0, drops[0][0]
    for index, (at, _) in enumerate(drops):
        total = sum(ms for other, ms in drops[index:] if other - at <= WINDOW_S)
        if total > burst_ms:
            burst_ms, burst_at = total, at
    spacing = sorted(b - a for (a, _), (b, _) in zip(drops, drops[1:]))
    return (f"n={len(rows):4d}  total={caps.get('total_gap_ms') or 0:7.0f}ms"
            f"  max={caps.get('max_gap_ms') or 0:5.0f}ms"
            f"  spacing min/median={spacing[0] if spacing else 0:5.2f}/"
            f"{spacing[len(spacing) // 2] if spacing else 0:5.2f}s"
            f"  worst {WINDOW_S:g}s burst={burst_ms:5.0f}ms at {burst_at:6.1f}s")


wanted = sys.argv[1:] or ["P01001", "P01002", "P01004", "P01005",
                          "P01006", "P01010", "P01012", "P01017", "P01020"]
backend = get_backend()
for session in sorted(backend.list_sessions(1),
                      key=lambda s: str(s.get("session_id") or "")):
    sid = str(session.get("session_id") or "")
    if not any(sid.startswith(p) for p in wanted) or "_S01_" in sid:
        continue
    timing = load_manifest_artifact(session, ROOT, "timing_latest")
    print(f"{sid}  "
          + (profile(timing, chunks_for(session)) if timing
             else "no timing artifact"))
