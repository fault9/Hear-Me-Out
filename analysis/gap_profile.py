"""Where do the dropped capture blocks land? Spread out is harmless; a burst
inside one utterance is not.

The session manifest names a timing artifact, but a re-run leaves the old path
behind, so fall back to the newest timing.json actually on disk.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/Hear-Me-Out/services/app_api"))
from study.storage import get_backend  # noqa: E402

ROOT = Path(os.environ.get("STUDY_DATA_DIR", os.path.expanduser("~/study-data")))
WINDOW_S = 0.5


def timing_for(session: dict) -> tuple[dict | None, str]:
    latest = ((session.get("artifact_manifest") or {}).get("analysis")
              or {}).get("timing_latest") or {}
    path = latest.get("path") if isinstance(latest, dict) else None
    if path and (ROOT / path).is_file():
        return json.loads((ROOT / path).read_text()), "manifest"
    if not path:
        return None, "no timing artifact"
    found = sorted((ROOT / path).parent.parent.glob("*/timing.json"))
    if not found:
        return None, "artifact missing on disk"
    return json.loads(found[-1].read_text()), f"fallback {found[-1].parent.name}"


def profile(timing: dict) -> str:
    caps = (timing.get("integrity") or {}).get("capture_gaps") or {}
    rows = caps.get("gaps") or []
    rate = float(caps.get("sample_rate_hz") or 16000)
    chunks = {int(c["chunk_sequence"]): c
              for c in ((timing.get("capture") or {}).get("chunks") or [])}
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
    for i, (t, _) in enumerate(drops):
        total = sum(ms for u, ms in drops[i:] if u - t <= WINDOW_S)
        if total > burst_ms:
            burst_ms, burst_at = total, t
    spacing = sorted(b - a for (a, _), (b, _) in zip(drops, drops[1:]))
    return (f"n={len(rows):4d}  total={caps.get('total_gap_ms') or 0:7.0f}ms"
            f"  max={caps.get('max_gap_ms') or 0:5.0f}ms"
            f"  spacing min/median={spacing[0] if spacing else 0:5.2f}/"
            f"{spacing[len(spacing) // 2] if spacing else 0:5.2f}s"
            f"  worst {WINDOW_S:g}s burst={burst_ms:5.0f}ms at {burst_at:6.1f}s")


wanted = sys.argv[1:] or ["P01001", "P01002", "P01004", "P01005",
                          "P01006", "P01010", "P01017", "P01020"]
backend = get_backend()
for session in sorted(backend.list_sessions(1),
                      key=lambda s: str(s.get("session_id") or "")):
    sid = str(session.get("session_id") or "")
    if not any(sid.startswith(p) for p in wanted) or "_S01_" in sid:
        continue
    timing, source = timing_for(session)
    print(f"{sid}  {profile(timing) if timing else source}"
          + (f"  [{source}]" if timing and source != "manifest" else ""))
