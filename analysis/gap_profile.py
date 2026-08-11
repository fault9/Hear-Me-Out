"""Where do the dropped capture blocks land? Spread out is harmless; a burst
inside one utterance is not."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.path.expanduser("~/Hear-Me-Out/services/app_api"))
from study.storage import get_backend  # noqa: E402

ROOT = Path(os.environ.get("STUDY_DATA_DIR", os.path.expanduser("~/study-data")))
wanted = sys.argv[1:] or ["P01001", "P01002", "P01004", "P01005",
                          "P01006", "P01010", "P01017", "P01020"]
backend = get_backend()
for session in backend.list_sessions(1):
    sid = str(session.get("session_id") or "")
    if not any(sid.startswith(p) for p in wanted):
        continue
    latest = ((session.get("artifact_manifest") or {}).get("analysis")
              or {}).get("timing_latest") or {}
    path = latest.get("path") if isinstance(latest, dict) else None
    if not path:
        print(f"{sid}  no timing artifact")
        continue
    timing = json.loads((ROOT / path).read_text())
    caps = (timing.get("integrity") or {}).get("capture_gaps") or {}
    rows = caps.get("gaps") or []
    rate = float(caps.get("sample_rate_hz") or 16000)
    # A gap sits between two chunks; the chunk timeline says when in the
    # session it happened, which is what matters for a nominated window.
    chunks = {int(c["chunk_sequence"]): c
              for c in ((timing.get("capture") or {}).get("chunks") or [])}
    at = []
    for row in rows:
        chunk = chunks.get(int(row["after_chunk_sequence"]))
        if chunk is not None:
            at.append(float(chunk["timeline_start_ms"]) / 1000.0)
    at.sort()
    # Longest run of drops inside a 500 ms window: the burst that could eat a word.
    burst_ms, burst_at = 0.0, None
    for i, t in enumerate(at):
        window = [u for u in at[i:] if u - t <= 0.5]
        span = len(window) * float(caps.get("total_gap_ms") or 0) / max(len(rows), 1)
        if span > burst_ms:
            burst_ms, burst_at = span, t
    spacing = [b - a for a, b in zip(at, at[1:])]
    spacing.sort()
    print(f"{sid}  n={len(rows):4d}  total={caps.get('total_gap_ms'):8.0f}ms "
          f" max={caps.get('max_gap_ms'):5.0f}ms "
          f" spacing min/median={spacing[0] if spacing else 0:6.2f}/"
          f"{spacing[len(spacing)//2] if spacing else 0:6.2f}s "
          f" worst 0.5s burst={burst_ms:6.0f}ms"
          + (f" at {burst_at:.1f}s" if burst_at is not None else ""))
