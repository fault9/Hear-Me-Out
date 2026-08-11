"""What is wrong with the automatic response gaps?

Verified boundaries minus detected ones, from the gaps judged so far. The two
candidate causes leave different signatures:

  a start that is too early, by a roughly constant amount unrelated to when in
  the session it happened, means the gap was measured from the end of a
  fragment rather than the end of a turn - response gaps are built from raw
  detector intervals while turn episodes group them at 1.5 s first;

  an error that grows with time into the session means the participant clock,
  since participant intervals are read off the recording with a constant
  offset while the assistant is timed from playback packets.

Also reports detector precision, which decides whether the automatic gaps are
usable at all.

    python3 analysis/gap_error.py [export-dir]
"""
import csv, json, os, statistics, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/Hear-Me-Out/services/app_api"))
from study.artifacts import resolve_artifact_path  # noqa: E402
from study.session_scope import annotate_analysis_scopes  # noqa: E402
from study.storage import get_backend  # noqa: E402

ROOT = Path(os.environ.get("STUDY_DATA_ROOT") or os.path.expanduser("~/study-data"))


def drift_ratio(session: dict) -> float:
    raw = (session.get("files") or {}).get("participant_raw")
    resolved = resolve_artifact_path(ROOT, raw)
    if resolved is None:
        return 1.0
    timeline = resolved.parent / "client_timeline.json"
    if not timeline.is_file():
        return 1.0
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
    return sorted(values)[len(values) // 2] if values else 1.0


def latest_export() -> Path:
    exports = sorted((ROOT / "exports" / "study1").glob("dataset_*"))
    if not exports:
        raise SystemExit("no export found")
    return exports[-1]


def summarise(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label:26s} (none)")
        return
    ordered = sorted(values)
    print(f"  {label:26s} n={len(values):3d}  median={ordered[len(ordered) // 2]:8.0f}ms"
          f"  mean={statistics.fmean(values):8.0f}ms"
          f"  range=[{ordered[0]:.0f}, {ordered[-1]:.0f}]")


export = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_export()
with (export / "turn_gap_verification_queue.csv").open(newline="") as handle:
    queue = {f"{r['session_id']}:{r['gap_id']}": r for r in csv.DictReader(handle)}

store = ROOT / "review" / "study1" / "turn_gap_verdicts.jsonl"
verdicts = {}
for line in store.read_text().splitlines() if store.is_file() else []:
    if line.strip():
        row = json.loads(line)
        verdicts[str(row.get("gap_key"))] = row
if not verdicts:
    raise SystemExit(f"no gap verdicts at {store}")

backend = get_backend()
ratios = {str(s["session_id"]): drift_ratio(s) for s in annotate_analysis_scopes(
    backend.list_sessions(1), backend.list_runs(1))}

real, spurious, rows = 0, 0, []
for key, verdict in verdicts.items():
    row = queue.get(key)
    if row is None:
        continue
    if str(verdict.get("verified_positive_gap")) not in ("1", "True", "true"):
        spurious += 1
        continue
    real += 1
    try:
        rows.append({
            "session_id": row["session_id"],
            "at_s": float(row["gap_start_ms"]) / 1000.0,
            "ratio": ratios.get(row["session_id"], 1.0),
            "start": float(verdict["verified_gap_start_ms"]) - float(row["gap_start_ms"]),
            "end": float(verdict["verified_gap_end_ms"]) - float(row["gap_end_ms"]),
            "duration": (float(verdict["verified_gap_duration_ms"])
                         - float(row["gap_duration_ms"])),
        })
    except (KeyError, TypeError, ValueError):
        continue

judged = real + spurious
print(f"export {export.name}: {judged} gaps judged, "
      f"{real} real ({100 * real / judged:.0f}% precision), {spurious} rejected\n")
if not rows:
    raise SystemExit("no verified boundaries recorded yet")

print(f"boundary error (verified minus detected), {len(rows)} gaps")
for field in ("start", "end", "duration"):
    summarise(field, [r[field] for r in rows])

# Grouping shifts the start and leaves the end alone; a clock error grows with
# time into the session and moves both.
print("\nby time into session")
for low, high in ((0, 60), (60, 120), (120, 200), (200, 1e9)):
    part = [r for r in rows if low <= r["at_s"] < high]
    if part:
        label = f"{low:.0f}-{high:.0f}s" if high < 1e9 else f"over {low:.0f}s"
        summarise(label + " start", [r["start"] for r in part])

print("\nby session clock")
for label, part in (("clean (<=1.005)", [r for r in rows if r["ratio"] <= 1.005]),
                    ("drifting (>1.005)", [r for r in rows if r["ratio"] > 1.005])):
    if part:
        summarise(label + " start", [r["start"] for r in part])
        summarise(label + " duration", [r["duration"] for r in part])
