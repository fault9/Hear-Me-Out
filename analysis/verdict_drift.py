"""Did the capture drift reach the overlap verification?

Participant intervals are read off the recording with a constant offset while
the assistant is timed from playback packets, so where the recording lost time
the two tracks separate by (ratio - 1) x time into the session - and the review
player seeks both to the same stamps, so that separation is what the reviewer
heard.

If it mattered, nominated overlaps should be rejected more often exactly where
the predicted separation is large. If rejection is flat against it, the ear was
judging the speech rather than the alignment and the verdicts stand.

    python3 analysis/verdict_drift.py [export-dir]
"""
import collections, csv, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/Hear-Me-Out/services/app_api"))
from study.artifacts import resolve_artifact_path  # noqa: E402
from study.session_scope import annotate_analysis_scopes  # noqa: E402
from study.storage import get_backend  # noqa: E402

ROOT = Path(os.environ.get("STUDY_DATA_ROOT") or os.path.expanduser("~/study-data"))
CLEAN_RATIO = 1.005


def drift_ratio(session: dict) -> float:
    """Wall-clock ms per ms of audio, from the chunk timeline."""
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
    base = ROOT / "exports" / "study1"
    exports = sorted(base.glob("dataset_*"))
    if not exports:
        raise SystemExit(f"no export under {base}")
    return exports[-1]


export = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_export()
queue = export / "turn_verification_queue.csv"
if not queue.is_file():
    raise SystemExit(f"no turn_verification_queue.csv in {export}")
with queue.open(newline="") as handle:
    events = {row["event_key"]: row for row in csv.DictReader(handle)}

verdicts = {}
store = ROOT / "review" / "study1" / "turn_verdicts.jsonl"
for line in store.read_text().splitlines() if store.is_file() else []:
    if line.strip():
        row = json.loads(line)
        verdicts[str(row.get("event_key"))] = row     # last write wins

backend = get_backend()
sessions = {str(s["session_id"]): s for s in annotate_analysis_scopes(
    backend.list_sessions(1), backend.list_runs(1))}
ratios = {sid: drift_ratio(session) for sid, session in sessions.items()}

judged = []
for key, verdict in verdicts.items():
    event = events.get(key)
    if event is None:
        continue
    sid = event["session_id"]
    ratio = ratios.get(sid, 1.0)
    try:
        at_ms = float(event["overlap_start_ms"])
    except (TypeError, ValueError, KeyError):
        continue
    real = str(verdict.get("verified_overlap")) in ("1", "True", "true")
    judged.append((abs(ratio - 1.0) * at_ms, ratio, at_ms, real, sid))

if not judged:
    raise SystemExit("no verdicts matched this export's queue")

print(f"export {export.name}: {len(judged)} judged events matched\n")
print(f"{'predicted separation':22s} {'n':>5s} {'confirmed':>10s} {'rate':>7s}")
buckets = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 1e9)]
for low, high in buckets:
    rows = [r for r in judged if low <= r[0] < high]
    if not rows:
        continue
    real = sum(1 for r in rows if r[3])
    label = f"{low:.0f}-{high:.0f} ms" if high < 1e9 else f"over {low:.0f} ms"
    print(f"{label:22s} {len(rows):5d} {real:10d} {100 * real / len(rows):6.1f}%")

print(f"\n{'session clock':22s} {'n':>5s} {'confirmed':>10s} {'rate':>7s}")
for label, rows in (
        ("clean (<=1.005)", [r for r in judged if r[1] <= CLEAN_RATIO]),
        ("drifting (>1.005)", [r for r in judged if r[1] > CLEAN_RATIO])):
    if not rows:
        continue
    real = sum(1 for r in rows if r[3])
    print(f"{label:22s} {len(rows):5d} {real:10d} {100 * real / len(rows):6.1f}%")

print(f"\n{'participant':14s} {'ratio':>6s} {'n':>5s} {'confirmed':>10s} {'rate':>7s}")
by_participant = collections.defaultdict(list)
for row in judged:
    by_participant[row[4][:6]].append(row)
for participant, rows in sorted(by_participant.items()):
    real = sum(1 for r in rows if r[3])
    print(f"{participant:14s} {rows[0][1]:6.3f} {len(rows):5d} {real:10d} "
          f"{100 * real / len(rows):6.1f}%")
