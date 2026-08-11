"""What would grouping turns do to the response-gap candidates?

Response gaps are currently built from raw detector intervals while overlap
episodes group intervals into turns at 1.5 s first. This recomputes the gaps
both ways from the same timing artifacts and compares the candidate sets
against the verdicts already recorded.

The question it answers is whether the change is worth its cost: if the gaps a
listener rejected disappear under grouping while the gaps they accepted
survive with the same boundaries, then grouping removes false positives rather
than reshuffling the queue, and the recorded verdicts can be carried over by
boundary instead of being re-judged.

    python3 analysis/gap_grouping_preview.py [export-dir]
"""
import csv, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/Hear-Me-Out/services/app_api"))
from study.artifacts import load_manifest_artifact  # noqa: E402
from study.session_scope import annotate_analysis_scopes  # noqa: E402
from study.storage import get_backend  # noqa: E402
from study.turn_taking import (build_positive_response_gaps,  # noqa: E402
                               group_turns)

ROOT = Path(os.environ.get("STUDY_DATA_ROOT") or os.path.expanduser("~/study-data"))
TOLERANCE_MS = 20.0


def key(row: dict) -> tuple:
    """A gap identified by where it sits, not by its ordinal in the list."""
    return (round(float(row["gap_start_ms"]) / TOLERANCE_MS),
            round(float(row["gap_end_ms"]) / TOLERANCE_MS))


def latest_export() -> Path:
    exports = sorted((ROOT / "exports" / "study1").glob("dataset_*"))
    if not exports:
        raise SystemExit("no export found")
    return exports[-1]


export = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_export()
with (export / "turn_gap_verification_queue.csv").open(newline="") as handle:
    queued = list(csv.DictReader(handle))

store = ROOT / "review" / "study1" / "turn_gap_verdicts.jsonl"
verdicts = {}
for line in store.read_text().splitlines() if store.is_file() else []:
    if line.strip():
        row = json.loads(line)
        verdicts[str(row.get("gap_key"))] = row

backend = get_backend()
sessions = {str(s["session_id"]): s for s in annotate_analysis_scopes(
    backend.list_sessions(1), backend.list_runs(1))}

# Recompute both ways from the same intervals, per session.
grouped_keys, raw_counts, grouped_counts = {}, 0, 0
for sid in {r["session_id"] for r in queued}:
    session = sessions.get(sid)
    timing = load_manifest_artifact(session, ROOT, "timing_latest") if session else None
    if not timing:
        continue
    participant = timing.get("participant_intervals") or []
    assistant = timing.get("assistant_intervals") or []
    raw = build_positive_response_gaps(participant, assistant)
    grouped = build_positive_response_gaps(group_turns(participant),
                                           group_turns(assistant))
    raw_counts += len(raw)
    grouped_counts += len(grouped)
    grouped_keys[sid] = {key(g) for g in grouped}

print(f"export {export.name}")
print(f"candidates now (raw intervals):   {raw_counts}")
print(f"candidates if turns are grouped:  {grouped_counts} "
      f"({100 * grouped_counts / raw_counts:.0f}% of current)\n"
      if raw_counts else "")

judged, survives = {"real": [], "rejected": []}, {"real": 0, "rejected": 0}
for row in queued:
    verdict = verdicts.get(f"{row['session_id']}:{row['gap_id']}")
    if verdict is None:
        continue
    real = str(verdict.get("verified_positive_gap")) in ("1", "True", "true")
    bucket = "real" if real else "rejected"
    judged[bucket].append(row)
    if key(row) in grouped_keys.get(row["session_id"], set()):
        survives[bucket] += 1

for bucket in ("real", "rejected"):
    total = len(judged[bucket])
    if not total:
        continue
    kept = survives[bucket]
    print(f"judged {bucket:9s} {total:3d}   survive grouping: {kept:3d} "
          f"({100 * kept / total:3.0f}%)   lost: {total - kept}")

print("\nGrouping is worth its cost when the rejected gaps mostly vanish and\n"
      "the real ones mostly survive: the first is the false positives being\n"
      "removed, the second is the recorded verdicts being carryable by\n"
      "boundary rather than re-judged.")
