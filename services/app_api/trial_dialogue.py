"""Re-derive one session's dialogue transcript into a throwaway analysis id.

Writes analysis/dialogue/<id>/ inside the session directory and prints the
result. The manifest is not updated, so the canonical transcript and every
consumer of it are untouched: this is for checking a transcript change on a
real session before committing a whole batch to it.

Usage: STUDY_DATA_ROOT=/workspace/data uv run python trial_dialogue.py \
           P01001_R01_S03_A01 [analysis-id] [max-lines]
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

from study.dialogue_transcript import prepare_dialogue_transcript

COLUMNS = ("files_json, transcript_json, schedule_json, voice_condition, "
           "artifact_manifest_json")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    session_id = sys.argv[1]
    analysis_id = sys.argv[2] if len(sys.argv) > 2 else "trial"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    root = Path(os.path.expanduser(
        os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))
    media = Path(os.environ.get("STUDY_DATA_DIR", str(root / "media")))
    connection = sqlite3.connect(str(root / "study.db"))
    row = connection.execute(
        f"SELECT {COLUMNS} FROM session WHERE session_id=?",
        (session_id,)).fetchone()
    if not row:
        sys.exit(f"unknown session {session_id}")
    files, transcript, schedule, condition, manifest_json = row
    manifest = json.loads(manifest_json or "{}")
    timing_path = ((manifest.get("analysis") or {}).get("timing_latest")
                   or {}).get("path")
    if not timing_path:
        sys.exit("session has no timing artifact")
    timing = json.loads((media / timing_path).read_text())

    session = {
        "session_id": session_id,
        "voice_condition": condition,
        "files": json.loads(files or "{}"),
        "transcript": json.loads(transcript or "{}"),
        "schedule": json.loads(schedule or "[]"),
        "artifact_manifest": manifest,
    }
    result = prepare_dialogue_transcript(session, media, analysis_id, timing)
    print(result["schema"], json.dumps(result["summary"]))
    provenance = next((row["text_provenance"] for row in result["utterances"]
                       if row["speaker"] == "participant"), {})
    print("participant method:", provenance.get("method"))
    print()
    for row in result["utterances"][:limit]:
        who = "A" if row["speaker"] == "assistant" else "P"
        print("%s %7.1f-%7.1f %s" % (
            who, row["start_ms"] / 1000, row["end_ms"] / 1000, row["text"]))


if __name__ == "__main__":
    main()
