"""Print the per-session analysis ledger for a study.

Usage (on the container):  python3 scripts/analysis_ledger.py [study_id]

States: ANALYZED (ended + metrics + VC complete/skipped + validity verdict),
PARTIAL (ended but pipeline incomplete -> run analysis), husk (never ended),
practice (never VC-scored by design).
"""

import json
import os
import sqlite3
import sys

DB = os.path.join(os.environ.get("STUDY_DATA_ROOT", "/workspace/data"), "study.db")
STUDY = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    query = ("SELECT session_id, participant_id, voice_condition, ended_at, "
             "metrics_json, vc_quality_status, artifact_manifest_json "
             "FROM session WHERE study_id=? ORDER BY participant_id, session_id")
    full, partial, husk = [], [], []
    last = None
    for row in conn.execute(query, (STUDY,)):
        if row["participant_id"] != last:
            last = row["participant_id"]
            print()
            print(last)
        manifest = json.loads(row["artifact_manifest_json"] or "{}")
        summary = (manifest.get("analysis") or {}).get("technical_validity_summary") or {}
        validity = summary.get("status", "-")
        ended = row["ended_at"] is not None
        metrics = bool(row["metrics_json"]) and row["metrics_json"] != "null"
        vc = row["vc_quality_status"] or "-"
        if row["voice_condition"] == "practice":
            state = "practice"
        elif not ended:
            state = "husk"
            husk.append(row["session_id"])
        elif metrics and vc in ("complete", "skipped") and validity not in ("pending", "-"):
            state = "ANALYZED"
            full.append(row["session_id"])
        else:
            state = "PARTIAL"
            partial.append(row["session_id"])
        print("  %-32s %-18s ended=%s metrics=%s vc=%-9s val=%-12s -> %s" % (
            row["session_id"], row["voice_condition"],
            "y" if ended else "N", "y" if metrics else "N", vc, validity, state))
    print()
    print("analyzed: %d | partial: %d | never-ended husks: %d"
          % (len(full), len(partial), len(husk)))
    if partial:
        print("needs attention:", ", ".join(partial))


if __name__ == "__main__":
    main()
