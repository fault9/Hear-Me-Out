"""Worker for one-session, one-participant, or whole-study VC-quality runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from study.artifacts import atomic_write_json, file_record  # noqa: E402
from study.storage import get_backend  # noqa: E402
from study.transition_analysis import prepare_session_analysis  # noqa: E402
from study.vc_quality_analysis import STUDY_DATA_DIR, status_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
VC_QUALITY_DIR = Path(os.environ.get("VC_QUAL_DIR", REPO_ROOT / "services" / "vc_quality"))
VC_QUALITY_SCRIPT = VC_QUALITY_DIR / "vc_quality.py"


def _write(**values) -> None:
    values["pid"] = os.getpid()
    values["heartbeat"] = time.time()
    path = status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(values))
    temp.replace(path)


def _parse_stdout(stdout: str) -> dict:
    text = stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ValueError("vc_quality.py produced no JSON")
        return json.loads(text[start:])


def _score(job: dict) -> dict:
    cmd = ["uv", "run", "--project", str(VC_QUALITY_DIR), "python",
           str(VC_QUALITY_SCRIPT), "one", "--converted", str(STUDY_DATA_DIR / job["converted"]),
           "--target", str(STUDY_DATA_DIR / job["target"]),
           "--source", str(STUDY_DATA_DIR / job["source"]),
           "--segment-mode", "fixed", "--segment-win", "2", "--segment-hop", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip()[-1000:] or f"vc_quality.py exited {proc.returncode}")
    result = _parse_stdout(proc.stdout)
    for key in ("converted", "target", "source"):
        result[f"{key}_path"] = job[key]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study_id", type=int)
    parser.add_argument("--participant")
    parser.add_argument("--session")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    backend = get_backend()
    sessions = backend.list_sessions(args.study_id)
    if args.participant:
        sessions = [s for s in sessions if s["participant_id"] == args.participant]
    if args.session:
        sessions = [s for s in sessions if s["session_id"] == args.session]
    sessions = [s for s in sessions if (s.get("files") or {}).get("participant")
                and (s.get("files") or {}).get("participant_raw")
                and (args.force or s.get("vc_quality_status") != "complete")]
    total = len(sessions)
    done = 0
    analysis_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    common = {"study_id": args.study_id, "participant_id": args.participant,
              "session_id": args.session, "analysis_id": analysis_id}
    _write(running=True, done=0, total=total, current=None, **common)

    for session in sessions:
        sid = session["session_id"]
        _write(running=True, done=done, total=total, current=sid, **common)
        backend.update_session_vc_quality(sid, "running", {"analysis_id": analysis_id})
        try:
            inputs = prepare_session_analysis(session, STUDY_DATA_DIR, analysis_id)
            scores = [{"region": job["region"], "metrics": _score(job)}
                      for job in inputs["score_jobs"]]
            result = {"status": "complete", "analysis_id": analysis_id,
                      "inputs": inputs, "scores": scores}
            out_dir = ((STUDY_DATA_DIR / (session.get("files") or {})["participant"]).parent /
                       "analysis" / "vc_quality" / analysis_id)
            atomic_write_json(out_dir / "results.json", result, exclusive=True)
            result["result_artifact"] = file_record(out_dir / "results.json", relative_to=STUDY_DATA_DIR)
            backend.update_session_vc_quality(sid, "complete", result)
        except Exception as exc:  # one failed session must not abort a study batch
            backend.update_session_vc_quality(sid, "failed", {
                "status": "failed", "analysis_id": analysis_id,
                "error": f"{type(exc).__name__}: {exc}"})
        done += 1
        _write(running=True, done=done, total=total, current=None, **common)
    _write(running=False, done=done, total=total, current=None, **common)


if __name__ == "__main__":
    main()
