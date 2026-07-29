"""Worker for one-session, one-participant, or whole-study VC-quality runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from study.artifacts import atomic_write_json, file_record  # noqa: E402
from study.storage import get_backend  # noqa: E402
from study.transition_analysis import prepare_session_analysis  # noqa: E402
from study.vc_quality_analysis import STUDY_DATA_DIR, status_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
VC_QUALITY_DIR = Path(os.environ.get("VC_QUAL_DIR", REPO_ROOT / "services" / "vc_quality"))
VC_QUALITY_SCRIPT = VC_QUALITY_DIR / "vc_quality.py"
METRIC_PROFILE = "xvc_objective_v2"


def _write(**values) -> None:
    values["pid"] = os.getpid()
    values["heartbeat"] = time.time()
    path = status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(values))
    temp.replace(path)


def _score_batch(jobs: list[dict], heartbeat: Callable[[], None] | None = None
                 ) -> tuple[list[dict], dict]:
    if not jobs:
        return [], {"stdout": "", "stderr": ""}
    with tempfile.TemporaryDirectory(prefix="hmo_vcq_") as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.jsonl"
        output_path = Path(temp_dir) / "results.jsonl"
        with manifest_path.open("w", encoding="utf-8") as stream:
            for index, job in enumerate(jobs):
                row = {
                    "converted_path": str(STUDY_DATA_DIR / job["converted"]),
                    "target_path": str(STUDY_DATA_DIR / job["target"]),
                    "source_path": str(STUDY_DATA_DIR / job["source"]),
                    "_job_index": index,
                }
                stream.write(json.dumps(row) + "\n")
        cmd = ["uv", "run", "--project", str(VC_QUALITY_DIR), "python",
               str(VC_QUALITY_SCRIPT), "batch", "--manifest", str(manifest_path),
               "--out", str(output_path)]
        stdout_path = Path(temp_dir) / "stdout.log"
        stderr_path = Path(temp_dir) / "stderr.log"
        timeout_s = max(3600, 900 * len(jobs))
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True)
            deadline = time.monotonic() + timeout_s
            while proc.poll() is None:
                if heartbeat:
                    heartbeat()
                if time.monotonic() >= deadline:
                    proc.kill()
                    proc.wait()
                    raise TimeoutError(f"vc_quality.py exceeded {timeout_s} seconds")
                time.sleep(5)
        if proc.returncode:
            error = (stderr_path.read_text().strip()
                     or stdout_path.read_text().strip())
            raise RuntimeError(
                error[-1000:] or f"vc_quality.py exited {proc.returncode}")
        output_text = output_path.read_text()
        diagnostics = {
            "stdout": stdout_path.read_text()[-8000:],
            "stderr": stderr_path.read_text()[-8000:],
        }
    rows = [json.loads(line) for line in output_text.splitlines() if line.strip()]
    if len(rows) != len(jobs):
        raise RuntimeError(f"vc_quality.py returned {len(rows)} rows for {len(jobs)} jobs")
    ordered: list[dict | None] = [None] * len(jobs)
    for result in rows:
        index = int(result.pop("_job_index"))
        job = jobs[index]
        for key in ("converted", "target", "source"):
            result[f"{key}_path"] = job[key]
        ordered[index] = result
    if any(result is None for result in ordered):
        raise RuntimeError("vc_quality.py returned duplicate or missing job indices")
    return [result for result in ordered if result is not None], diagnostics


def _completion(scores: list[dict]) -> tuple[str, list[dict]]:
    """Require all three X-VC metrics; never call a soft failure complete."""
    unavailable = []
    for score in scores:
        for metric in ("wer", "sim", "utmos"):
            if not isinstance(score.get(metric), (int, float)):
                unavailable.append({
                    "region": score.get("_region"),
                    "metric": metric,
                    "error": score.get(f"{metric}_error"),
                })
    return ("complete" if not unavailable else "partial"), unavailable


def _needs_scoring(session: dict) -> bool:
    result = session.get("vc_quality") or {}
    return (session.get("vc_quality_status") != "complete"
            or not isinstance(result, dict)
            or result.get("metric_profile") != METRIC_PROFILE)


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
                and (args.force or _needs_scoring(s))]
    total = len(sessions)
    done = 0
    analysis_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    common = {"study_id": args.study_id, "participant_id": args.participant,
              "session_id": args.session, "analysis_id": analysis_id}
    _write(running=True, done=0, total=total, current=None, **common)

    prepared: list[dict] = []
    jobs: list[dict] = []
    for session in sessions:
        sid = session["session_id"]
        _write(running=True, done=done, total=total, current=sid, **common)
        backend.update_session_vc_quality(sid, "running", {"analysis_id": analysis_id})
        try:
            inputs = prepare_session_analysis(session, STUDY_DATA_DIR, analysis_id)
            start = len(jobs)
            jobs.extend(inputs["score_jobs"])
            prepared.append({"session": session, "inputs": inputs,
                             "job_start": start, "job_end": len(jobs)})
        except Exception as exc:  # one failed session must not abort a study batch
            backend.update_session_vc_quality(sid, "failed", {
                "status": "failed", "analysis_id": analysis_id,
                "error": f"{type(exc).__name__}: {exc}"})
            done += 1
            _write(running=True, done=done, total=total, current=None, **common)

    try:
        _write(running=True, done=done, total=total, current="batch", **common)
        metrics, scorer_log = _score_batch(
            jobs,
            heartbeat=lambda: _write(
                running=True, done=done, total=total, current="batch", **common),
        )
    except Exception as exc:
        for item in prepared:
            backend.update_session_vc_quality(item["session"]["session_id"], "failed", {
                "status": "failed", "analysis_id": analysis_id,
                "error": f"{type(exc).__name__}: {exc}"})
            done += 1
        _write(running=False, done=done, total=total, current=None, **common)
        return

    for item in prepared:
        session = item["session"]
        sid = session["session_id"]
        try:
            session_metrics = metrics[item["job_start"]:item["job_end"]]
            for job, score in zip(item["inputs"]["score_jobs"], session_metrics):
                score["_region"] = job["region"]
            scores = [
                {"region": job["region"],
                 "metrics": {key: value for key, value in score.items()
                             if key != "_region"}}
                for job, score in zip(item["inputs"]["score_jobs"], session_metrics)
            ]
            completion, unavailable = _completion(session_metrics)
            result_status = completion if scores else "not_applicable"
            storage_status = completion if scores else "complete"
            result = {"status": result_status, "analysis_id": analysis_id,
                      "metric_profile": METRIC_PROFILE,
                      "inputs": item["inputs"], "scores": scores,
                      "unavailable_metrics": unavailable,
                      "scorer_log": scorer_log}
            out_dir = ((STUDY_DATA_DIR / (session.get("files") or {})["participant"]).parent /
                       "analysis" / "vc_quality" / analysis_id)
            atomic_write_json(out_dir / "results.json", result, exclusive=True)
            result["result_artifact"] = file_record(
                out_dir / "results.json", relative_to=STUDY_DATA_DIR)
            backend.update_session_vc_quality(sid, storage_status, result)
        except Exception as exc:
            backend.update_session_vc_quality(sid, "failed", {
                "status": "failed", "analysis_id": analysis_id,
                "error": f"{type(exc).__name__}: {exc}"})
        done += 1
        _write(running=True, done=done, total=total, current=None, **common)
    _write(running=False, done=done, total=total, current=None, **common)


if __name__ == "__main__":
    main()
