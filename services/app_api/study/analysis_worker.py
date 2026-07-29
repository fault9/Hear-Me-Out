"""Subprocess entrypoint for batch session analysis.

Run as:  python -m study.analysis_worker <study_id> [--force]
(cwd = services/app_api, under the app-api venv).

Kept OUT of the API process on purpose: the metrics stack (Whisper, audiobox,
sentence-transformers, librosa.pyin) is CPU-bound and holds the GIL for long
stretches. Running it in a thread inside uvicorn starves the asyncio event loop,
so `/analyze/status` polls and even plain page reloads hang while a batch runs.
A separate process has its own GIL and its own CPU budget, so the API stays
responsive. Progress is reported via a small JSON status file that the API's
`AnalysisRunner` reads.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Make `study` + `metrics` (parents[1] = services/app_api) and the shared `common`
# package (parents[2] = services/) importable when launched as `-m study.analysis_worker`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Leave CPU headroom for the live path (PersonaPlex/VC proxy) if this ever runs
# alongside a session; the batch is deliberately deprioritised, not maximal.
try:
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
except Exception:  # noqa: BLE001
    pass

from study.analysis import (STUDY_DATA_DIR, _session_paths, run_session_analysis,
                            status_path)  # noqa: E402
from study.storage import get_backend  # noqa: E402
from study.timing_analysis import prepare_timing_analysis  # noqa: E402
from study.vc_quality_analysis import status_path as vc_quality_status_path  # noqa: E402

# Shared OTel helper (services/ is on sys.path via the insert above). No-op unless
# OTEL_* is configured; the worker exports to the same collector as app-api.
try:
    from common import otel  # noqa: E402
    from common import logging_setup  # noqa: E402
    otel.init_tracing("study-analysis")
    logging_setup.init_logging("study-analysis")
    _tracer = otel.get_tracer("study-analysis")
except Exception:  # noqa: BLE001
    otel = None
    logging_setup = None
    _tracer = None


def _write(**kw) -> None:
    p = status_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    kw["pid"] = os.getpid()
    kw["heartbeat"] = time.time()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(kw))
    tmp.replace(p)  # atomic-ish: readers never see a half-written file


def _read_vc_quality_status() -> dict:
    try:
        return json.loads(vc_quality_status_path().read_text())
    except (OSError, ValueError):
        return {"running": True, "done": 0, "total": 0, "current": None}


def _run_vc_quality(study_id: int, force: bool) -> str | None:
    args = [sys.executable, "-m", "study.vc_quality_worker", str(study_id)]
    if force:
        args.append("--force")
    proc = subprocess.Popen(args, cwd=Path(__file__).resolve().parents[1])
    while proc.poll() is None:
        status = _read_vc_quality_status()
        _write(running=True, phase="vc_quality", done=status.get("done", 0),
               total=status.get("total", 0), current=status.get("current"),
               study_id=study_id, error=None)
        time.sleep(2)
    if proc.returncode:
        return f"VC-quality worker exited with status {proc.returncode}"
    return None


def main() -> None:
    study_id = int(sys.argv[1])
    force = "--force" in sys.argv[2:]

    backend = get_backend()
    sessions = backend.list_sessions(study_id)
    def needs_timing(session: dict) -> bool:
        analysis = (session.get("artifact_manifest") or {}).get("analysis") or {}
        return not analysis.get("timing_latest")

    pending = [s for s in sessions
               if (s.get("files") or {}).get("participant")
               and (force or s.get("metrics") is None or needs_timing(s))]
    total = len(pending)
    done = 0
    _write(running=True, phase="preprocessing", done=0, total=total,
           current=None, study_id=study_id, error=None)

    for s in pending:
        _write(running=True, phase="preprocessing", done=done, total=total,
               current=s["session_id"], study_id=study_id, error=None)
        if logging_setup:
            logging_setup.set_log_session(s["session_id"], study_id)
        conv, raw, mt = _session_paths(s)
        span_cm = (otel.start_span(_tracer, "analysis.session",
                                   attributes={"study.session_id": s["session_id"],
                                               "study.study_id": study_id})
                   if otel else contextlib.nullcontext())
        try:
            with span_cm:
                if force or s.get("metrics") is None:
                    run_session_analysis(s["session_id"], conv, raw, mt)
                if force or needs_timing(s):
                    analysis_id = (
                        f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}."
                        f"{time.time_ns() % 1_000_000_000:09d}Z"
                    )
                    timing = prepare_timing_analysis(s, STUDY_DATA_DIR, analysis_id)
                    latest = backend.get_session(s["session_id"]) or s
                    manifest = copy.deepcopy(latest.get("artifact_manifest") or {})
                    manifest.setdefault("analysis", {})["timing_latest"] = timing[
                        "result_artifact"]
                    backend.update_session_artifacts(s["session_id"], manifest)
        except Exception as e:  # noqa: BLE001 - one bad session shouldn't kill the batch
            print(f"[analysis_worker] error for {s['session_id']}: {e}", file=sys.stderr)
        done += 1
        _write(running=True, phase="preprocessing", done=done, total=total,
               current=None, study_id=study_id, error=None)

    error = _run_vc_quality(study_id, force)
    final_status = _read_vc_quality_status()
    _write(running=False, phase="complete" if error is None else "failed",
           done=final_status.get("done", 0), total=final_status.get("total", 0),
           current=None, study_id=study_id, error=error)


if __name__ == "__main__":
    main()
