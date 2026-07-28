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
import json
import os
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

from study.analysis import _session_paths, run_session_analysis, status_path  # noqa: E402
from study.storage import get_backend  # noqa: E402

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


def main() -> None:
    study_id = int(sys.argv[1])
    force = "--force" in sys.argv[2:]

    backend = get_backend()
    sessions = backend.list_sessions(study_id)
    pending = [s for s in sessions
               if (s.get("files") or {}).get("participant")
               and (force or s.get("metrics") is None)]
    total = len(pending)
    done = 0
    _write(running=True, done=0, total=total, current=None, study_id=study_id)

    for s in pending:
        _write(running=True, done=done, total=total, current=s["session_id"], study_id=study_id)
        if logging_setup:
            logging_setup.set_log_session(s["session_id"], study_id)
        conv, raw, mt = _session_paths(s)
        span_cm = (otel.start_span(_tracer, "analysis.session",
                                   attributes={"study.session_id": s["session_id"],
                                               "study.study_id": study_id})
                   if otel else contextlib.nullcontext())
        try:
            with span_cm:
                run_session_analysis(s["session_id"], conv, raw, mt)
        except Exception as e:  # noqa: BLE001 - one bad session shouldn't kill the batch
            print(f"[analysis_worker] error for {s['session_id']}: {e}", file=sys.stderr)
        done += 1
        _write(running=True, done=done, total=total, current=None, study_id=study_id)

    _write(running=False, done=done, total=total, current=None, study_id=study_id)


if __name__ == "__main__":
    main()
