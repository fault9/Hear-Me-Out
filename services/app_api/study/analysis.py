"""Per-session analysis (Whisper transcription + VC-quality metrics).

This is model inference and competes with the live PersonaPlex/VC on the same
box, so it is NOT run during the study. The researcher triggers it as a batch
from the admin dashboard AFTER data collection (`AnalysisRunner`), which walks
the study's saved sessions and writes transcript/metrics back to each.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# metrics.py lives one level up (services/app_api/); make it importable.
_APP_API_DIR = str(Path(__file__).resolve().parents[1])
if _APP_API_DIR not in sys.path:
    sys.path.insert(0, _APP_API_DIR)

_DATA_ROOT = os.path.expanduser(os.environ.get("STUDY_DATA_ROOT", "/workspace/data"))
STUDY_DATA_DIR = Path(os.path.expanduser(os.environ.get("STUDY_DATA_DIR", str(Path(_DATA_ROOT) / "media"))))


def run_session_analysis(session_id: str, converted_wav: str | None,
                         raw_wav: str | None, model_transcript: list | None) -> None:
    from .storage import get_backend

    backend = get_backend()
    transcript = {"model": model_transcript or [], "participant": None}
    metrics = None
    audiobox = False

    try:
        # Prefer comparing raw (original) vs converted; for a natural condition
        # there's no separate raw clip, so analyze the converted clip against
        # itself (metrics are trivial but the transcript is still extracted).
        clip_b = converted_wav if converted_wav and os.path.exists(converted_wav) else None
        clip_a = raw_wav if raw_wav and os.path.exists(raw_wav) else clip_b
        if clip_a and clip_b:
            from metrics import analyze_voices

            metrics = analyze_voices(clip_a, clip_b)
            transcript["participant"] = (metrics.get("response_b") or {}).get("transcript")
            audiobox = bool(metrics.get("audiobox_available"))
    except Exception as e:  # noqa: BLE001 - analysis is best-effort; audio is already saved
        logger.warning(f"[study] analysis failed for {session_id}: {e}")

    try:
        backend.update_session_analysis(session_id, transcript, metrics, audiobox)
        # Mirror JSON next to the WAVs so the ZIP export is self-contained.
        base = converted_wav or raw_wav
        if base:
            out_dir = Path(base).parent
            (out_dir / "transcript.json").write_text(json.dumps(transcript, indent=2))
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        logger.info(f"[study] analysis complete for {session_id} (audiobox={audiobox})")
    except Exception as e:  # noqa: BLE001
        logger.error(f"[study] could not persist analysis for {session_id}: {e}")


def _session_paths(session: dict):
    files = session.get("files") or {}
    conv = files.get("participant")
    raw = files.get("participant_raw")
    conv_p = str(STUDY_DATA_DIR / conv) if conv else None
    raw_p = str(STUDY_DATA_DIR / raw) if raw else None
    tr = session.get("transcript")
    model_transcript = tr.get("model") if isinstance(tr, dict) else None
    return conv_p, raw_p, model_transcript


def status_path() -> Path:
    return STUDY_DATA_DIR / "_analysis_status.json"


_IDLE = {"running": False, "done": 0, "total": 0, "current": None, "study_id": None}
_STATUS_KEYS = ("running", "done", "total", "current", "study_id")


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (OSError, ValueError, TypeError):
        return True  # PermissionError etc. -> exists but not ours; treat as alive


class AnalysisRunner:
    """Admin-triggered batch analysis for a study.

    The heavy work runs in a SEPARATE PROCESS (`study.analysis_worker`) so the
    CPU-bound metrics stack never blocks the API's asyncio event loop (which made
    the dashboard and page reloads hang). Progress is read from a JSON status file
    the worker writes; this class only launches the worker and reports status."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None

    def _read_status(self) -> dict:
        try:
            st = json.loads(status_path().read_text())
        except (OSError, ValueError):
            return dict(_IDLE)
        if st.get("running"):
            # A stale 'running' (worker crashed, or the API restarted mid-run)
            # must not stick forever: verify the worker pid and heartbeat.
            pid = st.get("pid")
            alive = _pid_alive(pid) if pid else (self._proc is not None and self._proc.poll() is None)
            fresh = (time.time() - (st.get("heartbeat") or 0)) < 300
            if not (alive and fresh):
                st = {**st, "running": False}
        return st

    def get_status(self) -> dict:
        st = self._read_status()
        return {k: st.get(k) for k in _STATUS_KEYS}

    def start(self, backend, study_id: int, force: bool) -> dict:
        if self._read_status().get("running"):
            return self.get_status()

        app_api_dir = str(Path(__file__).resolve().parents[1])
        args = [sys.executable, "-m", "study.analysis_worker", str(study_id)]
        if force:
            args.append("--force")

        # Seed the status file so the UI shows 'running' before the worker has
        # finished importing its (slow) model stack.
        try:
            p = status_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"running": True, "done": 0, "total": 0, "current": None,
                                     "study_id": study_id, "pid": None, "heartbeat": time.time()}))
        except OSError as e:  # noqa: BLE001
            logger.warning(f"[study] could not seed analysis status: {e}")

        self._proc = subprocess.Popen(args, cwd=app_api_dir)
        logger.info(f"[study] launched analysis worker pid={self._proc.pid} study={study_id} force={force}")
        return {"running": True, "done": 0, "total": 0, "current": None, "study_id": study_id}


_runner: Optional[AnalysisRunner] = None


def get_runner() -> AnalysisRunner:
    global _runner
    if _runner is None:
        _runner = AnalysisRunner()
    return _runner
