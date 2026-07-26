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
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# metrics.py lives one level up (services/app_api/); make it importable.
_APP_API_DIR = str(Path(__file__).resolve().parents[1])
if _APP_API_DIR not in sys.path:
    sys.path.insert(0, _APP_API_DIR)

_REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_DATA_DIR = Path(os.environ.get("STUDY_DATA_DIR", str(_REPO_ROOT / "study_data")))


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


class AnalysisRunner:
    """Admin-triggered batch analysis for a study, run in a background thread so
    the request returns immediately. One run at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._state = {"running": False, "done": 0, "total": 0, "current": None, "study_id": None}

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def start(self, backend, study_id: int, force: bool) -> dict:
        with self._lock:
            if self._state["running"]:
                return dict(self._state)
            self._thread = threading.Thread(target=self._run, args=(backend, study_id, force), daemon=True)
            self._state = {"running": True, "done": 0, "total": 0, "current": None, "study_id": study_id}
            self._thread.start()
            return dict(self._state)

    def _run(self, backend, study_id: int, force: bool):
        sessions = backend.list_sessions(study_id)
        pending = [s for s in sessions
                   if (s.get("files") or {}).get("participant")
                   and (force or s.get("metrics") is None)]
        with self._lock:
            self._state["total"] = len(pending)
        for s in pending:
            with self._lock:
                self._state["current"] = s["session_id"]
            conv, raw, mt = _session_paths(s)
            try:
                run_session_analysis(s["session_id"], conv, raw, mt)
            except Exception as e:  # noqa: BLE001
                logger.error(f"[study] batch analysis error for {s['session_id']}: {e}")
            with self._lock:
                self._state["done"] += 1
        with self._lock:
            self._state["running"] = False
            self._state["current"] = None


_runner: Optional[AnalysisRunner] = None


def get_runner() -> AnalysisRunner:
    global _runner
    if _runner is None:
        _runner = AnalysisRunner()
    return _runner
