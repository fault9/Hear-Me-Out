"""Launch route-aware vc_quality.py analysis outside the live API process."""

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
_DATA_ROOT = Path(os.path.expanduser(os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))
STUDY_DATA_DIR = Path(os.path.expanduser(os.environ.get(
    "STUDY_DATA_DIR", str(_DATA_ROOT / "media"))))


def status_path() -> Path:
    return STUDY_DATA_DIR / "_vc_quality_status.json"


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (OSError, ValueError, TypeError):
        return True


class VCQualityRunner:
    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None

    def get_status(self) -> dict:
        try:
            status = json.loads(status_path().read_text())
        except (OSError, ValueError):
            return {"running": False, "done": 0, "total": 0, "current": None}
        if status.get("running"):
            fresh = time.time() - float(status.get("heartbeat") or 0) < 300
            if not (fresh and _pid_alive(status.get("pid"))):
                status["running"] = False
        return status

    def start(self, study_id: int, *, participant_id: str | None = None,
              session_id: str | None = None, force: bool = False) -> dict:
        if self.get_status().get("running"):
            return self.get_status()
        args = [sys.executable, "-m", "study.vc_quality_worker", str(study_id)]
        if participant_id:
            args.extend(["--participant", participant_id])
        if session_id:
            args.extend(["--session", session_id])
        if force:
            args.append("--force")
        seed = {"running": True, "done": 0, "total": 0, "current": None,
                "study_id": study_id, "participant_id": participant_id,
                "session_id": session_id, "heartbeat": time.time(), "pid": None}
        status_path().parent.mkdir(parents=True, exist_ok=True)
        status_path().write_text(json.dumps(seed))
        self._proc = subprocess.Popen(args, cwd=Path(__file__).resolve().parents[1])
        return seed


_runner: Optional[VCQualityRunner] = None


def get_vc_quality_runner() -> VCQualityRunner:
    global _runner
    if _runner is None:
        _runner = VCQualityRunner()
    return _runner
