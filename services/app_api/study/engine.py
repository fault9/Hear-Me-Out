"""VC-engine lifecycle manager (v2 — per-scenario engine).

A scenario's voice schedule picks a VC engine (MeanVC or X-VC). Only one engine
runs on :5002, so `ensure_engine` restarts it when a scenario needs a different
one, then loads that scenario's targets. Natural-only scenarios still go through
the proxy (to hide the prompt), so any engine that's already up is reused.
Progress is exposed as a step list for the participant's "preparing" screen.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REPO_ROOT = Path(__file__).resolve().parents[3]
VC_ENGINE_SCRIPT = REPO_ROOT / "infra" / "vc_engine.sh"

VC_HOST = os.environ.get("STUDY_VC_HOST", "127.0.0.1")
VC_PORT = int(os.environ.get("MEANVC_PORT", "5002"))
PPLX_PORT = int(os.environ.get("PERSONAPLEX_PROXY_PORT", os.environ.get("PERSONAPLEX_PORT", "8000")))
ENGINE_START_TIMEOUT = int(os.environ.get("STUDY_ENGINE_START_TIMEOUT", "300"))
DEFAULT_ENGINE = os.environ.get("VC_ENGINE", "meanvc")


@dataclass
class PrepareState:
    status: str = "idle"          # idle | preparing | ready | error
    steps: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    version: int = 0

    def to_dict(self) -> dict:
        return {"status": self.status, "steps": self.steps, "error": self.error, "version": self.version}


def _port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class VCEngineManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = PrepareState()
        self._state_lock = threading.Lock()
        self._current_engine: Optional[str] = None   # engine we started on :5002
        self._loaded_engine: Optional[str] = None     # engine whose targets are loaded
        self._force_restart: bool = False             # restart on next ensure (fresh GPU context)
        self._thread: Optional[threading.Thread] = None

    # ---- progress state ----
    def get_state(self) -> dict:
        with self._state_lock:
            return self._state.to_dict()

    def _set_status(self, status: str, error: Optional[str] = None):
        with self._state_lock:
            self._state.status = status
            self._state.error = error
            self._state.version += 1

    def _reset_steps(self):
        with self._state_lock:
            self._state.steps = []
            self._state.error = None
            self._state.status = "preparing"
            self._state.version += 1

    def _step(self, label: str) -> int:
        with self._state_lock:
            self._state.steps.append({"label": label, "state": "running"})
            self._state.version += 1
            return len(self._state.steps) - 1

    def _finish_step(self, idx: int, state: str = "done"):
        with self._state_lock:
            if 0 <= idx < len(self._state.steps):
                self._state.steps[idx]["state"] = state
                self._state.version += 1

    # ---- engine control ----
    def is_engine_up(self) -> bool:
        return _port_open(VC_HOST, VC_PORT)

    def _run_script(self, action: str, engine: Optional[str] = None) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        if engine:
            env["VC_ENGINE"] = engine
        return subprocess.run(["bash", str(VC_ENGINE_SCRIPT), action],
                              capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)

    def stop_engine(self) -> None:
        if VC_ENGINE_SCRIPT.exists():
            self._run_script("stop")
        self._current_engine = None
        self._loaded_engine = None
        self._set_status("idle")

    def _restart_to(self, engine: str):
        if not VC_ENGINE_SCRIPT.exists():
            raise RuntimeError(f"VC engine launcher missing: {VC_ENGINE_SCRIPT}")
        self._run_script("stop", engine)
        deadline = time.time() + 30
        while time.time() < deadline and self.is_engine_up():
            time.sleep(1)
        self._run_script("start", engine)
        deadline = time.time() + ENGINE_START_TIMEOUT
        while time.time() < deadline:
            if self.is_engine_up():
                self._current_engine = engine
                self._loaded_engine = None
                return
            time.sleep(2)
        raise RuntimeError(f"VC engine ({engine}) did not become reachable in time")

    def _load_targets(self, backend, targets):
        for t in targets:
            wav_path = t["wav_path"]
            if not os.path.exists(wav_path):
                raise RuntimeError(f"Target WAV missing: {wav_path}")
            with open(wav_path, "rb") as f:
                resp = requests.post(f"https://{VC_HOST}:{VC_PORT}/api/meanvc/load-target",
                                     files={"wav": (os.path.basename(wav_path), f, "audio/wav")},
                                     verify=False, timeout=120)
            resp.raise_for_status()
            engine_target_id = resp.json().get("target_id")
            if not engine_target_id:
                raise RuntimeError(f"load-target returned no target_id for {t['ref']}")
            backend.set_engine_target_id(t["id"], engine_target_id)

    def invalidate(self) -> None:
        """Force the next `ensure_engine` to restart the engine, even if the right
        one is already up. Used at run start so each participant gets a FRESH engine
        (clean CUDA context): X-VC (the GPU engine) degrades across long/repeated
        sessions and a port-only health check can't detect it, which silently
        produced dead-VC sessions for a second participant. Opt out with
        STUDY_REUSE_ENGINE=1."""
        with self._lock:
            self._force_restart = True

    def ensure_engine(self, backend, study_id: int, requested_engine: Optional[str]) -> None:
        """Blocking. Bring up the engine this scenario needs (restarting :5002 if it
        differs) and load its targets. A natural-only scenario passes None and
        reuses whatever engine is up (default if none)."""
        with self._lock:
            force = self._force_restart
            self._force_restart = False  # consume: exactly one fresh restart per run
            target_engine = requested_engine or (self._current_engine if self.is_engine_up() else DEFAULT_ENGINE)
            targets = [t for t in backend.list_targets(study_id) if t["engine"] == target_engine]
            already = (not force and self.is_engine_up() and self._current_engine == target_engine
                       and self._loaded_engine == target_engine
                       and all(t.get("engine_target_id") for t in targets))
            if already:
                self._set_status("ready")
                return

            self._reset_steps()
            try:
                if force or not (self.is_engine_up() and self._current_engine == target_engine):
                    s = self._step(f"Starting {target_engine} voice engine")
                    self._restart_to(target_engine)
                    self._finish_step(s)

                s = self._step(f"Loading {len(targets)} target voice(s)")
                backend.clear_engine_target_ids(study_id, target_engine)
                self._load_targets(backend, targets)
                self._loaded_engine = target_engine
                self._finish_step(s)

                s = self._step("Checking speech model")
                if not _port_open(VC_HOST, PPLX_PORT):
                    self._finish_step(s, "error")
                    raise RuntimeError(f"PersonaPlex not reachable on port {PPLX_PORT}")
                self._finish_step(s)

                self._set_status("ready")
            except Exception as e:  # noqa: BLE001
                self._set_status("error", str(e))
                raise

    def start_prepare_async(self, backend, study_id: int, requested_engine: Optional[str]) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._set_status("preparing")

        def _run():
            try:
                self.ensure_engine(backend, study_id, requested_engine)
            except Exception:  # noqa: BLE001
                pass

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()


_manager: Optional[VCEngineManager] = None


def get_manager() -> VCEngineManager:
    global _manager
    if _manager is None:
        _manager = VCEngineManager()
    return _manager
