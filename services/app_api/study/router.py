"""FastAPI router for the study platform (v2 — multi-study).

Admin endpoints (token-gated) manage many studies, their scenarios (with timed
voice schedules), engine-tagged targets, and questionnaires. Participant
endpoints run the resumable, time-limited flow; the system prompt + voice
schedule never reach the browser (the VC engine resolves them via
GET /condition/{session_id}). The engine for each scenario is prepared on demand,
restarting :5002 when a scenario needs a different one.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import (APIRouter, Depends, File, Form, Header, HTTPException,
                     UploadFile)
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .analysis import get_runner
from .engine import get_manager
from . import yaml_io
from .models import (REQUIRED_CARD_FIELDS, CreateStudyRequest, EnterRequest,
                     GenerateRequest, ProgressRequest, QuestionnaireRequest,
                     RunStartRequest, Scenario, SessionStartRequest,
                     SubmitRequest, UpdateStudyRequest, default_questionnaires)
from .storage import get_backend

try:
    from common import otel  # shared OpenTelemetry helper (services/ is on sys.path)
    from common import logging_setup
except Exception:  # noqa: BLE001 - tracing/structured logging are optional
    otel = None
    logging_setup = None

logger = logging.getLogger(__name__)
_tracer = otel.get_tracer("study-app-api") if otel else None


def _trace_session(**attrs):
    """Stamp span attributes AND tag this request's logs with session id + study id.
    Also record the active session so device-level GPU metrics are attributed to it."""
    if otel:
        otel.set_session_attributes(**attrs)
        if attrs.get("session_id"):
            otel.set_active_session(attrs["session_id"], attrs.get("study_id"))
    if logging_setup and attrs.get("session_id"):
        logging_setup.set_log_session(attrs["session_id"], attrs.get("study_id"))

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = Path(os.environ.get("WORKSPACE", str(REPO_ROOT.parent)))
_DATA_ROOT = os.path.expanduser(os.environ.get("STUDY_DATA_ROOT", "/workspace/data"))
STUDY_DATA_DIR = Path(os.path.expanduser(os.environ.get("STUDY_DATA_DIR", str(Path(_DATA_ROOT) / "media"))))
TARGETS_DIR = STUDY_DATA_DIR / "targets"
SESSIONS_DIR = STUDY_DATA_DIR / "sessions"
# Ensure the media root exists as soon as app-api boots (not just lazily on first
# upload), so the mounted volume is populated even before any study data is saved.
try:
    STUDY_DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

ADMIN_TOKEN = os.environ.get("STUDY_ADMIN_TOKEN") or "changeme-study-admin"
if ADMIN_TOKEN == "changeme-study-admin":
    logger.warning("STUDY_ADMIN_TOKEN is not set — using an insecure default. Set it in production.")


def require_admin(x_study_admin_token: str = Header(default="")):
    if x_study_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")


# ---------- helpers ----------
def _scenario_engine(scenario: dict) -> Optional[str]:
    """The VC engine a scenario needs (its vc segments' engine), or None if natural-only."""
    for seg in scenario.get("voice_schedule") or []:
        if seg.get("mode") == "vc" and seg.get("engine"):
            return seg["engine"]
    return None


def _schedule_label(scenario: dict) -> str:
    segs = scenario.get("voice_schedule") or []
    if not segs:
        return "natural"
    modes = [s.get("mode", "natural") for s in segs]
    eng = _scenario_engine(scenario)
    if len(set(modes)) == 1:
        return modes[0] + (f":{eng}" if modes[0] == "vc" else "")
    sw = segs[0].get("end_s")
    return f"{modes[0]}->{modes[1]}@{int(sw) if sw else '?'}" + (f":{eng}" if eng else "")


def _test_scenario(backend, study_id: int) -> Optional[dict]:
    """The study's practice/test scenario (is_test), if any. At most one."""
    for sc in backend.list_scenarios(study_id):
        if sc.get("is_test"):
            return sc
    return None


def _participant_order(backend, participant: dict) -> list:
    """The participant's assigned scenario IDs in order, EXCLUDING the practice scenario.
    Real scenarios keep 1-based positions over this list, so the test scenario never
    shifts them (and playback/session ids stay valid)."""
    order = participant.get("scenario_order") or []
    test = _test_scenario(backend, participant["study_id"])
    tid = test.get("id") if test else None
    return [sid for sid in order if sid != tid]


def _resolve_scenario(backend, participant: dict, scenario_order: int) -> dict:
    # scenario_order 0 => the practice/test scenario (always runs first).
    if scenario_order == 0:
        test = _test_scenario(backend, participant["study_id"])
        if not test:
            raise HTTPException(status_code=400, detail="no test scenario for this study")
        return test
    order = _participant_order(backend, participant)
    if scenario_order < 1 or scenario_order > len(order):
        raise HTTPException(status_code=400, detail="scenario_order out of range")
    scenario = backend.get_scenario(order[scenario_order - 1])
    if not scenario:
        raise HTTPException(status_code=400, detail="scenario not found")
    return scenario


def _scenario_card(scenario: dict, scenario_order: int) -> dict:
    card = scenario.get("scenario_card") or {}
    out = {
        "scenario_order": scenario_order,
        "scenario_id": scenario.get("id"),
        "title": scenario.get("title", ""),
        "is_test": bool(scenario.get("is_test")),
        "extra_fields": [f for f in (card.get("extra_fields") or []) if f.get("label")],
        "post_items": scenario.get("post_items") or [],   # scenario-specific post questions
        "time_limit_s": scenario.get("time_limit_s", 300),
    }
    for key, _label in REQUIRED_CARD_FIELDS:
        out[key] = card.get(key, "")
    return out


def _is_vc_to_natural(scenario: dict) -> bool:
    segs = scenario.get("voice_schedule") or []
    return len(segs) >= 2 and segs[0].get("mode") == "vc" and segs[-1].get("mode") == "natural"


def _validate_scenario(body: Scenario):
    """All participant-facing card fields, the title, and the system prompt are
    required (a blank system prompt also crashes PersonaPlex)."""
    missing = []
    if not (body.title or "").strip():
        missing.append("Title")
    if not (body.system_prompt or "").strip():
        missing.append("System prompt")
    for key, label in REQUIRED_CARD_FIELDS:
        if not (getattr(body.scenario_card, key, "") or "").strip():
            missing.append(label)
    if missing:
        raise HTTPException(status_code=422, detail="Please fill in: " + ", ".join(missing))


def _run_public(run: Optional[dict]) -> dict:
    if not run:
        return {"status": "not_started"}
    return {"status": run["status"], "current_step": run.get("current_step") or {},
            "completed": run.get("completed") or {}, "remaining_seconds": run.get("remaining_seconds", 0),
            "attempt": run.get("attempt", 1)}


def _list_voices() -> list[str]:
    d = os.environ.get("PERSONAPLEX_VOICES_DIR", "")
    if d and os.path.isdir(d):
        return sorted(f.name for f in Path(d).glob("*.pt"))
    return ["NATF2.pt"]


def _list_engines() -> list[str]:
    engines = ["meanvc"]
    if (WORKSPACE / "X-VC").exists():
        engines.append("xvc")
    return engines


def build_study_router() -> APIRouter:
    router = APIRouter(prefix="/api/study")
    backend = get_backend()
    manager = get_manager()
    TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def _study_detail(study: dict) -> dict:
        return {**study,
                "scenarios": backend.list_scenarios(study["id"]),
                "targets": backend.list_targets(study["id"]),
                "participants": backend.list_participants(study["id"])}

    # =============================== ADMIN ===============================
    @router.get("/voices", dependencies=[Depends(require_admin)])
    async def voices():
        return {"voices": _list_voices()}

    @router.get("/engines", dependencies=[Depends(require_admin)])
    async def engines():
        return {"engines": _list_engines()}

    @router.post("/stop-engine", dependencies=[Depends(require_admin)])
    async def stop_engine():
        manager.stop_engine()
        return {"engine": "stopped"}

    @router.get("/template", dependencies=[Depends(require_admin)])
    async def template():
        if yaml_io.TEMPLATE_PATH.exists():
            return FileResponse(str(yaml_io.TEMPLATE_PATH), media_type="application/x-yaml",
                                filename="pilot_study.yaml")
        raise HTTPException(status_code=404, detail="Template not found")

    @router.get("/studies", dependencies=[Depends(require_admin)])
    async def list_studies():
        return {"studies": backend.list_studies()}

    @router.get("/studies/{study_id}/yaml", dependencies=[Depends(require_admin)])
    async def export_yaml(study_id: int):
        if not backend.get_study(study_id):
            raise HTTPException(status_code=404, detail="Unknown study")
        text = yaml_io.dump_yaml(yaml_io.study_to_dict(backend, study_id))
        return Response(text, media_type="application/x-yaml",
                        headers={"Content-Disposition": f"attachment; filename=study{study_id}.yaml"})

    @router.post("/studies/{study_id}/import", dependencies=[Depends(require_admin)])
    async def import_yaml(study_id: int, file: UploadFile = File(...)):
        if not backend.get_study(study_id):
            raise HTTPException(status_code=404, detail="Unknown study")
        data = yaml_io.parse_yaml(await file.read())
        yaml_io.apply_import(backend, study_id, data)
        return {"study": _study_detail(backend.get_study(study_id))}

    @router.post("/studies", dependencies=[Depends(require_admin)])
    async def create_study(body: CreateStudyRequest):
        study = backend.create_study(body.name, body.description)
        backend.update_study(study["id"], None, None, default_questionnaires())
        return {"study": _study_detail(backend.get_study(study["id"]))}

    @router.get("/studies/{study_id}", dependencies=[Depends(require_admin)])
    async def get_study(study_id: int):
        study = backend.get_study(study_id)
        if not study:
            raise HTTPException(status_code=404, detail="Unknown study")
        return {"study": _study_detail(study)}

    @router.put("/studies/{study_id}", dependencies=[Depends(require_admin)])
    async def update_study(study_id: int, body: UpdateStudyRequest):
        study = backend.update_study(study_id, body.name, body.description, None, body.settings)
        return {"study": _study_detail(study)}

    @router.delete("/studies/{study_id}", dependencies=[Depends(require_admin)])
    async def archive_study(study_id: int):
        backend.archive_study(study_id, True)
        return {"ok": True}

    @router.put("/studies/{study_id}/questionnaires", dependencies=[Depends(require_admin)])
    async def set_questionnaires(study_id: int, body: dict):
        backend.update_study(study_id, None, None, body.get("questionnaires", body))
        return {"study": _study_detail(backend.get_study(study_id))}

    @router.post("/studies/{study_id}/scenarios", dependencies=[Depends(require_admin)])
    async def add_scenario(study_id: int, body: Scenario):
        return {"scenario": backend.add_scenario(study_id, body.model_dump())}

    @router.put("/studies/{study_id}/scenarios/{scenario_id}", dependencies=[Depends(require_admin)])
    async def update_scenario(study_id: int, scenario_id: int, body: Scenario):
        _validate_scenario(body)
        return {"scenario": backend.update_scenario(scenario_id, body.model_dump())}

    @router.delete("/studies/{study_id}/scenarios/{scenario_id}", dependencies=[Depends(require_admin)])
    async def delete_scenario(study_id: int, scenario_id: int):
        backend.delete_scenario(scenario_id)
        return {"ok": True}

    @router.put("/studies/{study_id}/scenarios/{scenario_id}/post-items", dependencies=[Depends(require_admin)])
    async def set_scenario_post_items(study_id: int, scenario_id: int, body: dict):
        """Set only a scenario's post questions (no card validation) — used by
        'copy to all scenarios'."""
        sc = backend.get_scenario(scenario_id)
        if not sc:
            raise HTTPException(status_code=404, detail="Unknown scenario")
        backend.update_scenario(scenario_id, {**sc, "post_items": body.get("post_items", [])})
        return {"ok": True}

    @router.post("/studies/{study_id}/targets", dependencies=[Depends(require_admin)])
    async def upload_target(study_id: int, wav: UploadFile = File(...), ref: str = Form(...),
                            speaker_id: str = Form(""), label: str = Form(""), engine: str = Form("meanvc")):
        d = TARGETS_DIR / f"study{study_id}"
        d.mkdir(parents=True, exist_ok=True)
        dest = d / f"{ref}.wav"
        with open(dest, "wb") as f:
            shutil.copyfileobj(wav.file, f)
        t = backend.add_target(study_id, ref, speaker_id or ref, label or wav.filename or ref, str(dest), engine)
        return {"target": t}

    @router.delete("/studies/{study_id}/targets/{target_id}", dependencies=[Depends(require_admin)])
    async def delete_target(study_id: int, target_id: int):
        backend.delete_target(target_id)
        return {"ok": True}

    @router.post("/studies/{study_id}/participants/generate", dependencies=[Depends(require_admin)])
    async def gen_participants(study_id: int, body: GenerateRequest):
        scenario_ids = [s["id"] for s in backend.list_scenarios(study_id)]
        if not scenario_ids:
            raise HTTPException(status_code=400, detail="Add at least one scenario first")
        created = backend.generate_participants(study_id, max(1, body.count), scenario_ids)
        return {"participants": created}

    @router.get("/studies/{study_id}/runs", dependencies=[Depends(require_admin)])
    async def list_runs(study_id: int):
        return {"runs": backend.list_runs(study_id)}

    @router.get("/studies/{study_id}/sessions", dependencies=[Depends(require_admin)])
    async def list_sessions(study_id: int):
        return {"sessions": backend.list_sessions(study_id)}

    @router.post("/studies/{study_id}/analyze", dependencies=[Depends(require_admin)])
    async def analyze(study_id: int, force: bool = False):
        """Run Whisper transcription + VC-quality metrics over the study's saved
        sessions (batch, background). Deferred here so it doesn't compete with the
        live study. force=true re-analyzes already-processed sessions."""
        if not backend.get_study(study_id):
            raise HTTPException(status_code=404, detail="Unknown study")
        return get_runner().start(backend, study_id, force)

    @router.get("/studies/{study_id}/analyze/status", dependencies=[Depends(require_admin)])
    async def analyze_status(study_id: int):
        return get_runner().get_status()

    @router.get("/studies/{study_id}/export", dependencies=[Depends(require_admin)])
    async def export(study_id: int, format: str = "json"):
        study = backend.get_study(study_id)
        data = {
            "study": study,
            "scenarios": backend.list_scenarios(study_id),
            "targets": backend.list_targets(study_id),
            "participants": backend.list_participants(study_id),
            "runs": backend.list_runs(study_id),
            "sessions": backend.list_sessions(study_id),
            "answers": backend.list_answers(study_id),
        }
        if format == "json":
            return JSONResponse(data)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("study_export.json", json.dumps(data, indent=2))
            if SESSIONS_DIR.exists():
                for p in SESSIONS_DIR.rglob("*"):
                    if p.is_file():
                        z.write(p, str(p.relative_to(STUDY_DATA_DIR)))
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/zip",
                                 headers={"Content-Disposition": f"attachment; filename=study{study_id}_export.zip"})

    # ============================ PARTICIPANT ============================
    def _require_participant(code: str) -> dict:
        p = backend.get_participant_by_code(code)
        if not p:
            raise HTTPException(status_code=404, detail="Invalid code")
        return p

    @router.post("/enter")
    async def enter(body: EnterRequest):
        p = _require_participant(body.code)
        study = backend.get_study(p["study_id"])
        if not study:
            raise HTTPException(status_code=404, detail="Study not found")
        order = _participant_order(backend, p)   # real scenarios, practice excluded
        scenarios = []
        for i, sid in enumerate(order):
            sc = backend.get_scenario(sid)
            if sc:
                scenarios.append(_scenario_card(sc, i + 1))
        test = _test_scenario(backend, p["study_id"])   # practice scenario, runs first
        test_card = _scenario_card(test, 0) if test else None
        run = backend.get_latest_run(p["participant_id"])
        settings = study.get("settings") or {}
        return {"participant_id": p["participant_id"], "study_name": study["name"],
                "scenarios": scenarios, "test_scenario": test_card,
                "questionnaires": study.get("questionnaires") or {},
                "welcome_text": settings.get("welcome_text", ""),
                "estimated_duration": settings.get("estimated_duration", ""),
                "run": _run_public(run)}

    @router.post("/run/start")
    async def run_start(body: RunStartRequest):
        p = _require_participant(body.code)
        live = backend.get_live_run(p["study_id"])
        if live and live["participant_id"] != p["participant_id"]:
            raise HTTPException(status_code=409,
                                detail="Another session is in progress, please try again shortly.")
        run = backend.start_run(p["participant_id"], body.mode)
        # Optional: give each participant a fresh VC engine (clean CUDA context). Off
        # by default — a full run reuses one engine fine. Enable if you observe X-VC
        # degrading across participants: STUDY_FRESH_ENGINE_PER_RUN=1.
        if os.environ.get("STUDY_FRESH_ENGINE_PER_RUN", "").lower() in ("1", "true", "yes"):
            manager.invalidate()
        return {"run": _run_public(run)}

    @router.get("/run/prepare/status")
    async def prepare_status():
        return manager.get_state()

    @router.get("/run/prepare/stream")
    async def prepare_stream():
        async def gen():
            last = -1
            while True:
                state = manager.get_state()
                if state["version"] != last:
                    last = state["version"]
                    yield f"data: {json.dumps(state)}\n\n"
                if state["status"] in ("ready", "error", "idle"):
                    break
                await asyncio.sleep(0.4)
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.post("/run/progress")
    async def run_progress(body: ProgressRequest):
        p = _require_participant(body.code)
        run = backend.get_latest_run(p["participant_id"])
        if not run:
            raise HTTPException(status_code=400, detail="No active run")
        backend.update_run_progress(run["id"], body.current_step, body.completed)
        return {"ok": True}

    @router.post("/run/submit")
    async def run_submit(body: SubmitRequest):
        p = _require_participant(body.code)
        run = backend.get_latest_run(p["participant_id"])
        if not run:
            raise HTTPException(status_code=400, detail="No active run")
        backend.submit_run(run["id"])
        return {"ok": True, "status": "submitted"}

    def _guard_window(participant_id: str):
        run = backend.get_latest_run(participant_id)
        if not run or run["status"] == "expired":
            raise HTTPException(status_code=440, detail="Session expired")
        if run["status"] == "submitted":
            raise HTTPException(status_code=409, detail="Run already submitted")
        return run

    @router.post("/session/start")
    async def session_start(body: SessionStartRequest):
        p = _require_participant(body.code)
        _guard_window(p["participant_id"])
        scenario = _resolve_scenario(backend, p, body.scenario_order)
        engine = _scenario_engine(scenario)
        # Prepare the engine this scenario needs (may restart :5002); the client
        # watches the prepare SSE and connects only when ready.
        manager.start_prepare_async(backend, p["study_id"], engine)

        # scenario_order 0 => the practice/test session (runs first). Distinct id so it's
        # easy to exclude from study counting/analysis; still recorded + saved.
        is_test = body.scenario_order == 0
        session_id = f"{p['participant_id']}_TEST" if is_test else f"{p['participant_id']}_S{body.scenario_order:02d}"
        _trace_session(session_id=session_id, participant_id=p["participant_id"],
                       study_id=p["study_id"], scenario_order=body.scenario_order, engine=engine)
        # target speaker id from the first vc segment (for metadata)
        target_speaker = ""
        for seg in scenario.get("voice_schedule") or []:
            if seg.get("mode") == "vc" and seg.get("target_ref"):
                for t in backend.list_targets(p["study_id"]):
                    if t["ref"] == seg["target_ref"]:
                        target_speaker = t["speaker_id"]
                        break
                break
        backend.create_session(session_id, p["participant_id"], f"scenario_{scenario['id']}",
                               body.scenario_order, _schedule_label(scenario), target_speaker)
        return {"session_id": session_id, "scenario": _scenario_card(scenario, body.scenario_order),
                "prepare": manager.get_state()}

    @router.post("/audio-check/start")
    async def audio_check_start(body: EnterRequest):
        """Warm the default VC engine and hand back a throwaway '_CHECK' session so
        the participant can run a short PersonaPlex exchange through the proxy."""
        p = _require_participant(body.code)
        _guard_window(p["participant_id"])
        manager.start_prepare_async(backend, p["study_id"], None)
        return {"session_id": f"{p['participant_id']}_CHECK", "prepare": manager.get_state()}

    @router.get("/condition/{session_id}")
    async def get_condition(session_id: str):
        """Internal: the active VC engine resolves the hidden prompt + voice
        schedule here (localhost). Never called by the browser."""
        _trace_session(session_id=session_id)
        # Audio-check session: generic natural pass-through, not a real scenario.
        if session_id.endswith("_CHECK"):
            return {
                "text_prompt": os.environ.get(
                    "STUDY_AUDIO_CHECK_PROMPT",
                    "You are performing a brief audio check. Warmly greet the participant, "
                    "confirm out loud that you can hear them, and ask them to continue."),
                "voice_prompt": os.environ.get("STUDY_DEFAULT_VOICE_PROMPT", "NATF2.pt"),
                "schedule": [{"mode": "natural", "start_s": 0, "end_s": None}],
            }
        session = backend.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Unknown session")
        _trace_session(session_id=session_id, study_id=session["study_id"])
        study = backend.get_study(session["study_id"])
        participants = {pp["participant_id"]: pp for pp in backend.list_participants(session["study_id"])}
        p = participants.get(session["participant_id"])
        if not p or not study:
            raise HTTPException(status_code=404, detail="Unknown participant/study")
        scenario = _resolve_scenario(backend, p, session["scenario_order"])
        targets = {t["ref"]: t for t in backend.list_targets(session["study_id"])}

        schedule = scenario.get("voice_schedule") or [{"mode": "natural", "start_s": 0, "end_s": None}]
        resolved = []
        for seg in schedule:
            r = {"mode": seg.get("mode", "natural"), "start_s": seg.get("start_s", 0),
                 "end_s": seg.get("end_s")}
            if seg.get("mode") == "vc":
                t = targets.get(seg.get("target_ref"))
                r["engine_target_id"] = (t or {}).get("engine_target_id")
            resolved.append(r)
        default_voice = os.environ.get("STUDY_DEFAULT_VOICE_PROMPT", "NATF2.pt")
        # PersonaPlex (moshi) requires a non-empty text prompt — an empty one leaves
        # its text_prompt_tokens None and crashes the connection. Fall back to a
        # neutral prompt when a scenario's system prompt is blank.
        text_prompt = (scenario.get("system_prompt") or "").strip() or os.environ.get(
            "STUDY_DEFAULT_SYSTEM_PROMPT",
            "You are a helpful conversational partner. Keep your replies concise.")
        return {"text_prompt": text_prompt,
                "voice_prompt": scenario.get("voice_prompt") or default_voice,
                "schedule": resolved,
                "study_id": session["study_id"]}

    @router.post("/session/{session_id}/save")
    async def session_save(session_id: str,
                           participant: UploadFile | None = File(None),
                           participant_raw: UploadFile | None = File(None),
                           model: UploadFile | None = File(None),
                           merged: UploadFile | None = File(None),
                           model_transcript: str = Form("null")):
        session = backend.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Unknown session")
        _trace_session(session_id=session_id, participant_id=session["participant_id"],
                       study_id=session.get("study_id"),
                       scenario_order=session.get("scenario_order"),
                       voice_condition=session.get("voice_condition"))
        out_dir = SESSIONS_DIR / session["participant_id"] / session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, up in (("participant", participant), ("participant_raw", participant_raw),
                         ("model", model), ("merged", merged)):
            if up is not None:
                dest = out_dir / f"{name}.wav"
                with open(dest, "wb") as f:
                    shutil.copyfileobj(up.file, f)
                files[name] = str(dest.relative_to(STUDY_DATA_DIR))

        model_turns = json.loads(model_transcript) if model_transcript and model_transcript != "null" else []
        metadata = {
            "participant_id": session["participant_id"], "session_id": session_id,
            "scenario_id": session["scenario_id"], "scenario_order": session["scenario_order"],
            "voice_condition": session["voice_condition"], "target_speaker_id": session["target_speaker_id"],
            "files": files,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        # Save audio + the model transcript only. Whisper/metrics inference is
        # deferred to the admin-triggered batch (it competes with live inference).
        backend.save_session(session_id, files, {"model": model_turns, "participant": None}, None, False)
        if not files:
            # No audio captured at all -> the VC/PersonaPlex path produced nothing
            # (e.g. engine misconfigured, or PersonaPlex never replied). The final
            # playback will be empty for this scenario. Surface it in the logs.
            logging.getLogger("study").warning(
                f"[study] session {session_id} saved with NO audio files "
                f"(model_turns={len(model_turns)}) — VC/PersonaPlex path likely failed")
        return {"ok": True, "files": files, "analysis": "deferred"}

    @router.get("/ping")
    async def ping():
        """Tiny endpoint the browser round-trips to measure network latency."""
        return {"ok": True}

    @router.post("/telemetry")
    async def telemetry(body: dict):
        """Client-measured latencies for a session (network RTT, connect, first-audio).
        Recorded as `client.*` latency histograms + logged, tagged with the session."""
        sid = body.get("session_id")
        marks = body.get("marks") or {}
        if logging_setup and sid:
            logging_setup.set_log_session(sid)
        if otel:
            otel.set_session_attributes(session_id=sid)
        for k, v in marks.items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if otel:
                otel.record_latency(f"client.{k}", val, session_id=sid)
        logger.info(f"[study] client telemetry session={sid} marks={marks}")
        return {"ok": True}

    @router.post("/session/{session_id}/end")
    async def session_end(session_id: str, body: dict):
        if not backend.get_session(session_id):
            raise HTTPException(status_code=404, detail="Unknown session")
        backend.end_session(session_id, body.get("reason", "goal_reached"))
        return {"ok": True}

    @router.post("/session/{session_id}/questionnaire")
    async def session_questionnaire(session_id: str, body: QuestionnaireRequest):
        p = _require_participant(body.code)
        backend.save_answer(p["participant_id"], session_id if session_id != "none" else None,
                            body.kind, body.payload)
        return {"ok": True}

    @router.get("/playback/{code}")
    async def playback(code: str, scenario: int = 0, track: str = "merged"):
        """Streams a recording for the post-session playback item. `scenario`
        (1-based order) + `track` (merged|participant) select it explicitly; unset
        falls back to the participant's VC->natural scenario's merged recording.
        A participant can only fetch their own recording."""
        p = backend.get_participant_by_code(code)
        if not p:
            raise HTTPException(status_code=404, detail="Invalid code")
        track_key = "participant" if track == "participant" else "merged"
        order = _participant_order(backend, p)   # positions match session ids (practice excluded)

        def serve(order_idx: int):
            session = backend.get_session(f"{p['participant_id']}_S{order_idx:02d}")
            files = (session or {}).get("files") or {}
            rel = files.get(track_key) or files.get("merged")
            if rel:
                path = STUDY_DATA_DIR / rel
                if path.exists():
                    return FileResponse(str(path), media_type="audio/wav")
            return None

        if scenario and 1 <= scenario <= len(order):
            r = serve(scenario)
            if r:
                return r
        else:
            for i, sid in enumerate(order):
                sc = backend.get_scenario(sid)
                if sc and _is_vc_to_natural(sc):
                    r = serve(i + 1)
                    if r:
                        return r
        raise HTTPException(status_code=404, detail="No playback recording available yet")

    return router
