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

from fastapi import (APIRouter, BackgroundTasks, Depends, File, Form, Header,
                     HTTPException, UploadFile)
from fastapi.responses import JSONResponse, StreamingResponse

from .analysis import run_session_analysis
from .engine import get_manager
from .models import (REQUIRED_CARD_FIELDS, CreateStudyRequest, EnterRequest,
                     GenerateRequest, ProgressRequest, QuestionnaireRequest,
                     RunStartRequest, Scenario, SessionStartRequest,
                     SubmitRequest, UpdateStudyRequest, default_questionnaires)
from .storage import get_backend

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = Path(os.environ.get("WORKSPACE", str(REPO_ROOT.parent)))
STUDY_DATA_DIR = Path(os.environ.get("STUDY_DATA_DIR", str(REPO_ROOT / "study_data")))
TARGETS_DIR = STUDY_DATA_DIR / "targets"
SESSIONS_DIR = STUDY_DATA_DIR / "sessions"

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


def _resolve_scenario(backend, participant: dict, scenario_order: int) -> dict:
    order = participant.get("scenario_order") or []
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
        "extra_fields": [f for f in (card.get("extra_fields") or []) if f.get("label")],
        "time_limit_s": scenario.get("time_limit_s", 300),
    }
    for key, _label in REQUIRED_CARD_FIELDS:
        out[key] = card.get(key, "")
    return out


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

    @router.get("/studies", dependencies=[Depends(require_admin)])
    async def list_studies():
        return {"studies": backend.list_studies()}

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
        study = backend.update_study(study_id, body.name, body.description, None)
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
        order = p.get("scenario_order") or []
        scenarios = []
        for i, sid in enumerate(order):
            sc = backend.get_scenario(sid)
            if sc:
                scenarios.append(_scenario_card(sc, i + 1))
        run = backend.get_latest_run(p["participant_id"])
        return {"participant_id": p["participant_id"], "study_name": study["name"],
                "scenarios": scenarios, "questionnaires": study.get("questionnaires") or {},
                "run": _run_public(run)}

    @router.post("/run/start")
    async def run_start(body: RunStartRequest):
        p = _require_participant(body.code)
        live = backend.get_live_run(p["study_id"])
        if live and live["participant_id"] != p["participant_id"]:
            raise HTTPException(status_code=409,
                                detail="Another session is in progress, please try again shortly.")
        run = backend.start_run(p["participant_id"], body.mode)
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

        session_id = f"{p['participant_id']}_S{body.scenario_order:02d}"
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

    @router.get("/condition/{session_id}")
    async def get_condition(session_id: str):
        """Internal: the active VC engine resolves the hidden prompt + voice
        schedule here (localhost). Never called by the browser."""
        session = backend.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Unknown session")
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
                "schedule": resolved}

    @router.post("/session/{session_id}/save")
    async def session_save(session_id: str, background_tasks: BackgroundTasks,
                           participant: UploadFile | None = File(None),
                           participant_raw: UploadFile | None = File(None),
                           model: UploadFile | None = File(None),
                           merged: UploadFile | None = File(None),
                           model_transcript: str = Form("null")):
        session = backend.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Unknown session")
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
        backend.save_session(session_id, files, {"model": model_turns, "participant": None}, None, False)

        converted_path = str(out_dir / "participant.wav") if participant is not None else None
        raw_path = str(out_dir / "participant_raw.wav") if participant_raw is not None else None
        background_tasks.add_task(run_session_analysis, session_id, converted_path, raw_path, model_turns)
        return {"ok": True, "files": files, "analysis": "scheduled"}

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

    return router
