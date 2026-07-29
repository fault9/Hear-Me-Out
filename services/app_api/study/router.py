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
import copy
import io
import json
import logging
import os
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import (APIRouter, Depends, File, Form, Header, HTTPException,
                     UploadFile)
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .analysis import get_runner
from .artifacts import (append_jsonl, atomic_write_bytes, atomic_write_json,
                        canonical_json_bytes, file_record, git_revision,
                        immutable_copy, sha256_bytes)
from .counterbalance import (CounterbalanceError, allocate as allocate_variants,
                             balance_report, choose_balanced_target,
                             has_deferred_target_assignment,
                             resolve_target_assignment,
                             target_assignment_configuration,
                             validate_and_compile)
from .engine import get_manager
from . import yaml_io
from .models import (REQUIRED_CARD_FIELDS, CreateStudyRequest, EnterRequest,
                     GenerateRequest, ProgressRequest, QuestionnaireRequest,
                     RunStartRequest, Scenario, SessionStartRequest,
                     SubmitRequest, UpdateStudyRequest, default_questionnaires)
from .playback import (ensure_stable_converted_interaction_playback,
                       ensure_stable_converted_playback,
                       ensure_transition_playback)
from .questionnaires import missing_required_answers
from .session_scope import (analysis_eligible, annotate_analysis_scopes,
                            session_study_role)
from .storage import get_backend
from .vc_quality_analysis import get_vc_quality_runner

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
EVENT_TOKEN = os.environ.get("STUDY_EVENT_TOKEN") or "local-study-events"
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


def _validate_required_answers(items: list[dict], payload: dict) -> None:
    missing = missing_required_answers(items, payload)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Required questionnaire item {missing[0]!r} is missing")


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
    scenario = copy.deepcopy(scenario)
    override = (participant.get("assignment") or {}).get(str(scenario["id"]))
    if override:
        scenario["voice_schedule"] = copy.deepcopy(override.get("voice_schedule") or [])
        scenario["assigned_condition"] = override.get("condition")
    return scenario


def _session_dir(session: dict) -> Path:
    return (SESSIONS_DIR / f"study_{session['study_id']}" / session["participant_id"] /
            f"run_{int(session.get('run_attempt') or 1):02d}" /
            f"scenario_{int(session['scenario_order']):02d}" /
            f"attempt_{int(session.get('scenario_attempt') or 1):02d}_{session['session_id']}")


def _target_for_schedule(targets: list[dict], schedule: list[dict]) -> Optional[dict]:
    by_ref = {target["ref"]: target for target in targets}
    for segment in schedule:
        if segment.get("mode") == "vc" and segment.get("target_ref"):
            return by_ref.get(segment["target_ref"])
    return None


def _initialize_session_artifacts(backend, session: dict, study_snapshot: dict,
                                  schedule: list[dict], target: Optional[dict]) -> dict:
    out_dir = _session_dir(session)
    out_dir.mkdir(parents=True, exist_ok=False)
    config_bytes = canonical_json_bytes(study_snapshot)
    atomic_write_bytes(out_dir / "study_config.json", config_bytes, exclusive=True)
    session_config = {
        "study_id": session["study_id"],
        "participant_id": session["participant_id"],
        "variant_id": session.get("config_snapshot", {}).get("participant", {}).get("variant_id"),
        "session_id": session["session_id"],
        "run_id": session.get("run_id"),
        "run_attempt": session.get("run_attempt"),
        "scenario_id": session["scenario_id"],
        "scenario_order": session["scenario_order"],
        "scenario_attempt": session.get("scenario_attempt"),
        "condition": session["voice_condition"],
        "voice_schedule": schedule,
        "sample_rates_hz": {"input": 16000, "transmitted": 16000, "model_bound": 24000},
    }
    atomic_write_json(out_dir / "session_config.json", session_config, exclusive=True)
    (out_dir / "events.jsonl").touch(exist_ok=False)

    artifacts = {
        "study_config": file_record(out_dir / "study_config.json", relative_to=STUDY_DATA_DIR),
        "session_config": file_record(out_dir / "session_config.json", relative_to=STUDY_DATA_DIR),
        "events": {"path": str((out_dir / "events.jsonl").relative_to(STUDY_DATA_DIR))},
    }
    if target and Path(target["wav_path"]).exists():
        target_record = immutable_copy(target["wav_path"], out_dir / "target.wav")
        target_record["path"] = str((out_dir / "target.wav").relative_to(STUDY_DATA_DIR))
        target_record["ref"] = target["ref"]
        target_record["speaker_id"] = target.get("speaker_id")
        target_record["engine"] = target.get("engine")
        artifacts["target"] = target_record

    manifest = {
        "schema": "hmo.study-artifacts.v1",
        "created_at_unix": time.time(),
        "identifiers": {key: session.get(key) for key in (
            "study_id", "participant_id", "session_id", "run_id", "run_attempt",
            "scenario_id", "scenario_order", "scenario_attempt")},
        "condition": session["voice_condition"],
        "configuration_sha256": sha256_bytes(config_bytes),
        "software": {
            "hmo_commit": git_revision(REPO_ROOT),
            "xvc_commit": os.environ.get("XVC_GIT_COMMIT"),
            "vc_quality_commit": os.environ.get("VC_QUALITY_GIT_COMMIT") or git_revision(REPO_ROOT),
            "personaplex_version": os.environ.get("PERSONAPLEX_VERSION"),
        },
        "artifacts": artifacts,
        "analysis": {},
    }
    atomic_write_json(out_dir / "manifest.initial.json", manifest, exclusive=True)
    backend.update_session_artifacts(session["session_id"], manifest)
    return manifest


def _scenario_card(scenario: dict, scenario_order: int) -> dict:
    card = scenario.get("scenario_card") or {}
    out = {
        "scenario_order": scenario_order,
        "scenario_id": scenario.get("id"),
        "title": scenario.get("title", ""),
        "extra_fields": [f for f in (card.get("extra_fields") or []) if f.get("label")],
        "post_items": scenario.get("post_items") or [],   # scenario-specific post questions
        "time_limit_s": scenario.get("time_limit_s", 300),
        "study_role": card.get("study_role", "analytical"),
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
    preferred = os.environ.get("VC_ENGINE", "").strip().lower()
    if preferred in engines:
        engines.remove(preferred)
        engines.insert(0, preferred)
    return engines


def build_study_router() -> APIRouter:
    router = APIRouter(prefix="/api/study")
    backend = get_backend()
    manager = get_manager()
    allocation_lock = threading.Lock()
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
        if backend.list_participants(study_id):
            raise HTTPException(
                status_code=409,
                detail="Target voices are frozen after participant codes are generated. Create a new study variant.",
            )
        d = TARGETS_DIR / f"study{study_id}"
        d.mkdir(parents=True, exist_ok=True)
        dest = d / f"{ref}.wav"
        with open(dest, "wb") as f:
            shutil.copyfileobj(wav.file, f)
        t = backend.add_target(study_id, ref, speaker_id or ref, label or wav.filename or ref, str(dest), engine)
        return {"target": t}

    @router.delete("/studies/{study_id}/targets/{target_id}", dependencies=[Depends(require_admin)])
    async def delete_target(study_id: int, target_id: int):
        if backend.list_participants(study_id):
            raise HTTPException(status_code=409,
                                detail="Target voices are frozen after participant codes are generated.")
        backend.delete_target(target_id)
        return {"ok": True}

    @router.post("/studies/{study_id}/participants/generate", dependencies=[Depends(require_admin)])
    async def gen_participants(study_id: int, body: GenerateRequest):
        scenarios = backend.list_scenarios(study_id)
        scenario_ids = [s["id"] for s in scenarios]
        if not scenario_ids:
            raise HTTPException(status_code=400, detail="Add at least one scenario first")
        study = backend.get_study(study_id)
        participants = backend.list_participants(study_id)
        count = max(1, body.count)
        try:
            settings = study.get("settings") or {}
            targets = backend.list_targets(study_id)
            if has_deferred_target_assignment(settings):
                validate_and_compile(settings, scenarios, targets)
                allocations = [{"allocation_status": "awaiting_profile"}
                               for _ in range(count)]
            else:
                allocations = allocate_variants(
                    settings, scenarios, targets, participants, count)
        except CounterbalanceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        created = backend.generate_participants(study_id, count, scenario_ids,
                                                allocations or None)
        return {"participants": created}

    @router.get("/studies/{study_id}/counterbalance", dependencies=[Depends(require_admin)])
    async def counterbalance_status(study_id: int):
        study = backend.get_study(study_id)
        if not study:
            raise HTTPException(status_code=404, detail="Unknown study")
        try:
            return balance_report(study.get("settings") or {}, backend.list_scenarios(study_id),
                                  backend.list_targets(study_id),
                                  backend.list_participants(study_id))
        except CounterbalanceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/studies/{study_id}/runs", dependencies=[Depends(require_admin)])
    async def list_runs(study_id: int):
        return {"runs": backend.list_runs(study_id)}

    @router.get("/studies/{study_id}/sessions", dependencies=[Depends(require_admin)])
    async def list_sessions(study_id: int):
        return {"sessions": annotate_analysis_scopes(
            backend.list_sessions(study_id), backend.list_runs(study_id))}

    @router.post("/studies/{study_id}/analyze", dependencies=[Depends(require_admin)])
    async def analyze(study_id: int, force: bool = False):
        """Run Whisper transcription + VC-quality metrics over the study's saved
        sessions (batch, background). Deferred here so it doesn't compete with the
        live study. force=true re-analyzes already-processed sessions."""
        if not backend.get_study(study_id):
            raise HTTPException(status_code=404, detail="Unknown study")
        if get_vc_quality_runner().get_status().get("running"):
            raise HTTPException(status_code=409,
                                detail="VC-quality analysis is already running")
        return get_runner().start(backend, study_id, force)

    @router.get("/studies/{study_id}/analyze/status", dependencies=[Depends(require_admin)])
    async def analyze_status(study_id: int):
        return get_runner().get_status()

    @router.post("/studies/{study_id}/vc-quality", dependencies=[Depends(require_admin)])
    async def run_vc_quality(study_id: int, body: dict):
        """Run the real vc_quality.py post-hoc for one session, one participant,
        or every captured session in this study. Original artifacts are read-only."""
        if not backend.get_study(study_id):
            raise HTTPException(status_code=404, detail="Unknown study")
        if get_runner().get_status().get("running"):
            raise HTTPException(status_code=409,
                                detail="The full analysis pipeline is already running")
        participant_id = body.get("participant_id") or None
        session_id = body.get("session_id") or None
        if participant_id and session_id:
            raise HTTPException(status_code=422,
                                detail="Choose either participant_id or session_id, not both")
        sessions = backend.list_sessions(study_id)
        if participant_id and not any(s["participant_id"] == participant_id for s in sessions):
            raise HTTPException(status_code=404, detail="Participant has no sessions in this study")
        if session_id and not any(s["session_id"] == session_id for s in sessions):
            raise HTTPException(status_code=404, detail="Session is not part of this study")
        return get_vc_quality_runner().start(
            study_id, participant_id=participant_id, session_id=session_id,
            force=bool(body.get("force", False)))

    @router.get("/studies/{study_id}/vc-quality/status", dependencies=[Depends(require_admin)])
    async def vc_quality_status(study_id: int):
        return get_vc_quality_runner().get_status()

    @router.get("/studies/{study_id}/export", dependencies=[Depends(require_admin)])
    async def export(study_id: int, format: str = "json"):
        study = backend.get_study(study_id)
        sessions = annotate_analysis_scopes(
            backend.list_sessions(study_id), backend.list_runs(study_id))
        data = {
            "study": study,
            "scenarios": backend.list_scenarios(study_id),
            "targets": backend.list_targets(study_id),
            "participants": backend.list_participants(study_id),
            "runs": backend.list_runs(study_id),
            "sessions": sessions,
            "answers": backend.list_answers(study_id),
        }
        if format == "json":
            return JSONResponse(data)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("study_export.json", json.dumps(data, indent=2))
            z.writestr("study_config.yaml", yaml_io.dump_yaml(yaml_io.study_to_dict(backend, study_id)))
            study_sessions_dir = SESSIONS_DIR / f"study_{study_id}"
            if study_sessions_dir.exists():
                for p in study_sessions_dir.rglob("*"):
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
        settings = study.get("settings") or {}
        return {"participant_id": p["participant_id"], "study_name": study["name"],
                "scenarios": scenarios, "questionnaires": study.get("questionnaires") or {},
                "welcome_text": settings.get("welcome_text", ""),
                "estimated_duration": settings.get("estimated_duration", ""),
                "practice_intro_text": settings.get("practice_intro_text", ""),
                "main_intro_text": settings.get("main_intro_text", ""),
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
        study = backend.get_study(p["study_id"])
        if (has_deferred_target_assignment((study or {}).get("settings") or {}) and
                (p.get("allocation_status") != "assigned" or not p.get("target_ref"))):
            raise HTTPException(
                status_code=409,
                detail="Complete the background questionnaire before starting a scenario")
        run = _guard_window(p["participant_id"])
        if not backend.has_answer(p["participant_id"], run["id"], "background"):
            raise HTTPException(
                status_code=409,
                detail="Complete consent, audio check, and background questions before a scenario")
        scenario = _resolve_scenario(backend, p, body.scenario_order)
        engine = _scenario_engine(scenario) or (
            ((study or {}).get("settings") or {}).get("study_engine"))
        targets = backend.list_targets(p["study_id"])
        targets_by_ref = {target["ref"]: target for target in targets}
        vc_segments = [segment for segment in (scenario.get("voice_schedule") or [])
                       if segment.get("mode") == "vc"]
        segment_engines = {segment.get("engine") or engine for segment in vc_segments}
        segment_engines.discard(None)
        if len(segment_engines) > 1:
            raise HTTPException(
                status_code=409,
                detail="A scenario cannot switch between different VC engines mid-conversation")
        for segment in vc_segments:
            ref = segment.get("target_ref")
            target_for_segment = targets_by_ref.get(ref)
            expected_engine = segment.get("engine") or engine
            if not ref or not target_for_segment:
                raise HTTPException(
                    status_code=409,
                    detail=f"VC target {ref or '(missing)'} is not configured for this study")
            if expected_engine and target_for_segment.get("engine") != expected_engine:
                raise HTTPException(
                    status_code=409,
                    detail=(f"VC target {ref!r} uses {target_for_segment.get('engine')}, "
                            f"but this scenario requires {expected_engine}"))
        # Prepare the engine this scenario needs (may restart :5002); the client
        # watches the prepare SSE and connects only when ready.
        manager.start_prepare_async(backend, p["study_id"], engine)

        scenario_attempt = backend.next_session_attempt(p["participant_id"], run["id"],
                                                        body.scenario_order)
        session_id = (f"{p['participant_id']}_R{int(run['attempt']):02d}_"
                      f"S{body.scenario_order:02d}_A{scenario_attempt:02d}")
        _trace_session(session_id=session_id, participant_id=p["participant_id"],
                       study_id=p["study_id"], scenario_order=body.scenario_order, engine=engine)
        # target speaker id from the first vc segment (for metadata)
        target_speaker = ""
        target = _target_for_schedule(targets, scenario.get("voice_schedule") or [])
        if target:
            target_speaker = target.get("speaker_id") or ""
        study_snapshot = yaml_io.study_to_dict(backend, p["study_id"])
        config_snapshot = {
            "study": study_snapshot,
            "engine": engine,
            "participant": {"participant_id": p["participant_id"],
                            "variant_id": p.get("variant_id"),
                            "target_ref": p.get("target_ref"),
                            "scenario_order": p.get("scenario_order"),
                            "assignment": p.get("assignment") or {}},
            "scenario": scenario,
        }
        backend.create_session(
            session_id, p["participant_id"], f"scenario_{scenario['id']}",
            body.scenario_order, scenario.get("assigned_condition") or _schedule_label(scenario),
            target_speaker, run["id"], run["attempt"], scenario_attempt,
            scenario.get("voice_schedule") or [], config_snapshot,
        )
        session = backend.get_session(session_id)
        try:
            _initialize_session_artifacts(backend, session, study_snapshot,
                                          scenario.get("voice_schedule") or [], target)
        except (OSError, FileExistsError) as exc:
            backend.end_session(session_id, "artifact_initialization_failed")
            raise HTTPException(status_code=500,
                                detail=f"Could not initialize immutable session artifacts: {exc}") from exc
        return {"session_id": session_id, "scenario": _scenario_card(scenario, body.scenario_order),
                "run_attempt": run["attempt"], "scenario_attempt": scenario_attempt,
                "prepare": manager.get_state()}

    @router.post("/audio-check/start")
    async def audio_check_start(body: EnterRequest):
        """Warm the default VC engine and hand back a throwaway '_CHECK' session so
        the participant can run a short PersonaPlex exchange through the proxy."""
        p = _require_participant(body.code)
        run = _guard_window(p["participant_id"])
        if (not backend.has_answer(p["participant_id"], run["id"], "consent") or
                not backend.has_answer(p["participant_id"], run["id"], "background")):
            raise HTTPException(
                status_code=403,
                detail="Consent and background questions must be completed before audio testing")
        study = backend.get_study(p["study_id"])
        requested_engine = ((study or {}).get("settings") or {}).get("study_engine")
        manager.start_prepare_async(backend, p["study_id"], requested_engine)
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
        snapshot = session.get("config_snapshot") or {}
        scenario = snapshot.get("scenario") or _resolve_scenario(backend, p, session["scenario_order"])
        targets = {t["ref"]: t for t in backend.list_targets(session["study_id"])}

        schedule = session.get("schedule") or scenario.get("voice_schedule") or [
            {"mode": "natural", "start_s": 0, "end_s": None}]
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
                           model_transcript: str = Form("null"),
                           client_timeline: str = Form("null")):
        session = backend.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Unknown session")
        _trace_session(session_id=session_id, participant_id=session["participant_id"],
                       study_id=session.get("study_id"),
                       scenario_order=session.get("scenario_order"),
                       voice_condition=session.get("voice_condition"))
        out_dir = _session_dir(session)
        if not out_dir.exists():
            raise HTTPException(status_code=500, detail="Session artifact directory is missing")
        try:
            model_turns = (json.loads(model_transcript)
                           if model_transcript and model_transcript != "null" else [])
            timeline = (json.loads(client_timeline)
                        if client_timeline and client_timeline != "null" else None)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="Invalid session JSON artifact") from exc
        if not isinstance(model_turns, list):
            raise HTTPException(status_code=422, detail="Invalid model transcript")
        if (timeline is not None and
                (not isinstance(timeline, dict)
                 or timeline.get("schema") != "hmo.client-timeline.v1")):
            raise HTTPException(status_code=422, detail="Invalid client timeline")
        manifest = copy.deepcopy(session.get("artifact_manifest") or {})
        artifacts = manifest.setdefault("artifacts", {})
        files = {}
        for name, up in (("participant", participant), ("participant_raw", participant_raw),
                         ("model", model), ("merged", merged)):
            if up is not None:
                dest = out_dir / f"{name}.wav"
                try:
                    atomic_write_bytes(dest, await up.read(), exclusive=True)
                except FileExistsError as exc:
                    raise HTTPException(status_code=409,
                                        detail=f"Immutable artifact already exists: {name}.wav") from exc
                files[name] = str(dest.relative_to(STUDY_DATA_DIR))
                artifacts[name] = file_record(dest, relative_to=STUDY_DATA_DIR)

        try:
            atomic_write_json(out_dir / "model_transcript.json", model_turns, exclusive=True)
        except FileExistsError as exc:
            raise HTTPException(status_code=409,
                                detail="Immutable model transcript already exists") from exc
        artifacts["model_transcript"] = file_record(out_dir / "model_transcript.json",
                                                     relative_to=STUDY_DATA_DIR)
        if timeline is not None:
            try:
                atomic_write_json(out_dir / "client_timeline.json", timeline, exclusive=True)
            except FileExistsError as exc:
                raise HTTPException(status_code=409,
                                    detail="Immutable client timeline already exists") from exc
            artifacts["client_timeline"] = file_record(
                out_dir / "client_timeline.json", relative_to=STUDY_DATA_DIR)
        events_path = out_dir / "events.jsonl"
        if events_path.exists():
            artifacts["events"] = file_record(events_path, relative_to=STUDY_DATA_DIR)
        metadata = {
            "participant_id": session["participant_id"], "session_id": session_id,
            "run_id": session.get("run_id"), "run_attempt": session.get("run_attempt"),
            "scenario_attempt": session.get("scenario_attempt"),
            "scenario_id": session["scenario_id"], "scenario_order": session["scenario_order"],
            "voice_condition": session["voice_condition"], "target_speaker_id": session["target_speaker_id"],
            "study_role": session_study_role(session),
            "analysis_eligible": analysis_eligible(session),
            "files": files,
        }
        atomic_write_json(out_dir / "metadata.json", metadata, exclusive=True)
        artifacts["metadata"] = file_record(out_dir / "metadata.json", relative_to=STUDY_DATA_DIR)
        expected = {"participant", "participant_raw", "model", "merged"}
        # Proxy teardown can upload its artifacts while the browser is uploading
        # these files. Re-read and merge so neither completion path loses records.
        latest_manifest = copy.deepcopy(
            (backend.get_session(session_id) or {}).get("artifact_manifest") or {})
        latest_artifacts = latest_manifest.setdefault("artifacts", {})
        latest_artifacts.update(artifacts)
        manifest = latest_manifest
        manifest["capture"] = {
            "saved_at_unix": time.time(),
            "complete": expected.issubset(files),
            "missing": sorted(expected - set(files)),
            "client_timeline_complete": timeline is not None,
        }
        atomic_write_json(out_dir / "manifest.capture.json", manifest, exclusive=True)
        backend.update_session_artifacts(session_id, manifest)
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

    @router.post("/internal/session/{session_id}/proxy-artifacts")
    async def ingest_proxy_artifacts(
            session_id: str,
            proxy_received_wav: UploadFile | None = File(None),
            participant_proxy_wav: UploadFile | None = File(None),
            personaplex_input_opus: UploadFile | None = File(None),
            personaplex_input_decoded_wav: UploadFile | None = File(None),
            personaplex_output_opus: UploadFile | None = File(None),
            metadata: str = Form("{}"),
            x_study_event_token: str = Header(default="")):
        if x_study_event_token != EVENT_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid event token")
        session = backend.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Unknown session")
        out_dir = _session_dir(session)
        if (out_dir / "manifest.final.json").exists():
            raise HTTPException(status_code=409, detail="Session artifacts are finalized")
        try:
            proxy_metadata = json.loads(metadata or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="Invalid proxy artifact metadata") from exc
        if not isinstance(proxy_metadata, dict):
            raise HTTPException(status_code=422, detail="Invalid proxy artifact metadata")

        uploads = {
            "proxy_received.wav": proxy_received_wav,
            "participant_proxy.wav": participant_proxy_wav,
            "personaplex_input.opus": personaplex_input_opus,
            "personaplex_input_decoded.wav": personaplex_input_decoded_wav,
            "personaplex_output.opus": personaplex_output_opus,
        }
        records = {}
        for filename, upload in uploads.items():
            if upload is None:
                continue
            destination = out_dir / filename
            try:
                atomic_write_bytes(destination, await upload.read(), exclusive=True)
            except FileExistsError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=f"Immutable proxy artifact already exists: {filename}",
                ) from exc
            records[filename] = file_record(destination, relative_to=STUDY_DATA_DIR)
        atomic_write_json(out_dir / "proxy_timeline.json", proxy_metadata, exclusive=True)
        records["proxy_timeline"] = file_record(
            out_dir / "proxy_timeline.json", relative_to=STUDY_DATA_DIR)

        latest = backend.get_session(session_id) or session
        manifest = copy.deepcopy(latest.get("artifact_manifest") or {})
        manifest.setdefault("artifacts", {}).update(records)
        manifest["proxy_capture"] = {
            "saved_at_unix": time.time(),
            "complete": bool(records.get("proxy_received.wav")
                             and records.get("participant_proxy.wav")
                             and records.get("personaplex_input.opus")),
            "artifacts": sorted(records),
        }
        backend.update_session_artifacts(session_id, manifest)
        return {"ok": True, "artifacts": sorted(records)}

    @router.post("/internal/session/{session_id}/events")
    async def ingest_events(session_id: str, body: dict,
                            x_study_event_token: str = Header(default="")):
        if x_study_event_token != EVENT_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid event token")
        session = backend.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Unknown session")
        if (_session_dir(session) / "manifest.final.json").exists():
            raise HTTPException(status_code=409, detail="Session event timeline is finalized")
        rows = body.get("events") or []
        if not isinstance(rows, list) or len(rows) > 1000:
            raise HTTPException(status_code=422, detail="events must be a list of at most 1000 rows")
        events_path = _session_dir(session) / "events.jsonl"
        last_sequence = 0
        try:
            for line in events_path.read_text().splitlines():
                if line.strip():
                    last_sequence = max(last_sequence,
                                        int(json.loads(line).get("event_sequence") or 0))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        clean = []
        for row in sorted(rows, key=lambda item: item.get("event_sequence", 0)
                          if isinstance(item, dict) else 0):
            if not isinstance(row, dict) or not row.get("event"):
                continue
            if int(row.get("event_sequence") or 0) <= last_sequence:
                continue
            clean.append({**row, "session_id": session_id,
                          "ingested_at_unix": time.time()})
            last_sequence = int(row.get("event_sequence") or last_sequence)
        append_jsonl(events_path, clean)
        return {"ok": True, "accepted": len(clean)}

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
        session = backend.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Unknown session")
        reason = body.get("reason", "goal_reached")
        events_path = _session_dir(session) / "events.jsonl"
        # WebSocket close and the proxy's final event/artifact POSTs are separate
        # requests. An event file may not exist yet when short calls end because
        # live delivery is buffered, so use the frozen engine choice as the guard.
        snapshot = session.get("config_snapshot") or {}
        expect_proxy = snapshot.get("engine") == "xvc"
        if expect_proxy:
            deadline = time.monotonic() + float(
                os.environ.get("STUDY_PROXY_FINALIZE_TIMEOUT_S", "30"))
            while time.monotonic() < deadline:
                try:
                    stopped = (events_path.exists() and any(
                        json.loads(line).get("event") == "stream_stop"
                        for line in events_path.read_text().splitlines() if line.strip()))
                    latest = backend.get_session(session_id) or session
                    proxy_capture = (latest.get("artifact_manifest") or {}).get(
                        "proxy_capture") or {}
                    if stopped and proxy_capture.get("complete"):
                        break
                except (OSError, json.JSONDecodeError):
                    pass
                await asyncio.sleep(0.1)
        backend.end_session(session_id, reason)
        latest_session = backend.get_session(session_id) or session
        manifest = copy.deepcopy(latest_session.get("artifact_manifest") or {})
        manifest["ended_at_unix"] = time.time()
        manifest["end_reason"] = reason
        if events_path.exists():
            manifest.setdefault("artifacts", {})["events"] = file_record(
                events_path, relative_to=STUDY_DATA_DIR)
        final_path = _session_dir(session) / "manifest.final.json"
        if not final_path.exists():
            atomic_write_json(final_path, manifest, exclusive=True)
        backend.update_session_artifacts(session_id, manifest)
        return {"ok": True}

    @router.post("/session/{session_id}/questionnaire")
    async def session_questionnaire(session_id: str, body: QuestionnaireRequest):
        p = _require_participant(body.code)
        run = _guard_window(p["participant_id"])
        session = None
        if session_id != "none":
            session = backend.get_session(session_id)
            if not session or session["participant_id"] != p["participant_id"]:
                raise HTTPException(status_code=404, detail="Unknown participant session")
        study = backend.get_study(p["study_id"])
        settings = (study or {}).get("settings") or {}
        questionnaires = (study or {}).get("questionnaires") or {}
        _validate_required_answers(questionnaires.get(body.kind) or [], body.payload)
        if session and body.kind in ("post", "practice_post"):
            snapshot_scenario = (session.get("config_snapshot") or {}).get("scenario") or {}
            _validate_required_answers(snapshot_scenario.get("post_items") or [], body.payload)
        prerequisite = {
            "consent": "eligibility",
            "background": "consent",
            "audio_check": "background",
            "pre_playback": "post",
            "playback": "pre_playback",
            "debrief": "playback",
        }.get(body.kind)
        if prerequisite and not backend.has_answer(
                p["participant_id"], run["id"], prerequisite):
            raise HTTPException(
                status_code=409,
                detail=f"Complete {prerequisite.replace('_', ' ')} before {body.kind.replace('_', ' ')}")
        target_config = target_assignment_configuration(settings)
        assigned = None
        if target_config and body.kind == str(
                target_config.get("questionnaire_kind") or "background"):
            try:
                resolved = resolve_target_assignment(settings, body.kind, body.payload)
                with allocation_lock:
                    latest = backend.get_participant_by_code(body.code)
                    if latest.get("allocation_status") == "awaiting_profile":
                        participants = backend.list_participants(p["study_id"])
                        target_ref = resolved.get("target_ref") or choose_balanced_target(
                            resolved.get("target_candidates") or [], participants)
                        allocation = allocate_variants(
                            settings, backend.list_scenarios(p["study_id"]),
                            backend.list_targets(p["study_id"]),
                            participants, 1,
                            target_ref=target_ref,
                            allocation_stratum=resolved["allocation_stratum"],
                        )[0]
                        assigned = backend.assign_participant(
                            p["participant_id"], allocation,
                            resolved["allocation_stratum"])
                    else:
                        assigned = latest
                        expected_target = resolved.get("target_ref")
                        if expected_target and assigned.get("target_ref") != expected_target:
                            raise HTTPException(
                                status_code=409,
                                detail="The voice target for this participant is already fixed")
            except CounterbalanceError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.kind == "eligibility" and body.payload.get("eligibility_18") != "Yes":
            raise HTTPException(status_code=403, detail="This study is limited to adults aged 18 or older")
        if (body.kind == "consent" and
                body.payload.get("consent_decision") != "I consent and wish to continue."):
            raise HTTPException(status_code=403, detail="Consent is required to continue")
        backend.save_answer(p["participant_id"], session_id if session_id != "none" else None,
                            body.kind, body.payload)
        return {"ok": True, "allocation": ({
            "status": assigned.get("allocation_status"),
            "stratum": assigned.get("allocation_stratum"),
            "variant_id": assigned.get("variant_id"),
            "target_ref": assigned.get("target_ref"),
        } if assigned else None)}

    @router.get("/playback/{code}")
    async def playback(code: str, scenario: int = 0, track: str = "merged",
                       condition: str = "", max_duration_s: int = 0):
        """Streams a recording for the post-session playback item. `scenario`
        (1-based order) + `track` (merged|participant) select it explicitly; unset
        falls back to the participant's VC->natural scenario's merged recording.
        A participant can only fetch their own recording."""
        p = backend.get_participant_by_code(code)
        if not p:
            raise HTTPException(status_code=404, detail="Invalid code")
        track_key = "participant" if track == "participant" else "merged"
        order = p.get("scenario_order") or []

        def serve(order_idx: int):
            run = backend.get_latest_run(p["participant_id"])
            session = backend.get_latest_session(p["participant_id"], order_idx,
                                                 run.get("id") if run else None)
            files = (session or {}).get("files") or {}
            rel = files.get(track_key) or files.get("merged")
            if rel:
                path = STUDY_DATA_DIR / rel
                if path.exists():
                    if (track_key == "participant" and condition == "vc_deactivation"
                            and max_duration_s):
                        try:
                            path, _manifest = ensure_transition_playback(
                                session, STUDY_DATA_DIR, max_duration_s)
                        except (FileNotFoundError, ValueError, OSError) as exc:
                            raise HTTPException(
                                status_code=409,
                                detail=f"Could not prepare the playback excerpt: {exc}") from exc
                    elif (track_key == "participant" and condition == "stable_converted"
                          and max_duration_s):
                        try:
                            path, _manifest = ensure_stable_converted_playback(
                                session, STUDY_DATA_DIR, max_duration_s)
                        except (FileNotFoundError, ValueError, OSError) as exc:
                            raise HTTPException(
                                status_code=409,
                                detail=f"Could not prepare the playback excerpt: {exc}") from exc
                    elif (track_key == "merged" and condition == "stable_converted"
                          and max_duration_s):
                        try:
                            path, _manifest = ensure_stable_converted_interaction_playback(
                                session, STUDY_DATA_DIR, max_duration_s)
                        except (FileNotFoundError, ValueError, OSError) as exc:
                            raise HTTPException(
                                status_code=409,
                                detail=f"Could not prepare the playback excerpt: {exc}") from exc
                    return FileResponse(str(path), media_type="audio/wav")
            return None

        if scenario and 1 <= scenario <= len(order):
            r = serve(scenario)
            if r:
                return r
        elif condition:
            assignment = p.get("assignment") or {}
            for i, sid in enumerate(order):
                if (assignment.get(str(sid)) or {}).get("condition") == condition:
                    r = serve(i + 1)
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
