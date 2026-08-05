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

from study.analysis import (STUDY_DATA_DIR, TRANSCRIPT_SCHEMA, _session_paths,
                            run_session_analysis, status_path)  # noqa: E402
from study.dialogue_transcript import (DIALOGUE_TRANSCRIPT_SCHEMA,
                                       prepare_dialogue_transcript)  # noqa: E402
from study.session_scope import analysis_eligible  # noqa: E402
from study.storage import get_backend  # noqa: E402
from study.technical_validity import (TECHNICAL_VALIDITY_SCHEMA,
                                      prepare_technical_validity)  # noqa: E402
from study.timing_analysis import TIMING_SCHEMA, prepare_timing_analysis  # noqa: E402
from study.transmitted_transcript import (TRANSMITTED_TRANSCRIPT_SCHEMA,
                                          prepare_transmitted_transcript)  # noqa: E402
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


def _latest_analysis_result(session: dict, key: str) -> dict | None:
    analysis = (session.get("artifact_manifest") or {}).get("analysis") or {}
    latest = analysis.get(key) or {}
    relative_path = latest.get("path") if isinstance(latest, dict) else None
    if not relative_path:
        return None
    try:
        result = json.loads((STUDY_DATA_DIR / relative_path).read_text())
        if isinstance(result, dict):
            result.setdefault("result_artifact", copy.deepcopy(latest))
        return result
    except (OSError, ValueError, TypeError):
        return None


def _needs_timing(session: dict) -> bool:
    result = _latest_analysis_result(session, "timing_latest")
    return not result or result.get("schema") != TIMING_SCHEMA


def _needs_validity(session: dict) -> bool:
    result = _latest_analysis_result(session, "technical_validity_latest")
    return (not result or result.get("schema") != TECHNICAL_VALIDITY_SCHEMA
            or result.get("status") == "incomplete")


def _needs_dialogue_transcript(session: dict) -> bool:
    result = _latest_analysis_result(session, "dialogue_transcript_latest")
    return not result or result.get("schema") != DIALOGUE_TRANSCRIPT_SCHEMA


def _needs_transmitted_transcript(session: dict) -> bool:
    result = _latest_analysis_result(session, "transmitted_transcript_latest")
    return not result or result.get("schema") != TRANSMITTED_TRANSCRIPT_SCHEMA


def _needs_preprocessing(session: dict) -> bool:
    transcript = session.get("transcript") or {}
    return (session.get("metrics") is None
            or not isinstance(transcript, dict)
            or transcript.get("schema") != TRANSCRIPT_SCHEMA
            or "participant_segments" not in transcript)


def _analysis_candidates(sessions: list[dict], force: bool) -> list[dict]:
    return [session for session in sessions
            if analysis_eligible(session)
            and session.get("ended_at") is not None
            and (force or _needs_preprocessing(session) or _needs_timing(session)
                 or _needs_dialogue_transcript(session)
                 or _needs_transmitted_transcript(session)
                 or _needs_validity(session))]


def main() -> None:
    study_id = int(sys.argv[1])
    force = "--force" in sys.argv[2:]

    backend = get_backend()
    sessions = backend.list_sessions(study_id)
    pending = _analysis_candidates(sessions, force)
    total = len(pending)
    done = 0
    _write(running=True, phase="preprocessing", done=0, total=total,
           current=None, study_id=study_id, error=None)

    for s in pending:
        _write(running=True, phase="preprocessing", done=done, total=total,
               current=s["session_id"], study_id=study_id, error=None)
        if logging_setup:
            logging_setup.set_log_session(s["session_id"], study_id)
        files = s.get("files") or {}
        conv, raw, mt = _session_paths(s)
        span_cm = (otel.start_span(_tracer, "analysis.session",
                                   attributes={"study.session_id": s["session_id"],
                                               "study.study_id": study_id})
                   if otel else contextlib.nullcontext())
        with span_cm:
            stage_errors: dict[str, str] = {}
            timing = None
            analysis_id = (
                f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}."
                f"{time.time_ns() % 1_000_000_000:09d}Z"
            )
            if files.get("participant") and (force or _needs_preprocessing(s)):
                try:
                    run_session_analysis(s["session_id"], conv, raw, mt)
                except Exception as exc:  # noqa: BLE001
                    stage_errors["preprocessing"] = str(exc)
                    print(f"[analysis_worker] preprocessing error for {s['session_id']}: {exc}",
                          file=sys.stderr)
            elif not files.get("participant"):
                stage_errors["preprocessing"] = "participant audio is missing"

            try:
                if force or _needs_preprocessing(s):
                    # Refresh after preprocessing because it updates the DB row.
                    s = backend.get_session(s["session_id"]) or s
                if not (s.get("files") or {}).get("participant_raw"):
                    # Died-at-connect attempts never captured audio; timing can
                    # never be reconstructed, so record why instead of raising
                    # the same KeyError on every pass.
                    stage_errors["timing"] = "raw microphone audio was never captured"
                    timing = _latest_analysis_result(s, "timing_latest")
                elif force or _needs_timing(s):
                    timing = prepare_timing_analysis(s, STUDY_DATA_DIR, analysis_id)
                    latest = backend.get_session(s["session_id"]) or s
                    manifest = copy.deepcopy(latest.get("artifact_manifest") or {})
                    manifest.setdefault("analysis", {})["timing_latest"] = timing[
                        "result_artifact"]
                    backend.update_session_artifacts(s["session_id"], manifest)
                    s = backend.get_session(s["session_id"]) or latest
                else:
                    timing = _latest_analysis_result(s, "timing_latest")
            except Exception as exc:  # noqa: BLE001
                stage_errors["timing"] = str(exc)
                print(f"[analysis_worker] timing error for {s['session_id']}: {exc}",
                      file=sys.stderr)

            try:
                latest = backend.get_session(s["session_id"]) or s
                if timing and (force or _needs_dialogue_transcript(latest)):
                    dialogue = prepare_dialogue_transcript(
                        latest, STUDY_DATA_DIR, analysis_id, timing)
                    transcript = copy.deepcopy(latest.get("transcript") or {})
                    embedded = copy.deepcopy(dialogue)
                    embedded["artifact"] = embedded.pop("result_artifact")
                    transcript["dialogue"] = embedded
                    backend.update_session_analysis(
                        s["session_id"], transcript, latest.get("metrics"),
                        bool(latest.get("audiobox_available")),
                    )
                    latest = backend.get_session(s["session_id"]) or latest
                    manifest = copy.deepcopy(latest.get("artifact_manifest") or {})
                    analysis = manifest.setdefault("analysis", {})
                    analysis["dialogue_transcript_latest"] = dialogue["result_artifact"]
                    analysis["dialogue_transcript_summary"] = {
                        "schema": dialogue["schema"],
                        "analysis_id": dialogue["analysis_id"],
                        "status": dialogue["status"],
                        "timing_status": dialogue["timeline"]["timing_status"],
                        **dialogue["summary"],
                    }
                    backend.update_session_artifacts(s["session_id"], manifest)
                    s = backend.get_session(s["session_id"]) or latest
            except Exception as exc:  # noqa: BLE001
                print(f"[analysis_worker] dialogue-transcript error for {s['session_id']}: {exc}",
                      file=sys.stderr)

            try:
                latest = backend.get_session(s["session_id"]) or s
                if timing and (force or _needs_transmitted_transcript(latest)):
                    if (latest.get("files") or {}).get("participant"):
                        transmitted = prepare_transmitted_transcript(
                            latest, STUDY_DATA_DIR, analysis_id, timing)
                        manifest = copy.deepcopy(latest.get("artifact_manifest") or {})
                        analysis = manifest.setdefault("analysis", {})
                        analysis["transmitted_transcript_latest"] = transmitted[
                            "result_artifact"]
                        analysis["transmitted_transcript_summary"] = {
                            "schema": transmitted["schema"],
                            "analysis_id": transmitted["analysis_id"],
                            "status": transmitted["status"],
                            "alignment": transmitted["alignment"],
                            **transmitted["summary"],
                        }
                        backend.update_session_artifacts(s["session_id"], manifest)
                        s = backend.get_session(s["session_id"]) or latest
                    else:
                        stage_errors["transmitted_transcript"] = (
                            "participant (transmitted) audio is missing")
            except Exception as exc:  # noqa: BLE001
                stage_errors["transmitted_transcript"] = str(exc)
                print(f"[analysis_worker] transmitted-transcript error for {s['session_id']}: {exc}",
                      file=sys.stderr)

            try:
                validity = prepare_technical_validity(
                    backend.get_session(s["session_id"]) or s,
                    STUDY_DATA_DIR, analysis_id, timing, stage_errors,
                )
                latest = backend.get_session(s["session_id"]) or s
                manifest = copy.deepcopy(latest.get("artifact_manifest") or {})
                analysis = manifest.setdefault("analysis", {})
                analysis["technical_validity_latest"] = validity["result_artifact"]
                analysis["technical_validity_summary"] = {
                    key: validity[key] for key in (
                        "schema", "evaluated_at", "status",
                        "valid_for_condition_analysis",
                        "valid_for_post_checkpoint_analysis",
                        "valid_for_timing_reconstruction",
                        "valid_for_confirmatory_timing_analysis",
                        "speech_boundary_validation_status", "analysis_checkpoint_s",
                        "failures", "warnings",
                    )
                }
                backend.update_session_artifacts(s["session_id"], manifest)
            except Exception as exc:  # noqa: BLE001
                print(f"[analysis_worker] technical-validity error for {s['session_id']}: {exc}",
                      file=sys.stderr)
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
