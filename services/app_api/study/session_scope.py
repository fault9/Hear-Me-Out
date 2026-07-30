"""Derive whether a captured session belongs in the analytical dataset."""

from __future__ import annotations

from collections import defaultdict


def session_study_role(session: dict) -> str:
    snapshot = session.get("config_snapshot") or {}
    scenario = snapshot.get("scenario") or {}
    card = scenario.get("scenario_card") or {}
    role = card.get("study_role")
    if role in {"practice", "analytical"}:
        return role
    return "practice" if session.get("voice_condition") == "practice" else "analytical"


def analysis_eligible(session: dict) -> bool:
    """Return whether post-hoc processing should run for this capture.

    All analytical attempts are processed. Canonical dataset inclusion is a
    separate decision so retries and failed attempts remain auditable.
    """
    return session_study_role(session) != "practice"


def technical_validity_summary(session: dict) -> dict:
    analysis = (session.get("artifact_manifest") or {}).get("analysis") or {}
    summary = analysis.get("technical_validity_summary")
    if isinstance(summary, dict):
        return summary
    return {"status": "pending", "valid_for_condition_analysis": False,
            "valid_for_post_checkpoint_analysis": False,
            "failures": [], "warnings": []}


def annotate_analysis_scope(session: dict) -> dict:
    role = session_study_role(session)
    return {
        **session,
        "study_role": role,
        "analysis_eligible": role != "practice",
        "analysis_processing_eligible": role != "practice",
        "technical_validity": technical_validity_summary(session),
    }


def annotate_analysis_scopes(sessions: list[dict], runs: list[dict]) -> list[dict]:
    """Add canonical-run/attempt inclusion fields without discarding captures.

    The latest submitted run is canonical. Within it, the highest technically
    valid attempt for each scenario is selected. Every other analytical capture
    remains available for preprocessing, diagnostics, and sensitivity checks.
    """
    latest_submitted: dict[str, dict] = {}
    for run in runs:
        if run.get("status") != "submitted":
            continue
        participant_id = str(run.get("participant_id") or "")
        previous = latest_submitted.get(participant_id)
        if previous is None or int(run.get("attempt") or 0) > int(previous.get("attempt") or 0):
            latest_submitted[participant_id] = run

    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for session in sessions:
        if not analysis_eligible(session):
            continue
        key = (str(session.get("participant_id") or ""),
               int(session.get("run_id") or 0),
               int(session.get("scenario_order") or 0))
        grouped[key].append(session)

    canonical_ids: set[str] = set()
    for (participant_id, run_id, _scenario_order), attempts in grouped.items():
        run = latest_submitted.get(participant_id)
        if not run or int(run.get("id") or 0) != run_id:
            continue
        valid = [session for session in attempts
                 if technical_validity_summary(session).get(
                     "valid_for_condition_analysis") is True]
        if valid:
            chosen = max(valid, key=lambda row: int(row.get("scenario_attempt") or 0))
            canonical_ids.add(str(chosen.get("session_id")))

    annotated = []
    for session in sessions:
        item = annotate_analysis_scope(session)
        reasons: list[str] = []
        role = item["study_role"]
        participant_id = str(session.get("participant_id") or "")
        run = latest_submitted.get(participant_id)
        validity = item["technical_validity"]
        canonical_run = bool(run and int(run.get("id") or 0) == int(session.get("run_id") or 0))
        canonical_attempt = str(session.get("session_id")) in canonical_ids

        if role == "practice":
            reasons.append("practice")
        elif run is None:
            reasons.append("no_submitted_run")
        elif not canonical_run:
            reasons.append("superseded_run")
        elif validity.get("status") == "pending":
            reasons.append("technical_validity_pending")
        elif validity.get("status") == "incomplete":
            reasons.append("technical_validity_incomplete")
        elif validity.get("valid_for_condition_analysis") is not True:
            reasons.extend(f"technical:{failure.get('code')}"
                           for failure in validity.get("failures") or [])
            if not validity.get("failures"):
                reasons.append("technical_invalid")
        elif not canonical_attempt:
            reasons.append("superseded_attempt")

        item.update({
            "canonical_run": canonical_run,
            "canonical_attempt": canonical_attempt,
            "analysis_included": not reasons,
            "analysis_exclusion_reasons": reasons,
        })
        annotated.append(item)
    return annotated
