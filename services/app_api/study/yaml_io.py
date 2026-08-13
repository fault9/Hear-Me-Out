"""YAML import/export for studies.

A study exports to (and imports from) a single YAML file: name, settings,
questionnaires, the target refs it expects, and the scenarios. Import populates
an existing study — the researcher must have uploaded the target voices first,
and every `target_ref` used in a scenario's voice schedule must match an
uploaded target (by Speaker ID / ref, and engine). Import fails with the list of
missing/mismatched targets rather than creating a half-configured study.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import HTTPException

from .counterbalance import CounterbalanceError, validate_and_compile

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "study.yaml"


def study_to_dict(backend, study_id: int) -> dict:
    study = backend.get_study(study_id)
    scenarios = backend.list_scenarios(study_id)
    targets = backend.list_targets(study_id)
    settings = study.get("settings") or {}
    return {
        "name": study["name"],
        "description": study.get("description", ""),
        "settings": settings,
        "counterbalancing": settings.get("counterbalancing") or {},
        "questionnaires": study.get("questionnaires") or {},
        "targets": [{"ref": t["ref"], "engine": t["engine"], "speaker_id": t["speaker_id"]} for t in targets],
        "scenarios": [{
            "title": s.get("title", ""),
            "system_prompt": s.get("system_prompt", ""),
            "voice_prompt": s.get("voice_prompt", ""),
            "time_limit_s": s.get("time_limit_s", 300),
            "scenario_card": s.get("scenario_card") or {},
            "voice_schedule": s.get("voice_schedule") or [],
            "post_items": s.get("post_items") or [],
        } for s in scenarios],
    }


def dump_yaml(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


def _target_problems(backend, study_id: int, data: dict) -> list[str]:
    tmap = {t["ref"]: t for t in backend.list_targets(study_id)}
    problems = []
    for expected in data.get("targets", []) or []:
        ref = expected.get("ref")
        if not ref:
            continue
        target = tmap.get(ref)
        if not target:
            problems.append(f"missing target voice '{ref}' (upload it with Speaker ID '{ref}')")
            continue
        expected_engine = expected.get("engine")
        if expected_engine and target.get("engine") != expected_engine:
            problems.append(
                f"target '{ref}' is {target.get('engine')} but the study expects {expected_engine}")
    for sc in data.get("scenarios", []) or []:
        for seg in sc.get("voice_schedule", []) or []:
            if seg.get("mode") == "vc" and seg.get("target_ref"):
                ref = seg["target_ref"]
                t = tmap.get(ref)
                if not t:
                    problems.append(f"missing target voice '{ref}' (upload it with Speaker ID '{ref}')")
                elif seg.get("engine") and t["engine"] != seg["engine"]:
                    problems.append(f"target '{ref}' is {t['engine']} but the scenario expects {seg['engine']}")
    config = data.get("counterbalancing") or (data.get("settings") or {}).get("counterbalancing") or {}
    for variant in config.get("variants", []) or []:
        ref = variant.get("target_ref")
        if ref and ref not in tmap:
            problems.append(f"missing counterbalance target voice '{ref}'")
    return sorted(set(problems))


def apply_import(backend, study_id: int, data: dict) -> None:
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="YAML did not parse to a study object")
    problems = _target_problems(backend, study_id, data)
    if problems:
        raise HTTPException(status_code=422,
                            detail="Upload the target voices first — " + "; ".join(problems))
    settings = dict(data.get("settings") or {})
    if "counterbalancing" in data:
        settings["counterbalancing"] = data.get("counterbalancing") or {}
    try:
        pseudo_scenarios = [{**scenario, "id": index + 1, "order_idx": index}
                            for index, scenario in enumerate(data.get("scenarios", []) or [])]
        validate_and_compile(settings, pseudo_scenarios, backend.list_targets(study_id))
    except CounterbalanceError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid counterbalancing: {exc}") from exc
    backend.update_study(study_id, data.get("name"), data.get("description"),
                         data.get("questionnaires"), settings)
    for s in backend.list_scenarios(study_id):
        backend.delete_scenario(s["id"])
    for i, sc in enumerate(data.get("scenarios", []) or []):
        backend.add_scenario(study_id, {**sc, "order_idx": i})


def parse_yaml(raw: bytes | str) -> dict:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {e}")
