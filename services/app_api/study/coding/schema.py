"""Label schema, validation, and deterministic consistency checks.

Plain-dict validation (no pydantic) so the coding pipeline can run in any
Python environment that can read the data root. `validate_labels` returns a
list of error strings (empty = valid); `consistency_issues` applies the
codebook's deterministic cross-field rules (Section 2 zero-vs-missing
semantics and derived-label preconditions).
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "hmo.coding-labels.v2"

REPAIR_CATEGORIES = (
    "repetition", "reformulation", "clarification", "explicit_correction",
    "repeated_confirmation", "restart_after_cutoff", "floor_recovery",
)
ACCESS_FLAG_TYPES = (
    "truncation", "abandonment", "failed_floor_acquisition",
    "premature_task_closure",
)
FINAL_ACCOUNT_VALUES = (
    "accurate", "partially_accurate", "inaccurate", "no_final_account",
)
GROUNDING_STAGES = ("acknowledgement", "update_claim", "incorporation", "retention")
VERIFIER_VERDICTS = ("agree", "disagree", "uncertain")
CONFIDENCE_FLAG_THRESHOLD = 0.8


def _is_binary(value: Any) -> bool:
    return value in (0, 1)


def _is_binary_or_null(value: Any) -> bool:
    return value is None or value in (0, 1)


def _is_confidence(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


def _check_id_list(value: Any, known_ids: set[str], where: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        errors.append(f"{where}: must be a list of utterance-id strings")
        return
    for v in value:
        if v not in known_ids:
            errors.append(f"{where}: unknown utterance id {v!r}")


def blank_labels(n_units: int) -> dict:
    """An empty label object shaped the way `validate_labels` expects, so a
    human coder fills values instead of hand-building nested JSON. A typing
    error in a reliability sample is indistinguishable from disagreement."""
    unit = {"unit_index": 0, "attempted": None, "complete_raw": None,
            "delivery_utterance_ids": [],
            **{stage: None for stage in GROUNDING_STAGES},
            "evidence_utterance_ids": {}, "confidence": {}}
    return {
        "units": [dict(unit, unit_index=i + 1) for i in range(n_units)],
        "repairs": [],
        "final_probe": {"utterance_id": None,
                        "spontaneous_final_account_utterance_id": None},
        "outcome": {"level": None, "evidence_utterance_ids": [],
                    "rationale": "", "confidence": None},
        "final_account_accuracy": {"value": None, "evidence_utterance_ids": [],
                                   "confidence": None},
        "access_flags": [],
        "notes": "",
    }


def validate_labels(labels: dict, packet: dict) -> list[str]:
    """Structural validation of a judge/human label object against its packet."""
    errors: list[str] = []
    if not isinstance(labels, dict):
        return ["labels: not a JSON object"]
    known_ids = {u.get("id") for u in packet.get("utterances", []) if isinstance(u, dict)}
    participant_ids = {u.get("id") for u in packet.get("utterances", [])
                       if isinstance(u, dict) and u.get("speaker") == "participant"}
    scenario = packet.get("scenario") or {}
    expected_units = len(scenario.get("unit_definitions") or [])
    levels = {row.get("score") for row in (scenario.get("outcome_levels") or [])
              if isinstance(row, dict)}

    units = labels.get("units")
    if not isinstance(units, list) or len(units) != expected_units:
        errors.append(f"units: expected a list of {expected_units} unit objects")
        units = []
    for i, unit in enumerate(units, start=1):
        where = f"units[{i}]"
        if not isinstance(unit, dict):
            errors.append(f"{where}: not an object")
            continue
        if unit.get("unit_index") != i:
            errors.append(f"{where}.unit_index: must equal {i}")
        if not _is_binary(unit.get("attempted")):
            errors.append(f"{where}.attempted: must be 0 or 1")
        if not _is_binary_or_null(unit.get("complete_raw")):
            errors.append(f"{where}.complete_raw: must be 0, 1, or null")
        _check_id_list(unit.get("delivery_utterance_ids", []), participant_ids,
                       f"{where}.delivery_utterance_ids", errors)
        for stage in GROUNDING_STAGES:
            if not _is_binary_or_null(unit.get(stage)):
                errors.append(f"{where}.{stage}: must be 0, 1, or null")
        evidence = unit.get("evidence_utterance_ids")
        if not isinstance(evidence, dict):
            errors.append(f"{where}.evidence_utterance_ids: must be an object")
        else:
            for key, value in evidence.items():
                _check_id_list(value, known_ids,
                               f"{where}.evidence_utterance_ids.{key}", errors)
        confidence = unit.get("confidence")
        if not isinstance(confidence, dict):
            errors.append(f"{where}.confidence: must be an object")
        else:
            for key, value in confidence.items():
                if not _is_confidence(value):
                    errors.append(f"{where}.confidence.{key}: must be a number in [0,1]")

    repairs = labels.get("repairs")
    if not isinstance(repairs, list):
        errors.append("repairs: must be a list")
        repairs = []
    for i, move in enumerate(repairs):
        where = f"repairs[{i}]"
        if not isinstance(move, dict):
            errors.append(f"{where}: not an object")
            continue
        if move.get("category") not in REPAIR_CATEGORIES:
            errors.append(f"{where}.category: must be one of {REPAIR_CATEGORIES}")
        if move.get("utterance_id") not in participant_ids:
            errors.append(f"{where}.utterance_id: unknown participant utterance")
        if not isinstance(move.get("trouble"), str) or not move.get("trouble", "").strip():
            errors.append(f"{where}.trouble: must be a non-empty string")
        if not _is_confidence(move.get("confidence")):
            errors.append(f"{where}.confidence: must be a number in [0,1]")

    probe = labels.get("final_probe")
    if not isinstance(probe, dict):
        errors.append("final_probe: must be an object")
    else:
        for key in ("utterance_id", "spontaneous_final_account_utterance_id"):
            value = probe.get(key)
            if value is not None and value not in known_ids:
                errors.append(f"final_probe.{key}: unknown utterance id {value!r}")

    outcome = labels.get("outcome")
    if not isinstance(outcome, dict):
        errors.append("outcome: must be an object")
    else:
        if outcome.get("level") not in (levels or {1, 2, 3, 4}):
            errors.append(f"outcome.level: must be one of {sorted(levels or {1, 2, 3, 4})}")
        _check_id_list(outcome.get("evidence_utterance_ids", []), known_ids,
                       "outcome.evidence_utterance_ids", errors)
        if not isinstance(outcome.get("rationale"), str) or not outcome.get("rationale", "").strip():
            errors.append("outcome.rationale: must be a non-empty string")
        if not _is_confidence(outcome.get("confidence")):
            errors.append("outcome.confidence: must be a number in [0,1]")

    account = labels.get("final_account_accuracy")
    if not isinstance(account, dict):
        errors.append("final_account_accuracy: must be an object")
    else:
        if account.get("value") not in FINAL_ACCOUNT_VALUES:
            errors.append(f"final_account_accuracy.value: must be one of {FINAL_ACCOUNT_VALUES}")
        _check_id_list(account.get("evidence_utterance_ids", []), known_ids,
                       "final_account_accuracy.evidence_utterance_ids", errors)
        if not _is_confidence(account.get("confidence")):
            errors.append("final_account_accuracy.confidence: must be a number in [0,1]")

    flags = labels.get("access_flags", [])
    if not isinstance(flags, list):
        errors.append("access_flags: must be a list")
        flags = []
    for i, flag in enumerate(flags):
        where = f"access_flags[{i}]"
        if not isinstance(flag, dict):
            errors.append(f"{where}: not an object")
            continue
        if flag.get("type") not in ACCESS_FLAG_TYPES:
            errors.append(f"{where}.type: must be one of {ACCESS_FLAG_TYPES}")
        if flag.get("utterance_id") not in known_ids:
            errors.append(f"{where}.utterance_id: unknown utterance id")
        if not _is_confidence(flag.get("confidence")):
            errors.append(f"{where}.confidence: must be a number in [0,1]")

    return errors


def validate_checks(parsed: Any, required_fields: list[str] | None = None) -> list[str]:
    """Structural validation of a verifier response. Disagreement drives human
    review, so a dropped check is indistinguishable from agreement."""
    if not isinstance(parsed, dict):
        return ["verifier output: not a JSON object"]
    checks = parsed.get("checks")
    if not isinstance(checks, list) or not checks:
        return ["checks: must be a non-empty list of check objects"]
    errors: list[str] = []
    if required_fields:
        seen = {row.get("field") for row in checks if isinstance(row, dict)}
        errors += [f"checks: no verdict for {field!r}"
                   for field in required_fields if field not in seen]
    for i, row in enumerate(checks):
        where = f"checks[{i}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: not an object")
            continue
        if not isinstance(row.get("field"), str) or not row.get("field", "").strip():
            errors.append(f"{where}.field: must be a non-empty label path")
        verdict = row.get("verdict")
        if verdict not in VERIFIER_VERDICTS:
            errors.append(f"{where}.verdict: must be one of {VERIFIER_VERDICTS}")
        elif verdict != "agree" and not str(row.get("note") or "").strip():
            errors.append(f"{where}.note: required for a {verdict} verdict")
    return errors


def consistency_issues(labels: dict, packet: dict) -> list[str]:
    """Deterministic codebook rules that a structurally valid label set can
    still violate. Any issue flags the packet for human review."""
    issues: list[str] = []
    # Retention carries a second legitimate null beyond the section 2 rule:
    # section 4 leaves it unset when the participant never requested a final
    # summary and no spontaneous final account was given.
    probe = labels.get("final_probe") or {}
    no_final_account = not (probe.get("utterance_id")
                            or probe.get("spontaneous_final_account_utterance_id"))
    for unit in labels.get("units") or []:
        i = unit.get("unit_index")
        where = f"units[{i}]"
        attempted = unit.get("attempted")
        complete = unit.get("complete_raw")
        delivery = unit.get("delivery_utterance_ids") or []
        if attempted == 0 and complete == 1:
            issues.append(f"{where}: complete_raw=1 requires attempted=1")
        if complete == 1 and not delivery:
            issues.append(f"{where}: complete_raw=1 requires delivery_utterance_ids")
        if complete != 1 and delivery:
            issues.append(f"{where}: delivery_utterance_ids given but complete_raw!=1")
        for stage in GROUNDING_STAGES:
            value = unit.get(stage)
            if complete != 1 and value is not None:
                issues.append(f"{where}.{stage}: must be null when the unit was "
                              "not completely delivered")
            if (complete == 1 and value is None
                    and not (stage == "retention" and no_final_account)):
                issues.append(f"{where}.{stage}: must be 0 or 1 when the unit "
                              "was completely delivered")
            if value == 1 and not (unit.get("evidence_utterance_ids") or {}).get(stage):
                issues.append(f"{where}.{stage}: positive label without evidence")
    # The top outcome level requires the units present in the final state of
    # the record, which is what retention codes: the same rule, so decide it
    # here rather than leaving the coder and the verifier to argue it.
    scores = [row.get("score")
              for row in ((packet.get("scenario") or {}).get("outcome_levels") or [])
              if isinstance(row, dict) and isinstance(row.get("score"), int)]
    top = max(scores) if scores else None
    if top is not None and (labels.get("outcome") or {}).get("level") == top:
        issues += [f"outcome.level={top} requires the critical units present in "
                   f"the final account, but units[{unit.get('unit_index')}]"
                   ".retention is 0"
                   for unit in labels.get("units") or []
                   if unit.get("retention") == 0]
    account = labels.get("final_account_accuracy") or {}
    if (account.get("value") not in (None, "no_final_account")
            and no_final_account):
        issues.append("final_account_accuracy set but no final probe or "
                      "spontaneous final account was identified")
    return issues


def low_confidence_fields(labels: dict, threshold: float = CONFIDENCE_FLAG_THRESHOLD) -> list[str]:
    fields: list[str] = []
    for unit in labels.get("units") or []:
        i = unit.get("unit_index")
        for key, value in (unit.get("confidence") or {}).items():
            if isinstance(value, (int, float)) and value < threshold:
                fields.append(f"units[{i}].{key}")
    for i, move in enumerate(labels.get("repairs") or []):
        if isinstance(move.get("confidence"), (int, float)) and move["confidence"] < threshold:
            fields.append(f"repairs[{i}]")
    outcome = labels.get("outcome") or {}
    if isinstance(outcome.get("confidence"), (int, float)) and outcome["confidence"] < threshold:
        fields.append("outcome")
    account = labels.get("final_account_accuracy") or {}
    if isinstance(account.get("confidence"), (int, float)) and account["confidence"] < threshold:
        fields.append("final_account_accuracy")
    return fields


def derive_scenario_labels(labels: dict) -> dict:
    """Codebook Section 4 derived labels. Never judged directly."""
    units = labels.get("units") or []

    def _both(stage_a: str, stage_b: str):
        values = []
        for unit in units:
            a, b = unit.get(stage_a), unit.get(stage_b)
            if a is None or b is None:
                values.append(None)
            else:
                values.append(1 if (a == 1 and b == 1) else 0)
        if any(v is None for v in values) and not any(v == 0 for v in values):
            return None
        return 1 if values and all(v == 1 for v in values) else 0

    per_unit_false_update = []
    for unit in units:
        claim = unit.get("update_claim")
        if claim != 1:
            per_unit_false_update.append(None)
        else:
            incorporation, retention = unit.get("incorporation"), unit.get("retention")
            failed = (incorporation == 0) or (retention == 0)
            per_unit_false_update.append(1 if failed else 0)
    observed = [v for v in per_unit_false_update if v is not None]
    false_update = (1 if any(v == 1 for v in observed) else 0) if observed else None

    repairs = labels.get("repairs") or []
    return {
        "demonstrated_grounding": _both("incorporation", "retention"),
        "acknowledgement_without_uptake": [
            (1 if (u.get("acknowledgement") == 1 and u.get("incorporation") == 0) else
             (None if u.get("acknowledgement") is None or u.get("incorporation") is None else 0))
            for u in units
        ],
        "false_update_confirmation": false_update,
        "false_update_confirmation_per_unit": per_unit_false_update,
        "any_repair": 1 if repairs else 0,
        "repair_total": len(repairs),
    }
