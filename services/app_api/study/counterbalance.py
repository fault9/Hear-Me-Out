"""Validation and allocation for prespecified study variants.

Researchers define the design in YAML. Participant-code generation only
allocates the next least-filled valid variant; outcomes never influence it.
"""

from __future__ import annotations

import copy
import secrets
from collections import Counter, defaultdict
from typing import Any


class CounterbalanceError(ValueError):
    pass


def _configuration(settings: dict | None) -> dict:
    return (settings or {}).get("counterbalancing") or {}


def target_assignment_configuration(settings: dict | None) -> dict:
    return _configuration(settings).get("target_assignment") or {}


def has_deferred_target_assignment(settings: dict | None) -> bool:
    return bool(target_assignment_configuration(settings))


def choose_balanced_target(candidates: list[str], participants: list[dict]) -> str:
    """Choose randomly among the currently least-used fallback targets."""
    if not candidates:
        raise CounterbalanceError("No fallback target voices are configured")
    counts = Counter(p.get("target_ref") for p in participants if p.get("target_ref"))
    minimum = min(counts[ref] for ref in candidates)
    return secrets.choice([ref for ref in candidates if counts[ref] == minimum])


def resolve_target_assignment(settings: dict | None, questionnaire_kind: str,
                              payload: dict) -> dict:
    """Resolve a participant answer to a fixed target or fallback target pool."""
    config = target_assignment_configuration(settings)
    if not config:
        raise CounterbalanceError("counterbalancing.target_assignment is not configured")
    expected_kind = str(config.get("questionnaire_kind") or "background")
    if questionnaire_kind != expected_kind:
        raise CounterbalanceError(
            f"target assignment requires the {expected_kind!r} questionnaire")
    answer_id = str(config.get("answer_id") or "").strip()
    mapping = config.get("target_by_answer") or {}
    if not answer_id or not isinstance(mapping, dict) or not mapping:
        raise CounterbalanceError(
            "target_assignment requires answer_id and target_by_answer")

    answer = payload.get(answer_id)
    values = answer if isinstance(answer, list) else [answer]
    matches = [(str(value), mapping[value]) for value in values if value in mapping]
    if not matches:
        fallback_targets = [str(ref) for ref in (config.get("fallback_targets") or [])]
        if not fallback_targets:
            raise CounterbalanceError(
                f"Answer {answer_id!r} must select one configured target-assignment category")
        return {
            "allocation_stratum": "fallback",
            "target_candidates": fallback_targets,
        }
    targets = {str(target) for _, target in matches}
    if len(targets) != 1:
        raise CounterbalanceError(
            f"Answer {answer_id!r} maps to more than one target voice")
    return {
        "allocation_stratum": matches[0][0],
        "target_ref": targets.pop(),
    }


def _scenario_by_position(scenarios: list[dict]) -> dict[int, dict]:
    ordered = sorted(scenarios, key=lambda s: (s.get("order_idx", 0), s.get("id", 0)))
    return {i + 1: scenario for i, scenario in enumerate(ordered)}


def _render_schedule(schedule: list[dict], target_ref: str | None) -> list[dict]:
    rendered = copy.deepcopy(schedule)
    for segment in rendered:
        if segment.get("mode") == "vc" and target_ref:
            segment["target_ref"] = target_ref
    return rendered


def validate_and_compile(settings: dict | None, scenarios: list[dict],
                         targets: list[dict]) -> list[dict]:
    config = _configuration(settings)
    variants = config.get("variants") or []
    by_position = _scenario_by_position(scenarios)
    target_refs = {target.get("ref") for target in targets}
    target_assignment = target_assignment_configuration(settings)
    if target_assignment:
        answer_id = str(target_assignment.get("answer_id") or "").strip()
        mapping = target_assignment.get("target_by_answer") or {}
        if not answer_id or not isinstance(mapping, dict) or not mapping:
            raise CounterbalanceError(
                "target_assignment requires answer_id and target_by_answer")
        fallback_targets = [str(ref) for ref in
                            (target_assignment.get("fallback_targets") or [])]
        if len(set(fallback_targets)) != len(fallback_targets):
            raise CounterbalanceError(
                "target_assignment.fallback_targets must not contain duplicates")
        unknown = sorted(
            ({str(ref) for ref in mapping.values()} | set(fallback_targets)) - target_refs)
        if unknown:
            raise CounterbalanceError(
                f"target_assignment references unknown target voice(s): {', '.join(unknown)}")
        if str(target_assignment.get("questionnaire_kind") or "background") != "background":
            raise CounterbalanceError(
                "target_assignment.questionnaire_kind must currently be 'background'")
    if not variants:
        return []

    conditions = config.get("conditions") or {}
    if not conditions:
        raise CounterbalanceError("counterbalancing.conditions is required when variants are defined")

    expected = set(by_position)
    seen_ids: set[str] = set()
    compiled: list[dict] = []

    for raw in variants:
        variant_id = str(raw.get("id") or "").strip()
        if not variant_id or variant_id in seen_ids:
            raise CounterbalanceError("variant ids must be non-empty and unique")
        seen_ids.add(variant_id)
        target_ref = raw.get("target_ref")
        if target_ref is None and not target_assignment:
            raise CounterbalanceError(
                f"variant {variant_id}: target_ref is required without target_assignment")
        if target_ref is not None and target_ref not in target_refs:
            raise CounterbalanceError(f"variant {variant_id}: unknown target_ref {target_ref!r}")

        order = [int(value) for value in (raw.get("scenario_order") or [])]
        if len(order) != len(expected) or set(order) != expected:
            raise CounterbalanceError(
                f"variant {variant_id}: scenario_order must contain each position "
                f"{sorted(expected)} exactly once")
        raw_assignment = raw.get("condition_assignment") or {}
        assignment_by_position = {int(key): value for key, value in raw_assignment.items()}
        if set(assignment_by_position) != expected:
            raise CounterbalanceError(
                f"variant {variant_id}: condition_assignment must cover positions {sorted(expected)}")

        assignment: dict[str, Any] = {}
        for position, condition_id in assignment_by_position.items():
            if condition_id not in conditions:
                raise CounterbalanceError(
                    f"variant {variant_id}: unknown condition {condition_id!r} at scenario {position}")
            condition = conditions[condition_id]
            schedule = condition.get("voice_schedule") if isinstance(condition, dict) else condition
            if not isinstance(schedule, list) or not schedule:
                raise CounterbalanceError(
                    f"condition {condition_id!r} must define a non-empty voice_schedule")
            scenario = by_position[position]
            assignment[str(scenario["id"])] = {
                "condition": str(condition_id),
                "voice_schedule": _render_schedule(schedule, target_ref),
            }

        compiled.append({
            "variant_id": variant_id,
            "target_ref": target_ref,
            "scenario_order": [by_position[position]["id"] for position in order],
            "assignment": assignment,
        })

    return compiled


def _apply_target(allocation: dict, target_ref: str) -> dict:
    rendered = copy.deepcopy(allocation)
    rendered["target_ref"] = target_ref
    for override in rendered.get("assignment", {}).values():
        override["voice_schedule"] = _render_schedule(
            override.get("voice_schedule") or [], target_ref)
    return rendered


def _default_allocation(scenarios: list[dict], target_ref: str) -> dict:
    ordered = sorted(scenarios, key=lambda s: (s.get("order_idx", 0), s.get("id", 0)))
    return {
        "variant_id": "default",
        "target_ref": target_ref,
        "scenario_order": [scenario["id"] for scenario in ordered],
        "assignment": {
            str(scenario["id"]): {
                "condition": "configured",
                "voice_schedule": _render_schedule(
                    scenario.get("voice_schedule") or [], target_ref),
            }
            for scenario in ordered
        },
    }


def allocate(settings: dict | None, scenarios: list[dict], targets: list[dict],
             participants: list[dict], count: int, target_ref: str | None = None,
             allocation_stratum: str | None = None) -> list[dict]:
    variants = validate_and_compile(settings, scenarios, targets)
    if not variants:
        if target_ref and has_deferred_target_assignment(settings):
            return [_default_allocation(scenarios, target_ref) for _ in range(count)]
        return []

    if target_ref:
        candidates = [variant for variant in variants
                      if variant.get("target_ref") in (None, target_ref)]
        if not candidates:
            raise CounterbalanceError(
                f"No counterbalance variant is eligible for target {target_ref!r}")
        candidates = [_apply_target(variant, target_ref) for variant in candidates]
    else:
        candidates = variants

    counted = participants
    if allocation_stratum is not None:
        counted = [p for p in participants
                   if p.get("allocation_stratum") == allocation_stratum]
    counts = Counter(p.get("variant_id") for p in counted if p.get("variant_id"))
    allocations: list[dict] = []
    for _ in range(count):
        chosen = min(candidates, key=lambda v: (counts[v["variant_id"]], v["variant_id"]))
        allocations.append(copy.deepcopy(chosen))
        counts[chosen["variant_id"]] += 1
    return allocations


def balance_report(settings: dict | None, scenarios: list[dict], targets: list[dict],
                   participants: list[dict]) -> dict:
    variants = validate_and_compile(settings, scenarios, targets)
    deferred = has_deferred_target_assignment(settings)
    if not variants and not deferred:
        return {"configured": False, "valid": True, "participants": len(participants)}

    if not variants:
        variants = [_default_allocation(scenarios, "__assigned_at_background__")]

    variant_counts = Counter(p.get("variant_id") for p in participants)
    design_cells: dict[str, Counter] = defaultdict(Counter)
    position_cells: dict[str, Counter] = defaultdict(Counter)
    configured_target_counts = Counter()
    for variant in variants:
        if variant.get("target_ref"):
            configured_target_counts[variant["target_ref"]] += 1
        for ordinal, scenario_id in enumerate(variant["scenario_order"], start=1):
            condition = variant["assignment"][str(scenario_id)]["condition"]
            design_cells[str(scenario_id)][condition] += 1
            position_cells[str(scenario_id)][str(ordinal)] += 1

    warnings = []
    for label, matrix in (("condition", design_cells), ("ordinal position", position_cells)):
        for scenario_id, cells in matrix.items():
            values = list(cells.values())
            if values and max(values) - min(values) > 1:
                warnings.append(f"scenario {scenario_id} is not balanced across {label}")
    if (not deferred and configured_target_counts and
            max(configured_target_counts.values()) - min(configured_target_counts.values()) > 1):
        warnings.append("target voices are not balanced across configured variants")

    variants_by_id = {variant["variant_id"]: variant for variant in variants}
    allocated_design: dict[str, Counter] = defaultdict(Counter)
    allocated_positions: dict[str, Counter] = defaultdict(Counter)
    allocated_targets = Counter()
    for participant in participants:
        variant = variants_by_id.get(participant.get("variant_id"))
        if not variant:
            continue
        allocated_target = participant.get("target_ref") or variant.get("target_ref")
        if allocated_target:
            allocated_targets[allocated_target] += 1
        for ordinal, scenario_id in enumerate(variant["scenario_order"], start=1):
            condition = variant["assignment"][str(scenario_id)]["condition"]
            allocated_design[str(scenario_id)][condition] += 1
            allocated_positions[str(scenario_id)][str(ordinal)] += 1

    return {
        "configured": True,
        "valid": not warnings,
        "warnings": warnings,
        "variant_counts": {v["variant_id"]: variant_counts[v["variant_id"]] for v in variants},
        "awaiting_profile": sum(
            p.get("allocation_status") == "awaiting_profile" for p in participants),
        "stratum_variant_counts": {
            stratum: dict(Counter(
                p.get("variant_id") for p in participants
                if p.get("allocation_stratum") == stratum and p.get("variant_id")))
            for stratum in sorted({p.get("allocation_stratum") for p in participants
                                   if p.get("allocation_stratum")})
        },
        "configured_design": {scenario: dict(cells) for scenario, cells in design_cells.items()},
        "configured_positions": {scenario: dict(cells) for scenario, cells in position_cells.items()},
        "allocated_design": {scenario: dict(cells) for scenario, cells in allocated_design.items()},
        "allocated_positions": {scenario: dict(cells) for scenario, cells in allocated_positions.items()},
        "allocated_targets": dict(allocated_targets),
    }
