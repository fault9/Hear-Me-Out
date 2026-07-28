"""Validation and allocation for prespecified study variants.

Researchers define the design in YAML. Participant-code generation only
allocates the next least-filled valid variant; outcomes never influence it.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any


class CounterbalanceError(ValueError):
    pass


def _configuration(settings: dict | None) -> dict:
    return (settings or {}).get("counterbalancing") or {}


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
    if not variants:
        return []
    conditions = config.get("conditions") or {}
    if not conditions:
        raise CounterbalanceError("counterbalancing.conditions is required when variants are defined")

    by_position = _scenario_by_position(scenarios)
    expected = set(by_position)
    target_refs = {target.get("ref") for target in targets}
    seen_ids: set[str] = set()
    compiled: list[dict] = []

    for raw in variants:
        variant_id = str(raw.get("id") or "").strip()
        if not variant_id or variant_id in seen_ids:
            raise CounterbalanceError("variant ids must be non-empty and unique")
        seen_ids.add(variant_id)
        target_ref = raw.get("target_ref")
        if target_ref not in target_refs:
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


def allocate(settings: dict | None, scenarios: list[dict], targets: list[dict],
             participants: list[dict], count: int) -> list[dict]:
    variants = validate_and_compile(settings, scenarios, targets)
    if not variants:
        return []
    counts = Counter(p.get("variant_id") for p in participants if p.get("variant_id"))
    allocations: list[dict] = []
    for _ in range(count):
        chosen = min(variants, key=lambda v: (counts[v["variant_id"]], v["variant_id"]))
        allocations.append(copy.deepcopy(chosen))
        counts[chosen["variant_id"]] += 1
    return allocations


def balance_report(settings: dict | None, scenarios: list[dict], targets: list[dict],
                   participants: list[dict]) -> dict:
    variants = validate_and_compile(settings, scenarios, targets)
    if not variants:
        return {"configured": False, "valid": True, "participants": len(participants)}

    variant_counts = Counter(p.get("variant_id") for p in participants)
    design_cells: dict[str, Counter] = defaultdict(Counter)
    position_cells: dict[str, Counter] = defaultdict(Counter)
    configured_target_counts = Counter()
    for variant in variants:
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
    if (configured_target_counts and
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
        allocated_targets[variant["target_ref"]] += 1
        for ordinal, scenario_id in enumerate(variant["scenario_order"], start=1):
            condition = variant["assignment"][str(scenario_id)]["condition"]
            allocated_design[str(scenario_id)][condition] += 1
            allocated_positions[str(scenario_id)][str(ordinal)] += 1

    return {
        "configured": True,
        "valid": not warnings,
        "warnings": warnings,
        "variant_counts": {v["variant_id"]: variant_counts[v["variant_id"]] for v in variants},
        "configured_design": {scenario: dict(cells) for scenario, cells in design_cells.items()},
        "configured_positions": {scenario: dict(cells) for scenario, cells in position_cells.items()},
        "allocated_design": {scenario: dict(cells) for scenario, cells in allocated_design.items()},
        "allocated_positions": {scenario: dict(cells) for scenario, cells in allocated_positions.items()},
        "allocated_targets": dict(allocated_targets),
    }
