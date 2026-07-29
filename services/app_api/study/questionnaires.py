"""Shared questionnaire visibility and required-answer validation."""

from __future__ import annotations


def item_visible(item: dict, answers: dict) -> bool:
    condition = item.get("show_if") or {}
    field = condition.get("field")
    if not field:
        return True
    value = answers.get(field)
    allowed = condition.get("in") or []
    if isinstance(value, list):
        return any(candidate in allowed for candidate in value)
    return value in allowed


def missing_required_answers(items: list[dict], answers: dict) -> list[str]:
    missing = []
    for item in items:
        if item.get("type") == "notice" or not item_visible(item, answers):
            continue
        if not item.get("required"):
            continue
        value = answers.get(item.get("id"))
        answered = value is not None and value != ""
        if item.get("type") == "switch":
            answered = value is True
        elif item.get("type") == "checkbox":
            answered = isinstance(value, list) and bool(value)
        if not answered:
            missing.append(str(item.get("id")))
    return missing
