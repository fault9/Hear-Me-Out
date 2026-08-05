"""Reproduce the speech-boundary metrics frozen for pilot P02001.

Run from the repository root:
    python3 docs/study-artifacts/timing-validation/score_reviewed_textgrids.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAME_S = 0.020
OVERLAP_MINIMUM_S = 0.200
BOUNDARY_MATCH_MAXIMUM_S = 0.250

FILES = {
    "vc_activation": HERE / "annotations" / "P02001_vc_activation_review.TextGrid",
    "vc_deactivation": HERE / "annotations" / "P02001_vc_deactivation_review.TextGrid",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_textgrid(path: Path) -> tuple[float, dict[str, list[tuple[float, float]]]]:
    text = path.read_text()
    xmax_match = re.search(r"(?m)^xmax = ([0-9.]+)", text)
    if xmax_match is None:
        raise ValueError(f"TextGrid has no xmax: {path}")
    starts = list(re.finditer(r"(?m)^    item \[\d+\]:\s*$", text))
    tiers: dict[str, list[tuple[float, float]]] = {}
    interval_pattern = re.compile(
        r"intervals \[\d+\]:\s*\n"
        r"\s*xmin = ([0-9.]+)\s*\n"
        r"\s*xmax = ([0-9.]+)\s*\n"
        r'\s*text = "([^"]*)"'
    )
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start.start():end]
        name_match = re.search(r'name = "([^"]+)"', block)
        if name_match is None:
            continue
        tiers[name_match.group(1)] = [
            (float(match.group(1)), float(match.group(2)))
            for match in interval_pattern.finditer(block)
            if match.group(3).strip()
        ]
    return float(xmax_match.group(1)), tiers


def _activity(intervals: list[tuple[float, float]], xmax: float) -> list[bool]:
    frames = math.ceil(xmax / FRAME_S)
    return [
        any(start <= index * FRAME_S + FRAME_S / 2 < end
            for start, end in intervals)
        for index in range(frames)
    ]


def _classification(reference: list[bool], candidate: list[bool]) -> dict[str, float]:
    true_positive = sum(a and b for a, b in zip(reference, candidate))
    false_positive = sum(not a and b for a, b in zip(reference, candidate))
    false_negative = sum(a and not b for a, b in zip(reference, candidate))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }


def _overlaps(first: list[tuple[float, float]],
              second: list[tuple[float, float]]) -> list[tuple[float, float]]:
    rows = []
    for first_start, first_end in first:
        for second_start, second_end in second:
            start, end = max(first_start, second_start), min(first_end, second_end)
            if end - start >= OVERLAP_MINIMUM_S:
                rows.append((start, end))
    return rows


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _boundary_errors(reference: list[tuple[float, float]],
                     candidate: list[tuple[float, float]]) -> dict[str, float | int]:
    errors = []
    for boundary_index in (0, 1):
        for interval in reference:
            error = min(abs(interval[boundary_index] - row[boundary_index])
                        for row in candidate)
            if error <= BOUNDARY_MATCH_MAXIMUM_S:
                errors.append(error * 1000)
    return {
        "matched_boundaries": len(errors),
        "manual_boundaries": len(reference) * 2,
        "median_absolute_error_ms": statistics.median(errors),
        "p95_absolute_error_ms": _percentile(errors, 0.95),
    }


def score(path: Path) -> dict:
    xmax, tiers = _parse_textgrid(path)
    result: dict[str, object] = {"textgrid_sha256": _sha256(path)}
    for speaker in ("participant", "assistant"):
        manual = tiers[f"{speaker}_MANUAL_EDIT"]
        automatic = tiers[f"{speaker}_HMO_AUTO_do_not_edit"]
        result[speaker] = {
            "frame": _classification(
                _activity(manual, xmax), _activity(automatic, xmax)),
            "boundaries": _boundary_errors(manual, automatic),
        }
    manual_overlap = _overlaps(
        tiers["participant_MANUAL_EDIT"], tiers["assistant_MANUAL_EDIT"])
    automatic_overlap = _overlaps(
        tiers["participant_HMO_AUTO_do_not_edit"],
        tiers["assistant_HMO_AUTO_do_not_edit"])
    result["overlap_200ms"] = _classification(
        _activity(manual_overlap, xmax), _activity(automatic_overlap, xmax))
    return result


if __name__ == "__main__":
    print(json.dumps({condition: score(path) for condition, path in FILES.items()},
                     indent=2, sort_keys=True))
