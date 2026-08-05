"""Flagging, stratified human-review sampling, sheet export/import, and the
final label merge.

Flow (method §LLM-Assisted Coding):
  flags   — schema-invalid output, verifier disagreement/uncertainty, missing
            evidence, low confidence, and deterministic consistency issues
            all trigger blinded human review;
  sample  — at least SAMPLE_FRACTION of valid interactions, stratified by
            condition x scenario, are human-coded IN ADDITION to all flagged
            cases (sampling uses unblinded strata, but the exported sheets
            are the blinded packets);
  import  — filled human sheets become labels/human/<packet_id>.json;
  finalize— human labels take precedence where present, derived labels and
            timing-boundary joins are computed deterministically, and every
            final label records its provenance.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .packets import (classify_delivery_timing, read_index,
                      repair_is_post_boundary, transmitted_completeness)
from .schema import (GROUNDING_STAGES, consistency_issues,
                     derive_scenario_labels, low_confidence_fields,
                     validate_labels)

SAMPLE_FRACTION = 0.25


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _packet(root: Path, packet_id: str) -> dict:
    return json.loads((root / "packets" / f"{packet_id}.json").read_text())


def _meta(root: Path, packet_id: str) -> dict:
    return json.loads((root / "meta" / f"{packet_id}.json").read_text())


def compute_flags(root: Path) -> dict:
    """Per-packet review triggers. Written to review/flags.json."""
    flags: dict[str, list[str]] = {}
    for row in read_index(root):
        pid = row["packet_id"]
        reasons: list[str] = []
        judge = _read_json(root / "labels" / "judge" / f"{pid}.json")
        if judge is None:
            reasons.append("judge_missing")
        else:
            labels = judge.get("labels")
            if labels is None or judge.get("schema_errors"):
                reasons.append("schema_invalid")
            else:
                packet = _packet(root, pid)
                issues = consistency_issues(labels, packet)
                if issues:
                    reasons.append("contradictory_labels")
                low = low_confidence_fields(labels)
                if low:
                    reasons.append("low_confidence")
                outcome = labels.get("outcome") or {}
                account = labels.get("final_account_accuracy") or {}
                if not outcome.get("evidence_utterance_ids"):
                    reasons.append("missing_evidence:outcome")
                if (account.get("value") != "no_final_account"
                        and not account.get("evidence_utterance_ids")):
                    reasons.append("missing_evidence:final_account")
        verifier = _read_json(root / "labels" / "verifier" / f"{pid}.json")
        if verifier is None:
            if judge is not None and "schema_invalid" not in reasons:
                reasons.append("verifier_missing")
        else:
            if verifier.get("disagreements"):
                reasons.append("verifier_disagreement")
            if verifier.get("uncertain"):
                reasons.append("verifier_uncertain")
        if reasons:
            flags[pid] = reasons
    out = root / "review"
    out.mkdir(parents=True, exist_ok=True)
    (out / "flags.json").write_text(json.dumps(flags, indent=2, sort_keys=True))
    return flags


def stratified_sample(root: Path, seed: int, fraction: float = SAMPLE_FRACTION) -> dict:
    """Deterministic per-stratum sample of packets with valid judge labels.
    Strata come from the unblinded index (condition x scenario)."""
    strata: dict[tuple[str, str], list[str]] = {}
    for row in read_index(root):
        if row.get("analysis_included") is not True:
            continue
        pid = row["packet_id"]
        judge = _read_json(root / "labels" / "judge" / f"{pid}.json") or {}
        if judge.get("labels") is None or judge.get("schema_errors"):
            continue
        key = (str(row.get("voice_condition")), str(row.get("scenario_title")))
        strata.setdefault(key, []).append(pid)
    rng = random.Random(seed)
    selected: list[str] = []
    for key in sorted(strata):
        pool = sorted(strata[key])
        take = max(1, math.ceil(fraction * len(pool))) if pool else 0
        selected.extend(rng.sample(pool, take))
    result = {"seed": seed, "fraction": fraction, "selected": sorted(selected),
              "strata_sizes": {" / ".join(k): len(v) for k, v in sorted(strata.items())}}
    out = root / "review"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def export_review(root: Path) -> dict:
    """Write the review queue and one blinded coding sheet per queued packet."""
    flags = _read_json(root / "review" / "flags.json") or {}
    sample = _read_json(root / "review" / "sample.json") or {}
    expansion = _read_json(root / "review" / "reliability_expansion.json") or {}
    queue: dict[str, list[str]] = {}
    for pid, reasons in flags.items():
        queue.setdefault(pid, []).extend(f"flag:{reason}" for reason in reasons)
    for pid in sample.get("selected") or []:
        queue.setdefault(pid, []).append("stratified_sample")
    failed_fields = expansion.get("failed_fields") or []
    for pid in expansion.get("packet_ids") or []:
        queue.setdefault(pid, []).append(
            "reliability_expansion:" + ",".join(failed_fields))
    sheets = root / "review" / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    with (root / "review" / "queue.jsonl").open("w") as handle:
        for pid in sorted(queue):
            handle.write(json.dumps(
                {"packet_id": pid, "reasons": sorted(set(queue[pid]))},
                sort_keys=True) + "\n")
    for pid in sorted(queue):
        sheet_path = sheets / f"{pid}.json"
        if sheet_path.exists():
            continue  # never overwrite a sheet a human may be editing
        packet = _packet(root, pid)
        sheet = {
            "packet_id": pid,
            "instructions": (
                "Blinded human coding sheet. Read coding/codebook.md, then "
                "fill `labels` with the same structure the judge uses (see "
                "any labels/judge/*.json for shape). Set `coder` to your "
                "initials. Do not consult condition information."),
            "coder": "",
            "packet": packet,
            "labels": None,
        }
        sheet_path.write_text(json.dumps(sheet, indent=2, sort_keys=True))
    return {"queued": len(queue), "sheets_dir": str(sheets)}


def expand_for_reliability(root: Path) -> dict:
    """Queue every not-yet-human-coded valid packet when a frozen reliability
    threshold fails. Sheets contain the full label structure, which is a
    conservative superset of reviewing only the affected variable."""
    report = _read_json(root / "agreement" / "report.json")
    if report is None:
        raise FileNotFoundError(
            "run `python -m study.coding agreement` before reliability expansion")
    reliability = report.get("reliability") or {}
    failed = list(reliability.get("failed_fields") or [])
    packet_ids = []
    if failed:
        for row in read_index(root):
            if row.get("analysis_included") is not True:
                continue
            pid = row["packet_id"]
            if not (root / "labels" / "human" / f"{pid}.json").exists():
                packet_ids.append(pid)
    result = {
        "failed_fields": failed,
        "packet_ids": sorted(packet_ids),
        "requires_full_human_review": bool(failed),
        "note": ("Full coding sheets are queued as a conservative superset of "
                 "reviewing only the affected variables."),
    }
    out = root / "review"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reliability_expansion.json").write_text(
        json.dumps(result, indent=2, sort_keys=True))
    return result


def import_human(root: Path) -> dict:
    """Validate filled sheets and store them as human labels."""
    sheets = root / "review" / "sheets"
    out_dir = root / "labels" / "human"
    out_dir.mkdir(parents=True, exist_ok=True)
    imported, pending, invalid = [], [], {}
    for sheet_path in sorted(sheets.glob("*.json")) if sheets.exists() else []:
        sheet = _read_json(sheet_path) or {}
        pid = sheet.get("packet_id") or sheet_path.stem
        labels = sheet.get("labels")
        if labels is None:
            pending.append(pid)
            continue
        packet = _packet(root, pid)
        errors = validate_labels(labels, packet)
        if errors:
            invalid[pid] = errors
            continue
        (out_dir / f"{pid}.json").write_text(json.dumps({
            "packet_id": pid,
            "labels": labels,
            "provenance": {"role": "human", "coder": sheet.get("coder") or "",
                           "sheet": sheet_path.name},
        }, indent=2, sort_keys=True))
        imported.append(pid)
    return {"imported": imported, "pending": pending, "invalid": invalid}


def _label_diff(judge_labels: dict, human_labels: dict) -> list[str]:
    fields: list[str] = []
    j_units = judge_labels.get("units") or []
    h_units = human_labels.get("units") or []
    for i, (j, h) in enumerate(zip(j_units, h_units), start=1):
        for key in ("attempted", "complete_raw", "acknowledgement",
                    "update_claim", "incorporation", "retention"):
            if j.get(key) != h.get(key):
                fields.append(f"units[{i}].{key}")
    if (judge_labels.get("outcome") or {}).get("level") != \
            (human_labels.get("outcome") or {}).get("level"):
        fields.append("outcome.level")
    if (judge_labels.get("final_account_accuracy") or {}).get("value") != \
            (human_labels.get("final_account_accuracy") or {}).get("value"):
        fields.append("final_account_accuracy.value")
    if len(judge_labels.get("repairs") or []) != len(human_labels.get("repairs") or []):
        fields.append("repairs.count")
    return fields


def finalize(root: Path) -> dict:
    """Merge judge/human labels into final per-session labels with derived
    variables and deterministic timing-boundary joins."""
    final_dir = root / "labels" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    written, unresolved = [], []
    for row in read_index(root):
        pid = row["packet_id"]
        judge = _read_json(root / "labels" / "judge" / f"{pid}.json") or {}
        human = _read_json(root / "labels" / "human" / f"{pid}.json")
        judge_labels = judge.get("labels") if not judge.get("schema_errors") else None
        human_labels = (human or {}).get("labels")
        if human_labels is not None:
            labels, source = human_labels, "human"
        elif judge_labels is not None:
            labels, source = judge_labels, "llm_judge"
        else:
            unresolved.append(pid)
            continue
        meta = _meta(root, pid)
        units_enriched = []
        for unit in labels.get("units") or []:
            delivery = unit.get("delivery_utterance_ids") or []
            transmitted = transmitted_completeness(delivery, meta)
            enriched = {
                **unit,
                "complete_transmitted": transmitted["value"],
                "transmitted_content_recall": transmitted["recall"],
                "complete_transmitted_reason": transmitted["reason"],
                "delivery_relative_to_boundary": classify_delivery_timing(
                    delivery, meta),
            }
            if transmitted["value"] == 0:
                # Method rule: downstream grounding stages are missing when a
                # unit was not completely transmitted. The judge coded them
                # against the raw track; the enriched record nulls them.
                for stage in GROUNDING_STAGES:
                    enriched[stage] = None
                enriched["grounding_gated_reason"] = "unit_not_completely_transmitted"
            elif transmitted["value"] is None and delivery:
                enriched["transmitted_proxy"] = "raw"
            units_enriched.append(enriched)
        # Derived labels come from the ENRICHED units so transmitted gating
        # propagates into demonstrated grounding / false update confirmation.
        derived = derive_scenario_labels({**labels, "units": units_enriched})
        repairs_enriched = []
        post_boundary_count = 0
        boundary_known = meta.get("boundary_participant_timeline_ms") is not None
        for move in labels.get("repairs") or []:
            post = repair_is_post_boundary(move.get("utterance_id"), meta)
            if post:
                post_boundary_count += 1
            repairs_enriched.append({**move, "post_boundary": post})
        disagreement = (_label_diff(judge_labels, human_labels)
                        if judge_labels is not None and human_labels is not None
                        else [])
        record = {
            "packet_id": pid,
            "session_id": meta.get("session_id"),
            "labels": labels,
            "units": units_enriched,
            "repairs": repairs_enriched,
            "derived": {
                **derived,
                "repair_post_boundary": (post_boundary_count if boundary_known else None),
                "boundary_kind": meta.get("boundary_kind"),
            },
            "provenance": {
                "source": source,
                "judge_provenance": judge.get("provenance"),
                "human_provenance": (human or {}).get("provenance"),
                "judge_human_disagreement_fields": disagreement,
                "adjudication_recommended": bool(disagreement),
            },
        }
        (final_dir / f"{pid}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True))
        written.append(pid)
    return {"finalized": written, "unresolved": unresolved}
