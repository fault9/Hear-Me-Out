"""Build condition-blinded coding packets from immutable session artifacts.

A packet is the ONLY material a coder (LLM judge or blinded human) receives:
ordered utterances (speaker + text), the scenario specification, and
final-probe candidate markers. Everything that could reveal the voice
condition is stripped: timestamps, per-utterance voice modes, route segments,
route switches, overlap/barge-in events, target identifiers, and the raw
session id (packets are keyed by a salted hash).

The unblinded sidecar (`meta/<packet_id>.json`) retains what deterministic
post-processing needs — utterance timings, route regions, the switch time,
condition, and the session id — and is never part of a coder's input.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Iterable

from ..artifacts import load_manifest_artifact

PACKET_SCHEMA = "hmo.coding-packet.v1"
FINAL_PROBE_MATCH_THRESHOLD = 0.55
# Mechanical transmitted-presence rule (codebook §3): content-word recall of
# the raw delivery text within the transmitted-track text of the same
# intervals. Content words are alphanumeric tokens of length >= 4 or numbers.
TRANSMITTED_RECALL_THRESHOLD = float(
    os.environ.get("CODING_TRANSMITTED_RECALL_THRESHOLD", "0.6"))

# Keys that must never appear anywhere in a blinded packet.
FORBIDDEN_KEYS = {
    "voice_mode", "route_segments", "route_switches", "voice_condition",
    "target_ref", "target_speaker_id", "start_ms", "end_ms", "session_id",
    "participant_id", "condition", "schedule", "timing",
}
FORBIDDEN_TOKENS = (
    "stable_natural", "stable_converted", "vc_activation", "vc_deactivation",
)


class BlindingError(RuntimeError):
    pass


def coding_root(data_root: Path, study_id: int) -> Path:
    return Path(data_root) / "coding" / f"study{int(study_id)}"


def load_salt(root: Path) -> str:
    path = root / "blinding_salt.txt"
    if path.exists():
        return path.read_text().strip()
    root.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(16)
    path.write_text(salt + "\n")
    return salt


def packet_id_for(session_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{session_id}".encode()).hexdigest()[:16]


def _load_analysis_artifact(session: dict, data_root: Path, key: str) -> dict | None:
    return load_manifest_artifact(session, data_root, key)


def _scenario_spec(session: dict, scenario_row: dict | None) -> dict:
    snapshot = (session.get("config_snapshot") or {}).get("scenario") or {}
    scenario = snapshot or (scenario_row or {})
    card = scenario.get("scenario_card") or {}
    spec = card.get("analysis_spec") or {}
    return {
        "title": scenario.get("title") or "",
        "study_role": card.get("study_role") or "analytical",
        "unit_definitions": list(spec.get("critical_units") or []),
        "bounded_action": spec.get("bounded_action") or "",
        "required_final_account": spec.get("required_final_account") or "",
        "outcome_levels": list(spec.get("outcome_levels") or []),
        "final_probe_text": card.get("final_account_prompt") or "",
    }


_WORDS = re.compile(r"[a-z0-9']+")


def _normalize(text: str) -> str:
    return " ".join(_WORDS.findall(text.lower()))


def probe_similarity(text: str, probe_text: str) -> float:
    a, b = _normalize(text), _normalize(probe_text)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_final_probe_candidates(utterances: list[dict], probe_text: str,
                                threshold: float = FINAL_PROBE_MATCH_THRESHOLD) -> list[dict]:
    """Participant utterances resembling the required final-readback request,
    best match last-position-first so the protocol-final ask ranks first."""
    scored = []
    for utterance in utterances:
        if utterance.get("speaker") != "participant":
            continue
        score = probe_similarity(str(utterance.get("text") or ""), probe_text)
        if score >= threshold:
            scored.append({"utterance_id": utterance["id"], "similarity": round(score, 3)})
    scored.sort(key=lambda row: (-row["similarity"],))
    return scored


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def assert_blinded(packet: dict) -> None:
    for key, _ in _walk(packet):
        if key in FORBIDDEN_KEYS:
            raise BlindingError(f"forbidden key in blinded packet: {key!r}")
    serialized = json.dumps(packet).lower()
    for token in FORBIDDEN_TOKENS:
        if token in serialized:
            raise BlindingError(f"forbidden token in blinded packet: {token!r}")


def build_packet(session: dict, data_root: Path, salt: str,
                 scenario_row: dict | None = None) -> tuple[dict, dict] | None:
    """Return (blinded_packet, unblinded_meta) or None when the session has no
    dialogue transcript yet (or is a practice session)."""
    spec = _scenario_spec(session, scenario_row)
    if spec["study_role"] == "practice" or session.get("voice_condition") == "practice":
        return None
    dialogue = _load_analysis_artifact(session, data_root, "dialogue_transcript_latest")
    if not dialogue:
        return None

    pid = packet_id_for(str(session["session_id"]), salt)
    utterances_blind = []
    utterance_times = []
    for row in dialogue.get("utterances") or []:
        utterances_blind.append({
            "id": row["id"],
            "speaker": row["speaker"],
            "text": str(row.get("text") or ""),
            "asr_ok": (row.get("speaker") != "participant"
                       or (row.get("text_provenance") or {}).get("asr_status") == "complete"),
        })
        utterance_times.append({
            "id": row["id"],
            "speaker": row["speaker"],
            "start_ms": row.get("start_ms"),
            "end_ms": row.get("end_ms"),
        })

    probe_candidates = find_final_probe_candidates(
        utterances_blind, spec["final_probe_text"])

    packet = {
        "schema": PACKET_SCHEMA,
        "packet_id": pid,
        "scenario": {
            "title": spec["title"],
            "unit_definitions": spec["unit_definitions"],
            "bounded_action": spec["bounded_action"],
            "required_final_account": spec["required_final_account"],
            "outcome_levels": spec["outcome_levels"],
            "final_probe_text": spec["final_probe_text"],
        },
        "utterances": utterances_blind,
        "final_probe_candidates": probe_candidates,
        "notes_for_coder": (
            "Participant utterances with asr_ok=false had incomplete automatic "
            "transcription; treat absent text as unavailable evidence, not as "
            "silence."
        ),
    }
    assert_blinded(packet)

    timing = _load_analysis_artifact(session, data_root, "timing_latest") or {}
    switches = sorted(
        (row for row in timing.get("route_switches") or []
         if row.get("participant_timeline_ms") is not None),
        key=lambda row: float(row["participant_timeline_ms"]),
    )
    switch_ms = float(switches[0]["participant_timeline_ms"]) if switches else None
    transmitted = _load_analysis_artifact(
        session, data_root, "transmitted_transcript_latest")
    transmitted_texts = {
        row["id"]: {"text": str(row.get("text") or ""),
                    "asr_status": (row.get("text_provenance") or {}).get("asr_status")}
        for row in (transmitted or {}).get("utterances") or []
    }
    raw_texts = {
        row["id"]: str(row.get("text") or "")
        for row in dialogue.get("utterances") or []
        if row.get("speaker") == "participant"
    }
    # Switching conditions anchor the boundary at the executed route switch;
    # stable conditions use the matched analysis checkpoint so every session
    # carries the same before/after decomposition (method: the 45 s point is
    # retained as a matched checkpoint without changing stable routes).
    validity = ((session.get("artifact_manifest") or {}).get("analysis") or {}).get(
        "technical_validity_summary") or {}
    checkpoint_s = validity.get("analysis_checkpoint_s")
    if switch_ms is not None:
        boundary_ms, boundary_kind = switch_ms, "switch"
    elif checkpoint_s is not None:
        boundary_ms, boundary_kind = float(checkpoint_s) * 1000.0, "checkpoint"
    else:
        boundary_ms, boundary_kind = 45_000.0, "checkpoint_default"
    meta = {
        "packet_id": pid,
        "session_id": session["session_id"],
        "participant_id": session.get("participant_id"),
        "study_id": session.get("study_id"),
        "scenario_id": session.get("scenario_id"),
        "scenario_order": session.get("scenario_order"),
        "scenario_title": spec["title"],
        "voice_condition": session.get("voice_condition"),
        "switch_participant_timeline_ms": switch_ms,
        "boundary_participant_timeline_ms": boundary_ms,
        "boundary_kind": boundary_kind,
        "route_switches": switches,
        "utterance_times": utterance_times,
        "utterance_texts_raw": raw_texts,
        "transmitted": {
            "available": transmitted is not None,
            "status": (transmitted or {}).get("status"),
            "alignment": (transmitted or {}).get("alignment"),
            "analysis_id": (transmitted or {}).get("analysis_id"),
            "texts": transmitted_texts,
        },
        "dialogue_analysis_id": dialogue.get("analysis_id"),
        "dialogue_status": dialogue.get("status"),
    }
    return packet, meta


def write_packets(sessions: list[dict], data_root: Path, study_id: int,
                  scenarios_by_id: dict[str, dict] | None = None) -> dict:
    """Build and persist packets for every analytical session that has a
    dialogue transcript. Returns a summary dict."""
    root = coding_root(data_root, study_id)
    (root / "packets").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    salt = load_salt(root)
    index_rows, skipped = [], []
    for session in sessions:
        session_id = str(session.get("session_id") or "")
        if ("analysis_included" in session
                and session.get("analysis_included") is not True):
            reasons = session.get("analysis_exclusion_reasons") or ["not_included"]
            skipped.append({
                "session_id": session_id,
                "reason": "analysis_excluded:" + ";".join(map(str, reasons)),
            })
            continue
        scenario_row = (scenarios_by_id or {}).get(str(session.get("scenario_id")))
        built = build_packet(session, data_root, salt, scenario_row)
        if built is None:
            reason = ("practice" if session.get("voice_condition") == "practice"
                      else "no_dialogue_transcript")
            skipped.append({"session_id": session.get("session_id"), "reason": reason})
            continue
        packet, meta = built
        (root / "packets" / f"{packet['packet_id']}.json").write_text(
            json.dumps(packet, indent=2, sort_keys=True))
        (root / "meta" / f"{packet['packet_id']}.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True))
        index_rows.append({
            "packet_id": packet["packet_id"],
            "session_id": session["session_id"],
            "voice_condition": session.get("voice_condition"),
            "scenario_title": meta["scenario_title"],
            "scenario_order": session.get("scenario_order"),
            "analysis_included": session.get("analysis_included", True),
            "technical_status": (session.get("technical_validity") or {}).get("status"),
        })
    with (root / "index.jsonl").open("w") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"written": len(index_rows), "skipped": skipped, "root": str(root)}


def read_index(root: Path) -> list[dict]:
    path = root / "index.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def classify_delivery_timing(delivery_utterance_ids: list[str], meta: dict) -> str | None:
    """Deterministic before/after/across-boundary classification (codebook §3).

    The boundary is the executed route switch for switching conditions and the
    matched analysis checkpoint for stable conditions, so every session carries
    the field on the same decomposition. Returns None when the utterances
    cannot be resolved.
    """
    if not delivery_utterance_ids:
        return None
    times = {row["id"]: row for row in meta.get("utterance_times") or []}
    try:
        starts = [float(times[uid]["start_ms"]) for uid in delivery_utterance_ids]
        ends = [float(times[uid]["end_ms"]) for uid in delivery_utterance_ids]
    except (KeyError, TypeError, ValueError):
        return None
    boundary = meta.get("boundary_participant_timeline_ms")
    if boundary is None:
        return None
    first, last = min(starts), max(ends)
    if last <= boundary:
        return "before_transition"
    if first >= boundary:
        return "after_transition"
    return "across_transition"


_CONTENT_WORD = re.compile(r"[a-z]{4,}|[0-9]+")


def content_tokens(text: str) -> set[str]:
    return set(_CONTENT_WORD.findall(text.lower()))


def transmitted_completeness(delivery_utterance_ids: list[str], meta: dict,
                             threshold: float = TRANSMITTED_RECALL_THRESHOLD) -> dict:
    """Mechanical `complete_transmitted` code (codebook §3).

    Returns {"value": 0|1|None, "recall": float|None, "reason": str|None}.
    1  — content-word recall of the raw delivery text within the transmitted
         text of the same intervals reaches the threshold;
    0  — transmitted ASR completed on every delivery interval but recall
         stayed below the threshold (observed absence);
    None — inapplicable/unobservable, with the reason recorded.
    """
    if not delivery_utterance_ids:
        return {"value": None, "recall": None,
                "reason": "unit_not_completely_delivered"}
    transmitted = meta.get("transmitted") or {}
    if not transmitted.get("available"):
        return {"value": None, "recall": None,
                "reason": "transmitted_transcript_unavailable"}
    raw_texts = meta.get("utterance_texts_raw") or {}
    raw = content_tokens(" ".join(raw_texts.get(uid, "")
                                  for uid in delivery_utterance_ids))
    if not raw:
        return {"value": None, "recall": None, "reason": "no_content_tokens"}
    rows = [transmitted.get("texts", {}).get(uid) for uid in delivery_utterance_ids]
    transmitted_text = " ".join((row or {}).get("text") or "" for row in rows)
    recall = len(raw & content_tokens(transmitted_text)) / len(raw)
    if recall >= threshold:
        return {"value": 1, "recall": round(recall, 3), "reason": None}
    if any(row is None or row.get("asr_status") != "complete" for row in rows):
        return {"value": None, "recall": round(recall, 3),
                "reason": "transmitted_asr_incomplete"}
    return {"value": 0, "recall": round(recall, 3), "reason": None}


def repair_is_post_boundary(utterance_id: str, meta: dict) -> bool | None:
    times = {row["id"]: row for row in meta.get("utterance_times") or []}
    boundary = meta.get("boundary_participant_timeline_ms")
    if boundary is None:
        return None
    row = times.get(utterance_id)
    try:
        return float(row["start_ms"]) >= float(boundary)
    except (KeyError, TypeError, ValueError):
        return None
