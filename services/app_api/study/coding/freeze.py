"""Freeze and verify the coding materials (method: codebook, prompts, output
schema, model version, decoding settings, and validation rules are frozen on
pilot data before the main dataset is coded, and not revised afterwards)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import prompts
from .agreement import RELIABILITY_THRESHOLDS
from .schema import CONFIDENCE_FLAG_THRESHOLD, SCHEMA_VERSION

_FINGERPRINT_PACKET = {
    "schema": "fingerprint",
    "packet_id": "0" * 16,
    "scenario": {"title": "fingerprint", "unit_definitions": ["a", "b"],
                 "bounded_action": "x", "required_final_account": "y",
                 "outcome_levels": [{"score": 1, "label": "l", "criteria": "c"}],
                 "final_probe_text": "z"},
    "utterances": [{"id": "participant_001", "speaker": "participant",
                    "text": "t", "asr_ok": True}],
    "final_probe_candidates": [],
    "notes_for_coder": "",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def materials_fingerprint() -> dict:
    """Content hashes of everything that defines the coder's behavior other
    than the model itself."""
    return {
        "codebook_sha256": _sha(prompts.codebook_text()),
        "judge_system_sha256": _sha(prompts.judge_system_prompt()),
        "judge_template_sha256": _sha(prompts.judge_user_prompt(_FINGERPRINT_PACKET)),
        "verifier_system_sha256": _sha(prompts.verifier_system_prompt()),
        "schema_version": SCHEMA_VERSION,
        "confidence_flag_threshold": CONFIDENCE_FLAG_THRESHOLD,
        "reliability_thresholds": RELIABILITY_THRESHOLDS,
    }


def _manifest_path(root: Path) -> Path:
    return Path(root) / "freeze_manifest.json"


def freeze(root: Path, decoding: dict, note: str = "") -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "frozen_at_unix_s": time.time(),
        "fingerprint": materials_fingerprint(),
        "decoding": decoding,
        "note": note,
    }
    _manifest_path(root).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def manifest_status(root: Path) -> dict:
    path = _manifest_path(root)
    if not path.exists():
        return {"exists": False, "valid": False, "drift": ["no freeze manifest"]}
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"exists": True, "valid": False, "drift": ["unreadable freeze manifest"]}
    current = materials_fingerprint()
    frozen = manifest.get("fingerprint") or {}
    drift = [f"{key}: frozen={frozen.get(key)!r} current={value!r}"
             for key, value in current.items() if frozen.get(key) != value]
    return {
        "exists": True,
        "valid": not drift,
        "frozen_at_unix_s": manifest.get("frozen_at_unix_s"),
        "decoding": manifest.get("decoding"),
        "drift": drift,
    }
