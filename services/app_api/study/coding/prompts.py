"""Judge and verifier prompt construction.

Rendering is deterministic (sorted keys, no timestamps): the freeze manifest
records SHA-256 hashes of the system prompts and of a fixed-packet rendering,
so any post-freeze drift is detectable.
"""

from __future__ import annotations

import json
from pathlib import Path

CODEBOOK_PATH = Path(__file__).with_name("codebook.md")


def codebook_text() -> str:
    return CODEBOOK_PATH.read_text()


def _labels_skeleton(n_units: int) -> dict:
    unit = {
        "unit_index": 0,
        "attempted": "0|1",
        "complete_raw": "0|1",
        "delivery_utterance_ids": ["participant_00X, minimal set completing the unit"],
        "acknowledgement": "0|1|null",
        "update_claim": "0|1|null",
        "incorporation": "0|1|null",
        "retention": "0|1|null",
        "evidence_utterance_ids": {"<label>": ["utterance ids grounding each non-null label"]},
        "confidence": {"<label>": "number in [0,1] for each labeled field"},
    }
    return {
        "units": [dict(unit, unit_index=i + 1) for i in range(n_units)],
        "repairs": [{
            "utterance_id": "participant_00X",
            "category": "repetition|reformulation|clarification|explicit_correction|"
                        "repeated_confirmation|restart_after_cutoff|floor_recovery",
            "trouble": "one line: the trouble this move addresses",
            "confidence": "number in [0,1]",
        }],
        "final_probe": {
            "utterance_id": "participant utterance carrying the required final-readback request, or null",
            "spontaneous_final_account_utterance_id": "assistant utterance with an earlier spontaneous final account, or null",
        },
        "outcome": {
            "level": "integer level from the scenario's outcome levels",
            "evidence_utterance_ids": ["..."],
            "rationale": "2-4 sentences applying the level criteria and codebook rules",
            "confidence": "number in [0,1]",
        },
        "final_account_accuracy": {
            "value": "accurate|partially_accurate|inaccurate|no_final_account",
            "evidence_utterance_ids": ["..."],
            "confidence": "number in [0,1]",
        },
        "access_flags": [{
            "type": "truncation|abandonment|failed_floor_acquisition|premature_task_closure",
            "utterance_id": "...",
            "confidence": "number in [0,1]",
        }],
        "notes": "optional coder notes",
    }


def judge_system_prompt() -> str:
    return (
        "You are a blinded conversation-analysis coder for a spoken human-AI "
        "interaction study. You receive a de-identified dialogue transcript, "
        "the transmitted participant speech under the same utterance ids, and "
        "the scenario specification including the assistant's system prompt. "
        "You do not know, and must not speculate about, any audio-processing "
        "condition of the session.\n\n"
        "No final-readback markers are supplied: identify the participant's "
        "first request for a final summary yourself, and take the first "
        "assistant response to it. Select an earlier assistant turn only when "
        "it clearly constitutes a spontaneous final account of the recorded "
        "information, decision, and next step; if neither occurs, code "
        "delayed retention as null.\n\n"
        "Do not infer overlap, barge-in, premature onset, stop latency, "
        "response gaps, or any other timing outcome from the transcript. "
        "Those are derived separately from synchronized audio and system "
        "logs, and the transcript carries no timing information.\n\n"
        "Apply the codebook below exactly. Cite utterance ids as evidence for "
        "every non-null label. When evidence is genuinely ambiguous, choose "
        "the codebook-consistent label and lower your confidence rather than "
        "inventing certainty. Respond with a single JSON object matching the "
        "requested structure — no prose outside the JSON.\n\n"
        "==== CODEBOOK ====\n" + codebook_text()
    )


def judge_user_prompt(packet: dict) -> str:
    n_units = len(packet.get("scenario", {}).get("unit_definitions") or [])
    return (
        "Code the following interaction.\n\n"
        "==== SCENARIO SPECIFICATION ====\n"
        + json.dumps(packet["scenario"], indent=2, sort_keys=True)
        + "\n\n==== TRANSCRIPT (participant raw + assistant) ====\n"
        + json.dumps(packet["utterances"], indent=2, sort_keys=True)
        + "\n\n==== TRANSMITTED PARTICIPANT SPEECH (same utterance ids) ====\n"
        + json.dumps(packet.get("transmitted_utterances") or [], indent=2, sort_keys=True)
        + "\n\nNote: " + str(packet.get("notes_for_coder") or "")
        + "\n\nReturn ONLY a JSON object with this structure (values shown as "
          "type descriptions):\n"
        + json.dumps(_labels_skeleton(n_units), indent=2, sort_keys=True)
    )


def verifier_system_prompt() -> str:
    return (
        "You are an adversarial verifier for coded labels in a conversation-"
        "analysis study. You receive the same blinded transcript and scenario "
        "specification as the original coder, plus the coder's labels. Your "
        "job is to try to REFUTE each label against the cited evidence and "
        "the codebook. Do not defer to the coder: check every label "
        "independently. Pay particular attention to: units credited as "
        "delivered when the participant only assented to assistant-"
        "presupposed content; incorporation credited despite inverted or "
        "contradicted content; outcome levels that ignore the on-call "
        "determination, wrong-content replacement, or top-level content "
        "rules; and repair moves that are actually first deliveries or the "
        "protocol-required readback request.\n\n"
        "Respond with a single JSON object: {\"checks\": [{\"field\": "
        "\"<label path, e.g. units[1].incorporation or outcome.level>\", "
        "\"verdict\": \"agree|disagree|uncertain\", \"note\": \"one line\"}], "
        "\"summary\": \"one or two sentences\"}. Include one check per unit "
        "label, one for each repair move, one for outcome.level, and one for "
        "final_account_accuracy.value.\n\n"
        "==== CODEBOOK ====\n" + codebook_text()
    )


def verifier_user_prompt(packet: dict, labels: dict) -> str:
    return (
        "==== SCENARIO SPECIFICATION ====\n"
        + json.dumps(packet["scenario"], indent=2, sort_keys=True)
        + "\n\n==== TRANSCRIPT ====\n"
        + json.dumps(packet["utterances"], indent=2, sort_keys=True)
        + "\n\n==== CODER LABELS TO VERIFY ====\n"
        + json.dumps(labels, indent=2, sort_keys=True)
    )
