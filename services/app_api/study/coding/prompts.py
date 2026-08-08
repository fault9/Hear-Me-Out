"""Judge and verifier prompt construction.

Rendering is deterministic (packet payloads sorted, no timestamps) so the
freeze manifest's hashes of the system prompts and of a fixed-packet rendering
detect any post-freeze drift. The label skeleton keeps a fixed field order
instead: units and repairs precede the outcome that follows from them.
"""

from __future__ import annotations

import json
from pathlib import Path

CODEBOOK_PATH = Path(__file__).with_name("codebook.md")

# Keys of a unit's evidence and confidence maps, spelled out in the skeleton
# because the consistency checks look evidence up under exactly these names.
UNIT_LABELS = ("attempted", "complete_raw", "acknowledgement", "update_claim",
               "incorporation", "retention")


def codebook_text() -> str:
    return CODEBOOK_PATH.read_text()


def _labels_skeleton(n_units: int) -> dict:
    unit = {
        "unit_index": 0,
        "attempted": "0|1",
        "complete_raw": "0|1",
        "delivery_utterance_ids": ["participant utterance ids: the minimal set completing the unit"],
        "acknowledgement": "0|1|null",
        "update_claim": "0|1|null",
        "incorporation": "0|1|null",
        "retention": "0|1|null",
        "evidence_utterance_ids": {label: ["utterance ids grounding this label"]
                                   for label in UNIT_LABELS},
        "confidence": {label: "number in [0,1]" for label in UNIT_LABELS},
    }
    return {
        "units": [dict(unit, unit_index=i + 1) for i in range(n_units)],
        "repairs": [{
            "utterance_id": "participant utterance id",
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
            "evidence_utterance_ids": ["utterance ids"],
            "rationale": "2-4 sentences applying the level criteria and codebook rules",
            "confidence": "number in [0,1]",
        },
        "final_account_accuracy": {
            "value": "accurate|partially_accurate|inaccurate|no_final_account",
            "evidence_utterance_ids": ["utterance ids"],
            "confidence": "number in [0,1]",
        },
        "access_flags": [{
            "type": "truncation|abandonment|failed_floor_acquisition|premature_task_closure",
            "utterance_id": "utterance id",
            "confidence": "number in [0,1]",
        }],
        "notes": "optional coder notes, or an empty string",
    }


def _shape_rules(n_units: int) -> str:
    """The contract the skeleton cannot express on its own."""
    return (
        "Shape rules:\n"
        f"- `units` holds exactly {n_units} objects, with unit_index "
        f"1..{n_units}.\n"
        "- Every quoted value in the structure above is a description of the "
        "type expected there. Replace each one with a value of that type; "
        "never copy a description into your output.\n"
        "- Emit each value as JSON of the type described: numbers unquoted "
        "(1, 0, 0.92 — never \"1\" or \"0.92\") and null unquoted (never "
        "\"null\"). Only utterance ids, category, value, trouble, rationale "
        "and notes are strings.\n"
        "- The single object shown inside `repairs` and inside `access_flags` "
        "is an element template, not an example of expected output. Return an "
        "empty list when nothing in this interaction qualifies; do not add an "
        "entry to avoid an empty list.\n"
        "- `delivery_utterance_ids` is an empty list when complete_raw is not "
        "1.\n"
        "- The keys of a unit's `evidence_utterance_ids` and `confidence` are "
        "label names spelled exactly as shown. Include a key only for a label "
        "you set to a non-null value.\n"
        "- Emit the top-level fields in the order shown: settle the units and "
        "repairs first, then let the outcome and final-account labels follow "
        "from them.\n"
    )


def judge_system_prompt() -> str:
    """Role, then the codebook, then the task. Instructions placed after the
    long reference they govern are followed more closely than instructions
    the reference is appended to."""
    return (
        "You are a blinded conversation-analysis coder for a spoken human-AI "
        "interaction study. Apply the codebook below exactly.\n\n"
        "==== CODEBOOK ====\n" + codebook_text() + "\n\n"
        "==== TASK ====\n"
        "You receive a de-identified dialogue transcript, "
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
        "Four codebook rules are easy to pass over; check each one against "
        "your labels before emitting them. A unit counts as delivered only "
        "where the participant states its content, never where the assistant "
        "volunteered it and the participant confirmed. Incorporation requires "
        "the assistant to operationally use the content; acknowledging or "
        "repeating it is not incorporation. The top outcome level requires "
        "the bounded action and the critical units correctly present in the "
        "final state of the record, and wrong-content replacement caps the "
        "level below it. Repairs are read from every participant utterance in "
        "order: overt corrections such as \"no\" or \"it should have said X\", "
        "restarts, and re-confirmations after trouble all count, so an empty "
        "list is right only where none occurred.\n\n"
        "Cite utterance ids as evidence for every non-null label. When "
        "evidence is genuinely ambiguous, choose the codebook-consistent "
        "label and lower your confidence rather than inventing certainty. "
        "Respond with a single JSON object matching the requested structure "
        "— no prose outside the JSON."
    )


def judge_user_prompt(packet: dict) -> str:
    n_units = len(packet.get("scenario", {}).get("unit_definitions") or [])
    note = str(packet.get("notes_for_coder") or "").strip()
    return (
        "Code the following interaction.\n\n"
        "==== SCENARIO SPECIFICATION ====\n"
        + json.dumps(packet["scenario"], indent=2, sort_keys=True)
        + "\n\n==== TRANSCRIPT (participant raw + assistant) ====\n"
        + json.dumps(packet["utterances"], indent=2, sort_keys=True)
        + "\n\n==== TRANSMITTED PARTICIPANT SPEECH (same utterance ids) ====\n"
        + json.dumps(packet.get("transmitted_utterances") or [], indent=2, sort_keys=True)
        + (f"\n\nNote: {note}" if note else "")
        + "\n\nReturn ONLY a JSON object with this structure:\n"
        + json.dumps(_labels_skeleton(n_units), indent=2)
        + "\n\n" + _shape_rules(n_units)
    )


def label_field_paths(labels: dict) -> list[str]:
    """Every field the verifier owes a verdict on. Left to choose for itself
    it checks a handful, and an unexamined label reads downstream exactly like
    an unchallenged one."""
    paths = [f"units[{unit.get('unit_index')}].{label}"
             for unit in labels.get("units") or []
             for label in UNIT_LABELS if unit.get(label) is not None]
    paths += [f"repairs[{i}].category"
              for i, _ in enumerate(labels.get("repairs") or [])]
    return paths + ["repairs", "outcome.level", "final_account_accuracy.value"]


def verifier_system_prompt() -> str:
    return (
        "You are an adversarial verifier for coded labels in a conversation-"
        "analysis study. The codebook the coder applied follows; judge "
        "against it exactly.\n\n"
        "==== CODEBOOK ====\n" + codebook_text() + "\n\n"
        "==== TASK ====\n"
        "You receive the same blinded transcript and scenario "
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
        "\"summary\": \"one or two sentences\"}. The user message lists the "
        "fields to check: return exactly one check for each, copying the "
        "field string verbatim. The bare field `repairs` asks whether the set "
        "of repair moves is complete, so disagree there when the participant "
        "made a move the coder did not record. Write each verdict as one of "
        "the lowercase words agree, disagree, or uncertain — no other spelling "
        "is read. Every disagree or uncertain check must carry a note naming "
        "the codebook rule or the utterance the label fails against."
    )


def verifier_user_prompt(packet: dict, labels: dict) -> str:
    return (
        "==== SCENARIO SPECIFICATION ====\n"
        + json.dumps(packet["scenario"], indent=2, sort_keys=True)
        + "\n\n==== TRANSCRIPT ====\n"
        + json.dumps(packet["utterances"], indent=2, sort_keys=True)
        + "\n\n==== CODER LABELS TO VERIFY ====\n"
        + json.dumps(labels, indent=2, sort_keys=True)
        + "\n\n==== FIELDS TO CHECK (one check each, verbatim) ====\n"
        + "\n".join(label_field_paths(labels))
    )
