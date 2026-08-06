# Scenario Coding Codebook

Version: 1.0.0 (pre-freeze draft — frozen by `python -m study.coding freeze`,
which records this file's SHA-256 in the freeze manifest. After freezing, any
edit invalidates the manifest and the runner refuses to code non-pilot data.)

This codebook operationalizes the Measures section of the method for
LLM-assisted and human coding. The coder (LLM judge or blinded human) receives
only: the de-identified dialogue transcript (speaker + text, in order, without
timestamps, route, or condition information), the transmitted participant
speech under the same utterance ids, and the scenario specification (critical
units, bounded action, outcome levels, required final account, and the
assistant's system prompt). No final-readback markers are supplied — the coder
identifies the final-summary request itself (Section 4). Condition labels,
target-voice identifiers, questionnaire responses, timing-derived variables,
and audio are withheld.

## 1. Units of analysis

Each analytical scenario has exactly two prespecified critical-information
units (from the study template's `analysis_spec.critical_units`). All unit
codes are assigned per unit, per interaction.

## 2. Zero versus missing

- `0` means an observed opportunity in which the criterion failed.
- `null` (missing) means the stage was unobservable or inapplicable.
- Downstream grounding stages (acknowledgement, update claim, incorporation,
  retention) are `null` when the unit was not completely delivered.
- False update confirmation is `null` when no update claim occurred.
- Timing-conditional outcomes are `null` when the initiating event did not
  occur.

## 3. Delivery codes (per unit)

- `attempted` (0/1): the participant made any identifiable attempt to convey
  the unit, however partial.
- `complete_raw` (0/1): the unit's full propositional content appears in the
  participant's (raw-track) speech, in one utterance or accumulated across
  utterances. Paraphrase counts; the specific content must be present (e.g.,
  "left handlebar", not just "he hit me", when the unit specifies the
  handlebar).
- `delivery_utterance_ids`: the minimal ordered set of participant utterances
  that jointly complete the unit (empty when `complete_raw` = 0). The first
  and last of these anchor the deterministic before/after/across-transition
  classification, which is computed outside the coder from the timing
  artifact.
- `complete_transmitted`: computed mechanically, never judged. The unit's
  delivery utterances are located in the transmitted-track transcript (same
  interval ids as the raw transcript), and the content-word recall of the
  raw delivery text within the transmitted text is compared to a frozen
  threshold (content words: alphabetic tokens of length >= 4 or numbers;
  threshold 0.6). Recall at or above threshold codes 1; complete transmitted
  ASR with recall below threshold codes 0 (observed absence); an unavailable
  transmitted transcript or failed transmitted ASR codes `null` with the
  reason recorded. The recall value is retained for audit. When a unit codes
  0, its downstream grounding stages are set to `null` (Section 2); when it
  codes `null`, `complete_raw` acts as the gating proxy and provenance
  records `transmitted_proxy: raw`.

**Participant-stated rule.** A unit counts as delivered only when the
participant states its content. Assistant-presupposed content that the
participant merely assents to ("yeah", "that's right" after the assistant
volunteers the fact) does NOT count as delivery of the unit.

## 4. Grounding codes (per completely delivered unit; else null)

- `acknowledgement` (0/1): the assistant verbally registers the unit
  (repetition, confirmation, explicit acceptance).
- `update_claim` (0/1): the assistant explicitly claims to have recorded,
  updated, corrected, or accepted the unit or the record it concerns.
- `incorporation` (0/1): the assistant's subsequent reasoning, decisions, or
  statements operationally use the unit's content correctly. Inverted or
  contradicted content (e.g., recording "no other person involved" after the
  participant stated another cyclist hit them) is NOT incorporation.
- `retention` (0/1): the unit's content is correctly present in the
  assistant's response to the final-summary request. The coder identifies the
  participant's FIRST request for a final summary and takes the FIRST
  assistant response to it. An earlier assistant turn may be used only when it
  clearly constitutes a spontaneous final account of the recorded information,
  decision, and next step. When neither occurs, `retention` is `null` and
  `final_probe.utterance_id` is null.
- Derived (not judged directly):
  - acknowledgement without uptake = acknowledgement=1 and incorporation=0;
  - false update confirmation (per unit) = update_claim=1 and
    (incorporation=0 or retention=0);
  - scenario-level demonstrated grounding = both units have incorporation=1
    and retention=1.

## 5. Repair moves

A repair move is one continuous participant contribution oriented to
resolving one identifiable trouble in delivery, understanding, recording, or
action. Categories (mutually exclusive per move):

- `repetition` — re-saying substantially the same content after trouble;
- `reformulation` — re-saying with changed wording/structure;
- `clarification` — answering or issuing a clarification about prior content;
- `explicit_correction` — overtly correcting the assistant's version
  ("no, it was building 16");
- `repeated_confirmation` — re-confirming already-established content after
  trouble;
- `restart_after_cutoff` — restarting a contribution interrupted mid-delivery;
- `floor_recovery` — re-taking the floor to resume an interrupted delivery.

Exclusions: ordinary elaboration, the first delivery of a unit, and the first
protocol-required final-readback request. Escalation and withdrawal are
outcomes, not repair moves. Post-switch classification is computed outside
the coder from utterance timing.

## 6. Task outcome (ordinal 1–4)

Coded against the scenario's `outcome_levels` criteria. General rules:

- **On-call determination rule.** A clearly stated determination on the call
  ("your fail should be upgraded to pass") counts as the decision even when
  the assistant appends pro-forma confirmation or processing language
  ("the office will confirm", "processed within 5 days"). Deferral WITHOUT a
  stated determination is not a decision.
- **Wrong-content replacement rule.** A record/statement "replaced" with
  content that contradicts or omits the participant's critical units is not a
  corrected record: score at most the recording-level (Level 2), and the top
  level is unavailable on a wrong statement.
- **Top-level content rule.** Level 4 requires the bounded action AND the
  critical units correctly present in the final state of the record (per the
  scenario's criteria).
- Critical-information uptake and final-account accuracy are coded
  separately (Sections 4 and 7), never folded into the ordinal outcome.

## 7. Final-account accuracy

Coded on the assistant's response to the final probe (or earlier spontaneous
final account): `accurate` (all asserted record contents and decisions match
what actually happened in the conversation), `partially_accurate` (some
match, some wrong/omitted), `inaccurate` (substantially wrong),
`no_final_account` (no readback produced despite the probe, or no probe
occurred).

## 8. Interactional-access candidate flags

The coder may nominate, per utterance: `truncation` (participant contribution
audibly/textually cut off), `abandonment` (delivery attempt abandoned),
`failed_floor_acquisition` (participant tries and fails to take the floor),
`premature_task_closure` (assistant closes the task/call while the
participant is still pursuing the aim). These are candidate nominations only;
timing-derived access indicators are computed and verified outside the coder.

## 9. Evidence and confidence

Every non-null label must cite the utterance id(s) that ground it. Every
label carries a confidence in [0,1]. Labels with confidence < 0.8, labels
without evidence, verifier disagreement, schema-invalid output, or
deterministic consistency violations are flagged for blinded human review.

## 10. Self-report mapping (analysis-side, not coded)

`self_reported_outcome` "fully achieved the goal" corresponds to aim #1 on
ranked-aim cards (S1–S3) and to both aim items on the joint-aim card (S4);
"partly achieved" corresponds to lower rungs / one of two items.
