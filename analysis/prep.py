"""Build model frames from the dataset export.

Reads the dataset_export CSVs from analysis/data/ and writes tidy frames for
models.R to analysis/output/frames/. Only sessions with analysis_included set
survive. Empty cells stay empty (missing != zero, per the data dictionary).

Usage: python3 prep.py
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output", "frames")

SCENARIO_KEEP = [
    "session_id", "participant_id", "condition", "scenario_title",
    "analytical_position", "demonstrated_grounding", "repair_total",
    "repair_post_boundary", "any_repair", "outcome_level",
    "final_account_accuracy", "false_update_confirmation", "duration_s",
    "valid_for_condition_analysis", "post_effort", "post_frustration",
    "post_trust", "post_outcome_confidence", "post_finish_communicating",
    "post_final_response_accurate", "post_misunderstood",
    "post_self_reported_outcome", "post_wanted_to_end",
    # Provenance: which labels are human and which the judge's.
    "coding_label_source",
]

UNIT_KEEP = [
    "session_id", "participant_id", "condition", "unit_index", "attempted",
    "complete_raw", "complete_transmitted", "delivery_relative_to_boundary",
    "acknowledgement", "update_claim", "incorporation", "retention",
    # Distinguishes a stage the coder could not judge from one the
    # transmitted track gated away.
    "grounding_gated_reason",
    # Carried down from the interaction so a unit-level model can hold the
    # same covariates as the scenario-level one.
    "scenario_title", "analytical_position",
]

TURN_EVENT_KEEP = [
    "session_id", "participant_id", "condition", "episode_id", "initiator",
    "overlap_duration_ms", "overlap_200ms_candidate",
    "participant_barge_in_candidate", "assistant_premature_onset_candidate",
    "assistant_stop_latency_ms_candidate",
    "participant_stop_latency_ms_candidate", "verified_overlap",
    "verified_participant_barge_in", "verified_assistant_premature_onset",
    "successful_assistant_yielding", "disruptive_assistant_interruption",
    "assistant_backchannel_onset",
    "verified_assistant_stop_latency_ms", "verified_participant_stop_latency_ms",
    # An episode in a certified session is not the same as a verified
    # episode; without this the two cannot be told apart in a frame.
    "verifier_initials",
]

TURN_GAP_KEEP = [
    "session_id", "participant_id", "condition", "gap_id", "direction",
    "from_speaker", "to_speaker", "gap_duration_ms", "verified_positive_gap",
    "verified_gap_duration_ms",
]


def read_rows(name, required=True):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        if not required:
            return []
        sys.exit(f"missing {path} — copy the dataset_export CSVs into analysis/data/")
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def write_frame(name, rows, columns):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    scenarios = read_rows("scenarios.csv")
    included = [r for r in scenarios if truthy(r.get("analysis_included"))]
    print(f"scenarios: {len(scenarios)} rows, {len(included)} included")
    write_frame("scenario_level.csv", included, SCENARIO_KEEP)

    # Prespecified sensitivity analysis: drop participants with any technically
    # invalid attempt. The verdict comes from technical_status, since
    # valid_for_condition_analysis is also false for never-evaluated captures,
    # and is scoped to the frame — excluding an absent participant removes nothing.
    invalid_participants = {
        row["participant_id"] for row in scenarios
        if str(row.get("technical_status")).strip().lower() == "invalid"
    } & {row["participant_id"] for row in included}
    unevaluated = sorted(
        row["session_id"] for row in scenarios
        if str(row.get("technical_status")).strip().lower() not in
        ("valid", "invalid")
    )
    sensitivity = [row for row in included
                   if row["participant_id"] not in invalid_participants]
    write_frame("scenario_level_sensitivity_complete_technical.csv",
                sensitivity, SCENARIO_KEEP)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sensitivity_complete_technical.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "rule": ("exclude every participant with any analytical attempt "
                     "whose technical_status is invalid"),
            "excluded_participant_ids": sorted(invalid_participants),
            "unevaluated_session_ids": unevaluated,
            "primary_rows": len(included),
            "sensitivity_rows": len(sensitivity),
        }, handle, indent=2, sort_keys=True)

    units = read_rows("units.csv")
    keep_ids = {r["session_id"] for r in included}
    covariates = {r["session_id"]: {"scenario_title": r.get("scenario_title"),
                                    "analytical_position": r.get(
                                        "analytical_position")}
                  for r in included}
    units = [{**u, **covariates.get(u["session_id"], {})} for u in units]
    write_frame("unit_level.csv",
                [u for u in units if u["session_id"] in keep_ids], UNIT_KEEP)
    sensitivity_ids = {row["session_id"] for row in sensitivity}
    write_frame("unit_level_sensitivity_complete_technical.csv",
                [u for u in units if u["session_id"] in sensitivity_ids],
                UNIT_KEEP)

    # Turn-taking frames carry only sessions whose synchronization certified:
    # the method reports timing-derived indicators only when validated, so
    # uncertified candidates must never reach a turn-taking model.
    for name, keep, frame in (("turn_events.csv", TURN_EVENT_KEEP,
                               "turn_events_certified.csv"),
                              ("turn_gaps.csv", TURN_GAP_KEEP,
                               "turn_gaps_certified.csv")):
        rows = read_rows(name, required=False)
        certified = [r for r in rows
                     if truthy(r.get("analysis_included"))
                     and truthy(r.get("valid_for_manual_turn_verification"))
                     and truthy(r.get("crosswalk_complete"))]
        write_frame(frame, certified, keep)
        dropped = {r["session_id"] for r in rows} - {r["session_id"] for r in certified}
        if dropped:
            print(f"  {name}: {len(rows) - len(certified)} row(s) from "
                  f"{len(dropped)} uncertified session(s) excluded")

    by_participant = {}
    for r in included:
        by_participant.setdefault(r["participant_id"], []).append(r["condition"])
    complete = sum(1 for v in by_participant.values() if len(v) == 4)
    print(f"participants in frames: {len(by_participant)} ({complete} with all 4 conditions)")


if __name__ == "__main__":
    main()
