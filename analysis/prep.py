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
]

UNIT_KEEP = [
    "session_id", "participant_id", "condition", "unit_index", "attempted",
    "complete_raw", "complete_transmitted", "delivery_relative_to_boundary",
    "acknowledgement", "update_claim", "incorporation", "retention",
]


def read_rows(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
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

    # Sessions admitted under dated validity amendments (analysis-time rule;
    # the export's frozen inclusion flags are never rewritten).
    admitted = []
    amendments_path = os.path.join(HERE, "amendments.json")
    if os.path.exists(amendments_path):
        with open(amendments_path, encoding="utf-8") as handle:
            admitted_ids = {a["session_id"] for a in json.load(handle)}
        admitted = [r for r in scenarios if r["session_id"] in admitted_ids
                    and not truthy(r.get("analysis_included"))]
        missing = admitted_ids - {r["session_id"] for r in scenarios}
        if missing:
            print("WARNING: amended session(s) not in export:", sorted(missing))

    print(f"scenarios: {len(scenarios)} rows, {len(included)} included, "
          f"{len(admitted)} amended in")
    write_frame("scenario_level.csv", included, SCENARIO_KEEP)
    write_frame("scenario_level_amended.csv", included + admitted, SCENARIO_KEEP)

    units = read_rows("units.csv")
    keep_ids = {r["session_id"] for r in included}
    amended_ids = keep_ids | {r["session_id"] for r in admitted}
    write_frame("unit_level.csv",
                [u for u in units if u["session_id"] in keep_ids], UNIT_KEEP)
    write_frame("unit_level_amended.csv",
                [u for u in units if u["session_id"] in amended_ids], UNIT_KEEP)

    by_participant = {}
    for r in included:
        by_participant.setdefault(r["participant_id"], []).append(r["condition"])
    complete = sum(1 for v in by_participant.values() if len(v) == 4)
    print(f"participants in frames: {len(by_participant)} ({complete} with all 4 conditions)")


if __name__ == "__main__":
    main()
