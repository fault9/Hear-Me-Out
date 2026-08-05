"""Blind pooled descriptives — safe before collection ends.

Reports quality and self-report distributions POOLED ACROSS CONDITIONS from
the scenario-level frame. The condition column is dropped after design
bookkeeping so nothing here can reveal a contrast.

Usage: python3 descriptives.py [--amended]
"""

import argparse
import csv
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

LIKERT = ["post_effort", "post_frustration", "post_trust",
          "post_outcome_confidence", "post_finish_communicating",
          "post_final_response_accurate"]
CATEGORICAL = ["post_misunderstood", "post_self_reported_outcome",
               "post_wanted_to_end"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amended",
        action="store_true",
        help="include sessions admitted by the frozen content amendment",
    )
    args = parser.parse_args()
    frame_name = (
        "scenario_level_amended.csv" if args.amended else "scenario_level.csv")
    frame = os.path.join(HERE, "output", "frames", frame_name)
    with open(frame, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    participants = {r["participant_id"] for r in rows}
    print("analysis frame:", "amended content" if args.amended else "primary")
    per_condition = Counter(r["condition"] for r in rows)
    print(f"sessions: {len(rows)} | participants: {len(participants)}")
    print("design bookkeeping (n per condition, no outcomes):",
          dict(sorted(per_condition.items())))
    for r in rows:  # blind guard: nothing below may group by condition
        del r["condition"]
    print()

    durations = sorted(float(r["duration_s"]) for r in rows if r.get("duration_s"))
    if durations:
        mid = durations[len(durations) // 2]
        print(f"duration_s: median {mid:.0f}, min {durations[0]:.0f}, "
              f"max {durations[-1]:.0f}")
    valid = sum(1 for r in rows if str(r.get("valid_for_condition_analysis")).lower()
                in ("1", "true"))
    print(f"valid_for_condition_analysis: {valid}/{len(rows)}")
    print()

    for key in LIKERT:
        values = [int(float(r[key])) for r in rows
                  if r.get(key) not in (None, "",) and r[key].replace(".", "").isdigit()]
        if not values:
            print(f"{key}: no data")
            continue
        dist = Counter(values)
        mean = sum(values) / len(values)
        bars = " ".join(f"{i}:{dist.get(i, 0)}" for i in range(1, 8))
        print(f"{key}: mean {mean:.2f} | {bars}")
    print()

    for key in CATEGORICAL:
        values = [r[key] for r in rows if r.get(key)]
        if not values:
            print(f"{key}: no data")
            continue
        print(f"{key}:")
        for value, count in Counter(values).most_common():
            print(f"    {count:3d}  {value}")


if __name__ == "__main__":
    main()
