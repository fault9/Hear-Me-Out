"""Pooled descriptives, blind by default.

Reports the coding outcomes and self-report distributions POOLED ACROSS
CONDITIONS. Condition appears only in design bookkeeping and data-availability
counts, which say how much data exists rather than what it shows, so nothing
here can reveal a contrast.

    python3 descriptives.py                  pooled only (safe any time)
    python3 descriptives.py --by-condition   repeats every section per
                                             condition, for the runs that are
                                             already unblinded

The flag exists so that the tables beside a model estimate come from the same
frames the model used, rather than being assembled separately. It is gated the
way models.R is: run.sh passes it only for exploratory and confirmatory runs.
"""

import csv
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(HERE, "output", "frames")
DATA = os.path.join(HERE, "data")

LIKERT = ["post_effort", "post_frustration", "post_trust",
          "post_outcome_confidence", "post_finish_communicating",
          "post_final_response_accurate"]
CATEGORICAL = ["post_misunderstood", "post_self_reported_outcome",
               "post_wanted_to_end"]


def _read(directory, name):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _coded(row, field):
    return row.get(field) in ("0", "1")


def _rate(rows, field):
    """(hits, evaluable) — a blank is unevaluable, never a zero."""
    coded = [r for r in rows if _coded(r, field)]
    return sum(1 for r in coded if r[field] == "1"), len(coded)


def _line(label, rows, field, width=34):
    hits, total = _rate(rows, field)
    share = f"{100 * hits / total:5.1f}%" if total else "    - "
    print(f"  {label:<{width}} {hits:3d} / {total:3d}  {share}")


def _cross(rows, first, second):
    pairs = [(r[first], r[second]) for r in rows
             if _coded(r, first) and _coded(r, second)]
    table = Counter(pairs)
    return {key: table.get(key, 0)
            for key in (("1", "1"), ("1", "0"), ("0", "1"), ("0", "0"))}


def coverage(sessions, units, events):
    print("== COVERAGE ==")
    print(f"interactions {len(sessions)} | "
          f"participants {len({r['participant_id'] for r in sessions})} | "
          f"units {len(units)}")
    print("  n per condition (bookkeeping):",
          dict(sorted(Counter(r["condition"] for r in sessions).items())))
    print("  label source:",
          dict(Counter(r.get("coding_label_source") or "none" for r in sessions)))
    # Availability by condition is not an outcome: it says how much data each
    # cell holds, which is what a reader needs to judge whether an eventual
    # contrast rests on comparable coverage.
    verified = {r["session_id"] for r in (events or [])
                if (r.get("verifier_initials") or "").strip()}
    probe = {r["session_id"] for r in units if _coded(r, "retention")}
    print(f"  {'condition':<18} {'coded':>7} {'final acct':>11} {'turn verif':>11}")
    for condition in sorted({r["condition"] for r in sessions}):
        cell = [r for r in sessions if r["condition"] == condition]
        ids = {r["session_id"] for r in cell}
        print(f"  {condition:<18} {len(cell):>7} "
              f"{len(ids & probe):>11} {len(ids & verified):>11}")


def uptake(units):
    print("\n== INFORMATION UPTAKE (per critical unit) ==")
    print(f"  all units n = {len(units)}")
    _line("attempted", units, "attempted")
    _line("completely delivered", units, "complete_raw")

    delivered = [r for r in units if r.get("complete_raw") == "1"]
    gated = sum(1 for r in delivered
                if (r.get("grounding_gated_reason") or "").strip())
    print(f"\n  completely delivered n = {len(delivered)} "
          f"({gated} transmitted-gated)")
    for field, label in (("acknowledgement", "acknowledged"),
                         ("update_claim", "claimed recorded"),
                         ("incorporation", "operationally used"),
                         ("retention", "final-account fidelity")):
        _line(label, delivered, field)

    print("\n  conditional")
    _line("used | acknowledged",
          [r for r in delivered if r.get("acknowledgement") == "1"], "incorporation")
    _line("used | not acknowledged",
          [r for r in delivered if r.get("acknowledgement") == "0"], "incorporation")
    _line("retained | used",
          [r for r in delivered if r.get("incorporation") == "1"], "retention")
    _line("retained | claimed recorded",
          [r for r in delivered if r.get("update_claim") == "1"], "retention")

    # The links are not uniformly nested: a unit can be used without having
    # been acknowledged. The off-diagonals are the reportable part.
    print("\n  2x2 (both / first only / second only / neither)")
    for first, second in (("acknowledgement", "incorporation"),
                          ("incorporation", "retention"),
                          ("update_claim", "retention")):
        cell = _cross(delivered, first, second)
        print(f"  {first[:12]:>12} -> {second[:12]:<12} "
              f"{cell[('1','1')]:3d} {cell[('1','0')]:3d} "
              f"{cell[('0','1')]:3d} {cell[('0','0')]:3d}   "
              f"n = {sum(cell.values())}")


def repair(sessions, moves):
    print("\n== REPAIR ==")
    counts = [int(float(r["repair_total"])) for r in sessions
              if (r.get("repair_total") or "").strip()]
    if not counts:
        print("  no coded repair moves")
        return
    ordered = sorted(counts)
    mean = sum(counts) / len(counts)
    variance = (sum((x - mean) ** 2 for x in counts) / (len(counts) - 1)
                if len(counts) > 1 else 0.0)
    print(f"  total {sum(counts)} over {len(counts)} interactions")
    print(f"  mean {mean:.2f}  median {ordered[len(ordered) // 2]}  "
          f"range {ordered[0]}-{ordered[-1]}  none {counts.count(0)}")
    # Reported because it decides Poisson against negative binomial.
    print(f"  variance {variance:.2f}  (variance/mean {variance / mean:.2f})")
    print("  distribution:",
          " ".join(f"{k}:{v}" for k, v in sorted(Counter(counts).items())))

    per_participant = defaultdict(list)
    for row in sessions:
        if (row.get("repair_total") or "").strip():
            per_participant[row["participant_id"]].append(
                int(float(row["repair_total"])))
    means = sorted(sum(v) / len(v) for v in per_participant.values())
    print(f"  per-participant mean: {means[0]:.1f} to {means[-1]:.1f} "
          f"over {len(means)} participants")

    if moves:
        print("  categories:")
        for category, count in Counter(m["category"] for m in moves).most_common():
            print(f"    {category:<26} {count:3d}  {100 * count / len(moves):4.1f}%")
        floor = sum(1 for m in moves
                    if m["category"] in ("floor_recovery", "restart_after_cutoff"))
        print(f"    {'(floor-directed)':<26} {floor:3d}  "
              f"{100 * floor / len(moves):4.1f}%")


def turn_taking(sessions, events):
    print("\n== TURN-TAKING ==")
    if not events:
        print("  no certified turn events in the frames")
        return
    verified = [r for r in events if (r.get("verifier_initials") or "").strip()]
    reviewed = {r["session_id"] for r in verified}
    print(f"  PRELIMINARY: {len(verified)} of {len(events)} episodes verified, "
          f"{len(reviewed)} of {len(sessions)} interactions")
    if not verified:
        return

    def total(field):
        return sum(1 for r in verified if r.get(field) == "1")

    overlaps = total("verified_overlap")
    barge = total("verified_participant_barge_in")
    premature = total("verified_assistant_premature_onset")
    yielded = total("successful_assistant_yielding")
    disruptive = total("disruptive_assistant_interruption")
    print(f"  verified overlaps                {overlaps:3d}")
    print(f"    assistant premature onset      {premature:3d}  "
          f"{100 * premature / overlaps:4.1f}% of overlaps")
    print(f"      disruptive                   {disruptive:3d}  "
          f"{100 * disruptive / premature:4.1f}% of premature"
          if premature else "      disruptive                     -")
    print(f"    participant barge-in           {barge:3d}  "
          f"{100 * barge / overlaps:4.1f}% of overlaps")
    print(f"      assistant yielded            {yielded:3d}  "
          f"{100 * yielded / barge:4.1f}% of barge-ins"
          if barge else "      assistant yielded              -")
    per_session = Counter(r["session_id"] for r in verified
                          if r.get("verified_assistant_premature_onset") == "1")
    affected = sum(1 for s in reviewed if per_session[s])
    print(f"  interactions with >=1 premature onset  {affected} of "
          f"{len(reviewed)} reviewed")
    print(f"  per reviewed interaction: overlaps {overlaps / len(reviewed):.1f}  "
          f"premature {premature / len(reviewed):.1f}  "
          f"barge-ins {barge / len(reviewed):.1f}")
    latency = (("verified_assistant_stop_latency_ms", "assistant stop latency"),
               ("verified_participant_stop_latency_ms", "participant stop latency"))
    for field, label in latency:
        values = []
        for row in verified:
            try:
                values.append(float(row.get(field)))
            except (TypeError, ValueError):
                continue
        state = (f"n={len(values)} median "
                 f"{sorted(values)[len(values) // 2]:.0f} ms" if values
                 else "NO OBSERVATIONS COLLECTED")
        print(f"  {label:<32} {state}")


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        average = (start + stop) / 2 + 1
        for position in range(start, stop + 1):
            out[order[position]] = average
        start = stop + 1
    return out


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if not sx or not sy:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _spearman(xs, ys):
    return _pearson(_ranks(xs), _ranks(ys))


def _within(triples):
    """Participant-mean-centred pairs.

    Each participant contributes four interactions, so a raw correlation mixes
    within-participant covariation with between-participant differences in how
    people use a Likert scale. Centring leaves only the former.
    """
    grouped = defaultdict(list)
    for participant, first, second in triples:
        grouped[participant].append((first, second))
    xs, ys = [], []
    for pairs in grouped.values():
        if len(pairs) < 2:
            continue
        mx = sum(a for a, _ in pairs) / len(pairs)
        my = sum(b for _, b in pairs) / len(pairs)
        for a, b in pairs:
            xs.append(a - mx)
            ys.append(b - my)
    return xs, ys


def experience(sessions, units):
    """Behavioural measures against self-report. Exploratory and pooled."""
    print("\n== BEHAVIOUR vs SELF-REPORT (exploratory, pooled) ==")
    retained = defaultdict(list)
    for unit in units:
        if _coded(unit, "retention"):
            retained[unit["session_id"]].append(int(unit["retention"]))

    def number(row, field):
        try:
            return float(row[field])
        except (TypeError, ValueError, KeyError):
            return None

    measures = [
        ("repair_total", lambda r: number(r, "repair_total")),
        ("outcome_level", lambda r: number(r, "outcome_level")),
        ("retained_fraction",
         lambda r: (sum(retained[r["session_id"]]) / len(retained[r["session_id"]]))
         if retained[r["session_id"]] else None),
    ]
    print(f"  {'behavioural':<18} {'self-report':<30} {'n':>4} "
          f"{'rho':>7} {'rho within':>11}")
    for name, getter in measures:
        for field in LIKERT:
            triples = [(r["participant_id"], getter(r), number(r, field))
                       for r in sessions]
            triples = [t for t in triples if t[1] is not None and t[2] is not None]
            if len(triples) < 10:
                continue
            raw = _spearman([t[1] for t in triples], [t[2] for t in triples])
            xs, ys = _within(triples)
            inner = _spearman(xs, ys)
            print(f"  {name:<18} {field:<30} {len(triples):>4} "
                  f"{raw if raw is None else f'{raw:7.2f}'} "
                  f"{inner if inner is None else f'{inner:11.2f}'}")
    print("  (Spearman; no inference — the modelled version belongs in the")
    print("   exploratory analyses, with participant random intercepts.)")


def self_report(sessions):
    print("\n== SELF-REPORT (pooled) ==")
    durations = sorted(float(r["duration_s"]) for r in sessions
                       if r.get("duration_s"))
    if durations:
        print(f"  duration_s: median {durations[len(durations) // 2]:.0f}, "
              f"min {durations[0]:.0f}, max {durations[-1]:.0f}")
    for key in LIKERT:
        values = [int(float(r[key])) for r in sessions
                  if (r.get(key) or "").replace(".", "").isdigit()]
        if not values:
            print(f"  {key}: no data")
            continue
        bars = " ".join(f"{i}:{Counter(values).get(i, 0)}" for i in range(1, 8))
        print(f"  {key}: mean {sum(values) / len(values):.2f} | {bars}")
    for key in CATEGORICAL:
        values = [r[key] for r in sessions if r.get(key)]
        if not values:
            continue
        print(f"  {key}:")
        for value, count in Counter(values).most_common():
            print(f"      {count:3d}  {value}")


def by_condition(sessions, units, events, moves):
    """Every outcome section again, one condition at a time."""
    conditions = sorted({r["condition"] for r in sessions if r.get("condition")})
    for condition in conditions:
        cell = [r for r in sessions if r.get("condition") == condition]
        ids = {r["session_id"] for r in cell}
        print(f"\n{'=' * 72}\n=== {condition}  ({len(cell)} interactions)\n{'=' * 72}")
        uptake([u for u in units if u["session_id"] in ids])
        repair(cell, [m for m in moves if m["session_id"] in ids])
        turn_taking(cell, [e for e in events if e["session_id"] in ids])
        experience(cell, [u for u in units if u["session_id"] in ids])
        false_updates(cell)
        self_report([dict(r) for r in cell])


def false_updates(sessions):
    """An explicit claim to have recorded something, followed by failure to use
    or report it. Reported descriptively: the measure exists only where a
    claimed update occurred, so a condition contrast would condition on a
    post-treatment variable."""
    coded = [r for r in sessions if _coded(r, "false_update_confirmation")]
    if not coded:
        return
    hits = sum(1 for r in coded if r["false_update_confirmation"] == "1")
    print("\nfalse update confirmation")
    print(f"  {'coded interactions':<34} {len(coded)}")
    print(f"  {'with a false update confirmation':<34} {hits}"
          f"  ({100 * hits / len(coded):.1f}%)")


def main():
    sessions = _read(FRAMES, "scenario_level.csv")
    if sessions is None:
        raise SystemExit(f"missing {FRAMES}/scenario_level.csv — run prep.py first")
    units = _read(FRAMES, "unit_level.csv") or []
    events = _read(FRAMES, "turn_events_certified.csv") or []
    # Repair categories are not carried into a model frame; they come from the
    # export table, which sits beside the frames' own inputs.
    moves = _read(DATA, "repairs.csv") or []
    keep = {r["session_id"] for r in sessions}
    moves = [m for m in moves if m["session_id"] in keep]

    unblinded = "--by-condition" in sys.argv
    coverage(sessions, units, events)
    print(f"\n{'#' * 72}\n# POOLED ACROSS CONDITIONS\n{'#' * 72}")
    uptake(units)
    repair(sessions, moves)
    turn_taking(sessions, events)
    experience(sessions, units)
    false_updates(sessions)
    if unblinded:
        by_condition(sessions, units, events, moves)
        self_report([dict(r) for r in sessions])
        return
    for row in sessions:  # blind guard: nothing below may group by condition
        del row["condition"]
    self_report(sessions)


if __name__ == "__main__":
    main()
