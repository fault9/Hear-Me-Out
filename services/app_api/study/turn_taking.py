"""Pure interval operations for participant/assistant turn-taking measures."""

from __future__ import annotations

from typing import Any


TURN_EPISODE_SCHEMA = "hmo.turn-episode.v2"
TURN_GAP_SCHEMA = "hmo.turn-gap.v1"
OVERLAP_MINIMUM_MS = 200.0
# Silence that ends a turn. The RMS detectors emit an interval per stretch of
# speech separated by more than their 250 ms hangover, so one continuous turn
# arrives as several intervals. Pooled inter-interval gaps show a mode at
# 0.25-0.5 s decaying to a flat tail by ~1.5 s on the participant side and
# ~0.75 s on the assistant side; the looser of the two is used for both, which
# groups more rather than less and so cannot manufacture onsets.
ONSET_GROUPING_MS = 1500.0


def group_turns(intervals: list[dict], *,
                grouping_ms: float = ONSET_GROUPING_MS) -> list[dict]:
    turns: list[dict[str, Any]] = []
    for index, interval in enumerate(intervals):
        start, end = float(interval["start_ms"]), float(interval["end_ms"])
        if turns and start - turns[-1]["end_ms"] <= grouping_ms:
            turns[-1]["end_ms"] = max(turns[-1]["end_ms"], end)
            turns[-1]["members"].append((index, start, end))
        else:
            turns.append({"start_ms": start, "end_ms": end,
                          "members": [(index, start, end)]})
    return turns


def build_turn_episodes(participant: list[dict], assistant: list[dict], *,
                        overlap_minimum_ms: float = OVERLAP_MINIMUM_MS,
                        grouping_ms: float = ONSET_GROUPING_MS) -> list[dict]:
    """Link every participant/assistant turn intersection once.

    Directional flags are onset relationships, not semantic judgments. A
    participant barge-in candidate begins after assistant onset and before its
    offset; an assistant premature-onset candidate is the symmetric case.
    Exact equal onsets remain non-directional because the detectors are
    quantized and cannot establish who started first.

    Onsets belong to turns, not to detected intervals. Evaluated per interval,
    a speaker who already held the floor, paused, and resumed was recorded as
    starting the overlap - 26% of participant barge-in candidates and 24% of
    assistant premature-onset candidates over the pooled sessions, each naming
    the party opposite to the one that actually interrupted. Overlap duration
    sums the intersecting speech and so excludes silence inside either turn.
    """
    rows: list[dict[str, Any]] = []
    participant_turns = group_turns(participant, grouping_ms=grouping_ms)
    assistant_turns = group_turns(assistant, grouping_ms=grouping_ms)
    for participant_turn in participant_turns:
        p_start = participant_turn["start_ms"]
        p_end = participant_turn["end_ms"]
        for assistant_turn in assistant_turns:
            a_start = assistant_turn["start_ms"]
            a_end = assistant_turn["end_ms"]
            spans = [
                (max(p_span_start, a_span_start), min(p_span_end, a_span_end))
                for _, p_span_start, p_span_end in participant_turn["members"]
                for _, a_span_start, a_span_end in assistant_turn["members"]
                if min(p_span_end, a_span_end) > max(p_span_start, a_span_start)
            ]
            if not spans:
                continue
            overlap_start = min(start for start, _ in spans)
            overlap_end = max(end for _, end in spans)
            duration = sum(end - start for start, end in spans)

            participant_barge_in = a_start < p_start < a_end
            assistant_premature_onset = p_start < a_start < p_end
            if participant_barge_in:
                initiator = "participant"
            elif assistant_premature_onset:
                initiator = "assistant"
            else:
                initiator = "simultaneous"
            rows.append({
                "schema": TURN_EPISODE_SCHEMA,
                "participant_interval": participant_turn["members"][0][0],
                "assistant_interval": assistant_turn["members"][0][0],
                "participant_intervals": [row[0] for row in participant_turn["members"]],
                "assistant_intervals": [row[0] for row in assistant_turn["members"]],
                "participant_onset_ms": p_start,
                "participant_offset_ms": p_end,
                "assistant_onset_ms": a_start,
                "assistant_offset_ms": a_end,
                "overlap_start_ms": overlap_start,
                "overlap_end_ms": overlap_end,
                "overlap_duration_ms": duration,
                "onset_grouping_ms": grouping_ms,
                "initiator": initiator,
                "overlap_200ms_candidate": duration >= overlap_minimum_ms,
                "participant_barge_in_candidate": participant_barge_in,
                "assistant_premature_onset_candidate": assistant_premature_onset,
                "assistant_stop_latency_ms": (
                    a_end - p_start if participant_barge_in else None),
                "participant_stop_latency_ms": (
                    p_end - a_start if assistant_premature_onset else None),
            })

    rows.sort(key=lambda row: (
        row["overlap_start_ms"], row["participant_onset_ms"],
        row["assistant_onset_ms"], row["participant_interval"],
        row["assistant_interval"],
    ))
    for index, row in enumerate(rows, start=1):
        row["episode_id"] = f"overlap_{index:04d}"
    return rows


def build_positive_response_gaps(participant: list[dict],
                                 assistant: list[dict]) -> list[dict]:
    """Return silence-bounded speaker changes without manufacturing turns.

    For each interval onset, the latest completed speech interval is located.
    A positive gap is emitted only when that unambiguous preceding interval
    belongs to the other speaker. Ties involving both speakers are omitted.
    """
    intervals: list[dict[str, Any]] = []
    for speaker, source in (("participant", participant), ("assistant", assistant)):
        for index, interval in enumerate(source):
            intervals.append({
                "speaker": speaker,
                "interval": index,
                "start_ms": float(interval["start_ms"]),
                "end_ms": float(interval["end_ms"]),
            })

    rows: list[dict[str, Any]] = []
    for current in intervals:
        # A response gap requires actual silence before the onset. If any other
        # interval is still active (or ends exactly) at this onset, the speaker
        # change is an overlap/zero-gap boundary and belongs in turn episodes.
        if any(row is not current
               and row["start_ms"] <= current["start_ms"] <= row["end_ms"]
               for row in intervals):
            continue
        preceding = [row for row in intervals
                     if row["end_ms"] < current["start_ms"]]
        if not preceding:
            continue
        latest_end = max(row["end_ms"] for row in preceding)
        latest = [row for row in preceding if row["end_ms"] == latest_end]
        previous_speakers = {row["speaker"] for row in latest}
        if len(previous_speakers) != 1:
            continue
        previous = max(latest, key=lambda row: row["start_ms"])
        if previous["speaker"] == current["speaker"]:
            continue
        rows.append({
            "schema": TURN_GAP_SCHEMA,
            "direction": f"{previous['speaker']}_to_{current['speaker']}",
            "from_speaker": previous["speaker"],
            "to_speaker": current["speaker"],
            "from_interval": previous["interval"],
            "to_interval": current["interval"],
            "gap_start_ms": previous["end_ms"],
            "gap_end_ms": current["start_ms"],
            "gap_duration_ms": current["start_ms"] - previous["end_ms"],
        })

    rows.sort(key=lambda row: (row["gap_start_ms"], row["gap_end_ms"],
                               row["direction"]))
    for index, row in enumerate(rows, start=1):
        row["gap_id"] = f"gap_{index:04d}"
    return rows


def _legacy_turn_episodes(timing: dict) -> list[dict]:
    """Best-effort compatibility for artifacts without saved intervals."""
    keyed: dict[tuple[Any, Any, Any], dict] = {}

    def key(row: dict, fallback: tuple[str, int]) -> tuple[Any, Any, Any]:
        participant_index = row.get("participant_interval")
        assistant_index = row.get("assistant_interval")
        if participant_index is not None and assistant_index is not None:
            return ("pair", participant_index, assistant_index)
        return (fallback[0], fallback[1], None)

    for index, overlap in enumerate(timing.get("overlaps") or [], start=1):
        start = overlap.get("start_ms")
        end = overlap.get("end_ms")
        duration = overlap.get("duration_ms")
        keyed[key(overlap, ("overlap", index))] = {
            "schema": TURN_EPISODE_SCHEMA,
            "participant_interval": overlap.get("participant_interval"),
            "assistant_interval": overlap.get("assistant_interval"),
            "participant_onset_ms": None,
            "participant_offset_ms": None,
            "assistant_onset_ms": None,
            "assistant_offset_ms": None,
            "overlap_start_ms": start,
            "overlap_end_ms": end,
            "overlap_duration_ms": duration,
            "initiator": "unknown_legacy",
            "overlap_200ms_candidate": (
                float(duration) >= OVERLAP_MINIMUM_MS
                if duration is not None else True),
            "participant_barge_in_candidate": False,
            "assistant_premature_onset_candidate": False,
            "assistant_stop_latency_ms": None,
            "participant_stop_latency_ms": None,
            "legacy_reconstruction": True,
        }

    for index, barge_in in enumerate(timing.get("barge_ins") or [], start=1):
        item_key = key(barge_in, ("barge_in", index))
        row = keyed.setdefault(item_key, {
            "schema": TURN_EPISODE_SCHEMA,
            "participant_interval": barge_in.get("participant_interval"),
            "assistant_interval": barge_in.get("assistant_interval"),
            "participant_onset_ms": barge_in.get(
                "participant_onset_ms", barge_in.get("start_ms")),
            "participant_offset_ms": None,
            "assistant_onset_ms": None,
            "assistant_offset_ms": barge_in.get(
                "assistant_stop_ms", barge_in.get("end_ms")),
            "overlap_start_ms": barge_in.get(
                "participant_onset_ms", barge_in.get("start_ms")),
            "overlap_end_ms": barge_in.get(
                "assistant_stop_ms", barge_in.get("end_ms")),
            "overlap_duration_ms": barge_in.get("duration_ms"),
            "initiator": "participant",
            "overlap_200ms_candidate": False,
            "participant_barge_in_candidate": True,
            "assistant_premature_onset_candidate": False,
            "assistant_stop_latency_ms": barge_in.get("stop_latency_ms"),
            "participant_stop_latency_ms": None,
            "legacy_reconstruction": True,
        })
        row["participant_barge_in_candidate"] = True
        row["initiator"] = "participant"
        row["assistant_stop_latency_ms"] = barge_in.get("stop_latency_ms")
        if row.get("participant_onset_ms") is None:
            row["participant_onset_ms"] = barge_in.get(
                "participant_onset_ms", barge_in.get("start_ms"))
        if row.get("assistant_offset_ms") is None:
            row["assistant_offset_ms"] = barge_in.get(
                "assistant_stop_ms", barge_in.get("end_ms"))

    rows = list(keyed.values())
    rows.sort(key=lambda row: (
        float(row.get("overlap_start_ms") or 0.0),
        float(row.get("overlap_end_ms") or 0.0),
    ))
    for index, row in enumerate(rows, start=1):
        row["episode_id"] = f"legacy_overlap_{index:04d}"
    return rows


def turn_events_from_timing(timing: dict) -> tuple[list[dict], list[dict]]:
    """Load v5 events or derive them from the intervals retained in v4."""
    episodes = timing.get("turn_episodes")
    gaps = timing.get("positive_response_gaps")
    if isinstance(episodes, list) and isinstance(gaps, list):
        return episodes, gaps

    participant = timing.get("participant_intervals") or []
    assistant = timing.get("assistant_intervals") or []
    if participant or assistant:
        return (build_turn_episodes(participant, assistant),
                build_positive_response_gaps(participant, assistant))
    return _legacy_turn_episodes(timing), []
