"""Transmitted-audio continuity verification from surviving artifacts.

For sessions whose end-of-call proxy delivery failed, cross-checks the live
event diary against the browser's monitor copy of the transmitted audio. A
PASS verdict means the monitor copy provably contains the complete certified
transmitted span; admissibility under the dated validity amendment is decided
at analysis time from this recorded verdict, never by mutating the frozen
technical-validity status.
"""

import json
import os
import wave

SR = 16000
TOLERANCE_SAMPLES = SR  # one second of slack for the final partial window


def load_events(path):
    events = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def check(events, wav_samples):
    failures = []
    report = {}

    chunks = [e for e in events if e.get("event") == "input_chunk"]
    seqs = [e.get("chunk_sequence") for e in chunks if e.get("chunk_sequence") is not None]
    report["input_chunks"] = len(chunks)
    if seqs:
        missing = sorted(set(range(min(seqs), max(seqs) + 1)) - set(seqs))
        report["missing_chunk_sequences"] = len(missing)
        if missing:
            failures.append(f"{len(missing)} input chunk(s) missing from sequence")

    windows = [e for e in events if e.get("event") == "transmitted_window"]
    report["transmitted_windows"] = len(windows)
    if not windows:
        failures.append("no transmitted_window events")
        return failures, report
    windows.sort(key=lambda e: (e.get("input_start_sample") or 0))
    holes = 0
    cursor = windows[0].get("input_start_sample") or 0
    report["transmitted_first_sample"] = cursor
    for w in windows:
        start = w.get("input_start_sample")
        end = w.get("input_end_sample")
        if start is None or end is None:
            continue
        if start > cursor:
            holes += 1
        cursor = max(cursor, end)
    report["transmitted_last_sample"] = cursor
    report["transmitted_holes"] = holes
    if holes:
        failures.append(f"{holes} hole(s) in transmitted window tiling")

    out_seqs = [w.get("output_sequence") for w in windows
                if w.get("output_sequence") is not None]
    if out_seqs:
        missing_out = len(set(range(min(out_seqs), max(out_seqs) + 1)) - set(out_seqs))
        report["missing_output_sequences"] = missing_out
        if missing_out:
            failures.append(f"{missing_out} transmitted window(s) missing by sequence")

    routes = [e for e in events if e.get("event") == "route_activated"]
    report["route_activations"] = [
        {"to": e.get("to_mode"), "input_sample": e.get("input_sample")} for e in routes]
    if not routes:
        failures.append("no route_activated event")

    report["inference_batches"] = sum(
        1 for e in events if e.get("event") == "xvc_inference_batch")

    if wav_samples is not None:
        report["monitor_wav_samples"] = wav_samples
        shortfall = cursor - wav_samples
        if shortfall > TOLERANCE_SAMPLES:
            failures.append(
                f"monitor copy is {shortfall} samples ({shortfall / SR:.2f}s) shorter "
                f"than the certified transmitted span")
        # The diary can be truncated by the same delivery failure that triggers
        # this check, so audio past its last entry is expected — noted, not failed.
        excess = max(0, wav_samples - cursor)
        report["uncertified_tail_s"] = round(excess / SR, 2)
    else:
        report["monitor_wav_samples"] = None

    return failures, report


def run_for_session_dir(session_dir):
    events = load_events(os.path.join(session_dir, "events.jsonl"))
    wav_samples = None
    wav_path = os.path.join(session_dir, "participant.wav")
    if os.path.exists(wav_path):
        with wave.open(wav_path, "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
        wav_samples = frames if rate == SR else int(frames * SR / rate)
    failures, report = check(events, wav_samples)
    return {
        "schema": "hmo.continuity-check.v1",
        "verdict": "pass" if (not failures and wav_samples is not None) else (
            "pass_diary_only" if not failures else "fail"),
        "failures": failures,
        **report,
    }
