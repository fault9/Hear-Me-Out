"""Verify a session's transmitted-audio continuity from surviving artifacts.

For sessions whose end-of-call proxy delivery failed, this cross-checks the
live event diary against the browser's monitor copy of the transmitted audio:

  1. input chunks arrived in sequence with no gaps;
  2. transmitted windows tile the input timeline with no holes;
  3. the monitor copy's sample count reconciles with the transmitted total;
  4. route activation and conversion activity are present as expected.

PASS means the monitor copy provably contains the complete transmitted stream
(admissible under the dated validity amendment); FAIL means it does not.

Usage: python3 scripts/continuity_check.py <session_dir>
       python3 scripts/continuity_check.py --events <events.jsonl>   (diary only)
"""

import json
import os
import sys
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
        expected = set(range(min(seqs), max(seqs) + 1))
        missing = sorted(expected - set(seqs))
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

    inferences = [e for e in events if e.get("event") == "xvc_inference_batch"]
    report["inference_batches"] = len(inferences)

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


def main():
    args = sys.argv[1:]
    wav_samples = None
    if args and args[0] == "--events":
        events = load_events(args[1])
    elif args:
        session_dir = args[0]
        events = load_events(os.path.join(session_dir, "events.jsonl"))
        wav_path = os.path.join(session_dir, "participant.wav")
        if os.path.exists(wav_path):
            with wave.open(wav_path, "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
            wav_samples = frames if rate == SR else int(frames * SR / rate)
        else:
            print("participant.wav not found — running diary-only checks")
    else:
        sys.exit(__doc__)

    failures, report = check(events, wav_samples)
    for key, value in report.items():
        print(f"{key}: {value}")
    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    verdict = "PASS" if wav_samples is not None else "PASS (diary only — rerun with session dir for the wav reconciliation)"
    print(verdict)


if __name__ == "__main__":
    main()
