"""Compare observed, offline, and streaming X-VC on a technical-pilot session.

By default all derived audio is created in a temporary directory and deleted
after a terminal report is printed. Supplying ``--out`` retains the renders and
a JSON result, but this tool never writes to the study database or canonical
session analysis directories.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
APP_API_ROOT = REPO_ROOT / "services" / "app_api"
if str(APP_API_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_API_ROOT))

from study.transition_analysis import (  # noqa: E402
    participant_speech_intervals,
    read_events,
    route_regions,
)

EVENT_SAMPLE_RATE = 16000


def production_stream_profile() -> dict:
    """The single X-VC streaming profile used by the study."""
    return {
        "id": "production_stream",
        "mode": "streaming",
        "chunk_ms": 2400,
        "current_ms": 120,
        "smooth_ms": 20,
        "future_ms": 100,
        "silence_gate_rms": 0.008,
        "silence_hangover_ms": 360,
    }


def diagnostic_profiles() -> list[dict]:
    """Observed/offline references plus the frozen production stream."""
    return [
        {"id": "observed", "mode": "observed"},
        {"id": "offline", "mode": "offline"},
        production_stream_profile(),
    ]


def select_vc_utterances(
    events: list[dict],
    *,
    guard_s: float,
    padding_s: float,
    min_utterance_s: float,
) -> list[dict]:
    """Map RMS speech intervals to matched raw and transmitted VC intervals."""
    regions = route_regions(events)
    speech = participant_speech_intervals(events)
    selected: list[dict] = []
    padding = round(padding_s * EVENT_SAMPLE_RATE)
    guard = round(guard_s * EVENT_SAMPLE_RATE) if len(regions) > 1 else 0

    for region_index, region in enumerate(regions, start=1):
        if region["mode"] != "vc":
            continue
        input_start = int(region["input_start_sample"]) + guard
        input_end = int(region["input_end_sample"]) - guard
        tx_start = int(region["transmitted_start_sample"]) + guard
        tx_end = int(region["transmitted_end_sample"]) - guard
        if input_end <= input_start or tx_end <= tx_start:
            continue
        input_span = input_end - input_start
        tx_span = tx_end - tx_start
        for speech_start, speech_end in speech:
            start = max(input_start, speech_start)
            end = min(input_end, speech_end)
            if end <= start or (end - start) / EVENT_SAMPLE_RATE < min_utterance_s:
                continue
            padded_start = max(input_start, start - padding)
            padded_end = min(input_end, end + padding)
            mapped_start = tx_start + round(
                (padded_start - input_start) * tx_span / input_span
            )
            mapped_end = tx_start + round(
                (padded_end - input_start) * tx_span / input_span
            )
            selected.append({
                "region_index": region_index,
                "route_input_start_sample": int(region["input_start_sample"]),
                "route_input_end_sample": int(region["input_end_sample"]),
                "speech_start_sample": start,
                "speech_end_sample": end,
                "source_start_sample": padded_start,
                "source_end_sample": padded_end,
                "observed_start_sample": mapped_start,
                "observed_end_sample": mapped_end,
            })

    for index, row in enumerate(selected, start=1):
        row["utterance_id"] = f"u{index:03d}"
        row["duration_s"] = (
            row["source_end_sample"] - row["source_start_sample"]
        ) / EVENT_SAMPLE_RATE
    return selected


def _read_mono_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    dtype = {1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}.get(width)
    if dtype is None:
        raise ValueError(f"unsupported WAV sample width {width}: {path}")
    values = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    if width == 1:
        values = (values - 128.0) / 128.0
    else:
        values /= float(2 ** (8 * width - 1))
    return rate, values


def _write_wav(path: Path, rate: int, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def _write_event_slice(
    path: Path,
    rate: int,
    audio: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> None:
    start = round(start_sample * rate / EVENT_SAMPLE_RATE)
    end = round(end_sample * rate / EVENT_SAMPLE_RATE)
    start = max(0, min(len(audio), start))
    end = max(start, min(len(audio), end))
    _write_wav(path, rate, audio[start:end])


def _concatenate(paths: list[Path], destination: Path, separator_s: float = 0.15) -> None:
    rate: int | None = None
    pieces: list[np.ndarray] = []
    for path in paths:
        clip_rate, clip = _read_mono_wav(path)
        if rate is None:
            rate = clip_rate
        elif clip_rate != rate:
            raise ValueError("aggregate clips must use one sample rate")
        if pieces:
            pieces.append(np.zeros(round(separator_s * rate), dtype=np.float32))
        pieces.append(clip)
    if rate is None:
        raise ValueError("cannot concatenate an empty clip list")
    _write_wav(destination, rate, np.concatenate(pieces))


def _find_session(data_root: Path, session_id: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"session directory not found: {path}")
        return path
    matches = [
        path for path in data_root.glob("sessions/**/attempt_*")
        if path.is_dir() and path.name.endswith(f"_{session_id}")
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one directory for {session_id}, found {len(matches)} under {data_root}"
        )
    return matches[0]


def _default_xvc_dir() -> Path:
    workspace = REPO_ROOT.parent
    candidates = [
        Path(os.environ["XVC_DIR"]) if os.environ.get("XVC_DIR") else None,
        workspace / "X-VC",
        workspace / "X-VC-accent-finetuning",
    ]
    return next((path for path in candidates if path and path.is_dir()), workspace / "X-VC")


def _render_profiles(
    profiles: list[dict],
    utterances: list[dict],
    route_sources: dict[int, Path],
    target: Path,
    work_dir: Path,
    args: argparse.Namespace,
) -> dict[tuple[str, str], dict]:
    rendered = [profile for profile in profiles if profile["mode"] != "observed"]
    if not rendered:
        return {}
    xvc_dir = Path(args.xvc_dir).expanduser().resolve()
    config = Path(args.xvc_config or xvc_dir / "configs" / "xvc.yaml").resolve()
    checkpoint = Path(args.xvc_ckpt or xvc_dir / "ckpts" / "xvc.pt").resolve()
    for required in (xvc_dir, config, checkpoint):
        if not required.exists():
            raise FileNotFoundError(f"required X-VC path not found: {required}")

    offline_profiles = [profile for profile in rendered if profile["mode"] == "offline"]
    streaming_profiles = [
        profile for profile in rendered if profile["mode"] == "streaming"
    ]
    jobs = []
    for utterance in utterances:
        outputs = {
            profile["id"]: str(
                work_dir / "renders" / profile["id"]
                / f"{utterance['utterance_id']}.wav"
            )
            for profile in offline_profiles
        }
        if outputs:
            jobs.append({
                "job_id": utterance["utterance_id"],
                "source_path": str(utterance["source_path"]),
                "outputs": outputs,
            })
    for region_index, source_path in route_sources.items():
        outputs = {
            profile["id"]: str(
                work_dir / "route_renders" / profile["id"]
                / f"region_{region_index:02d}.wav"
            )
            for profile in streaming_profiles
        }
        if outputs:
            jobs.append({
                "job_id": f"region_{region_index:02d}",
                "source_path": str(source_path),
                "outputs": outputs,
            })
    manifest = {
        "schema": "hmo.xvc-pilot-render-request.v1",
        "target_path": str(target),
        "xvc": {
            "directory": str(xvc_dir),
            "config": str(config),
            "checkpoint": str(checkpoint),
            "device": args.xvc_device,
            "ema_load": True,
        },
        "profiles": rendered,
        "jobs": jobs,
    }
    manifest_path = work_dir / "render_request.json"
    result_path = work_dir / "render_results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    command = [
        "uv", "run", "--project", str(REPO_ROOT / "services" / "xvc"),
        "python", str(REPO_ROOT / "services" / "xvc" / "pilot_render.py"),
        "--manifest", str(manifest_path), "--results", str(result_path),
    ]
    print(
        f"\nRendering "
        f"{len(utterances) if offline_profiles else 0} offline utterances and "
        f"{len(route_sources)} complete streaming VC routes..."
    )
    subprocess.run(command, check=True)
    rows = json.loads(result_path.read_text(encoding="utf-8"))["rows"]
    raw_rows = {(row["job_id"], row["profile_id"]): row for row in rows}
    result: dict[tuple[str, str], dict] = {}
    route_audio: dict[tuple[int, str], tuple[int, np.ndarray]] = {}
    for utterance in utterances:
        utterance_id = utterance["utterance_id"]
        for profile in rendered:
            profile_id = profile["id"]
            if profile["mode"] == "offline":
                result[(utterance_id, profile_id)] = raw_rows[
                    (utterance_id, profile_id)
                ]
                continue
            region_index = utterance["region_index"]
            route_row = raw_rows[(f"region_{region_index:02d}", profile_id)]
            cache_key = (region_index, profile_id)
            if cache_key not in route_audio:
                route_audio[cache_key] = _read_mono_wav(Path(route_row["output_path"]))
            rate, audio = route_audio[cache_key]
            relative_start = (
                utterance["source_start_sample"]
                - utterance["route_input_start_sample"]
            )
            relative_end = (
                utterance["source_end_sample"]
                - utterance["route_input_start_sample"]
            )
            destination = (
                work_dir / "renders" / profile_id / f"{utterance_id}.wav"
            )
            _write_event_slice(destination, rate, audio, relative_start, relative_end)
            result[(utterance_id, profile_id)] = {
                **route_row,
                "utterance_id": utterance_id,
                "output_path": str(destination),
                "render_scope": "complete_vc_route",
            }
    return result


def _median(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.median(usable) if usable else None


def _metric(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def _score_profiles(
    profiles: list[dict],
    utterances: list[dict],
    target: Path,
    render_rows: dict[tuple[str, str], dict],
    work_dir: Path,
    verbose: bool,
) -> dict:
    from vc_quality import (
        evaluate_conversion,
        naturalness,
        transcribe_for_evaluation,
    )

    print("\nScoring per-utterance UTMOS and SIM...")
    raw_utmos: dict[str, dict] = {}
    for utterance in utterances:
        raw_utmos[utterance["utterance_id"]] = naturalness(
            str(utterance["source_path"])
        )

    detail = []
    profile_outputs: dict[str, list[Path]] = {}
    for profile in profiles:
        profile_id = profile["id"]
        outputs = []
        for utterance in utterances:
            utterance_id = utterance["utterance_id"]
            if profile["mode"] == "observed":
                converted = utterance["observed_path"]
                render = None
            else:
                render = render_rows[(utterance_id, profile_id)]
                converted = Path(render["output_path"])
            outputs.append(converted)
            score = evaluate_conversion(
                str(converted),
                str(target),
                source_path=str(utterance["source_path"]),
                compute_intelligibility=False,
                compute_speaker_similarity=True,
                compute_utmos=True,
            )
            raw_score = raw_utmos[utterance_id].get("utmos")
            converted_score = score.get("utmos")
            row = {
                "utterance_id": utterance_id,
                "profile_id": profile_id,
                "duration_s": utterance["duration_s"],
                "raw_utmos": raw_score,
                "utmos": converted_score,
                "utmos_delta": (
                    converted_score - raw_score
                    if converted_score is not None and raw_score is not None else None
                ),
                "sim": score.get("sim"),
                "utmos_error": score.get("utmos_error"),
                "sim_error": score.get("sim_error"),
                "render_rtf": render.get("render_rtf") if render else None,
                "silence_bypassed_windows": (
                    render.get("silence_bypassed_windows") if render else None
                ),
            }
            detail.append(row)
            if verbose:
                print(
                    f"  {utterance_id:<5} {profile_id:<22} "
                    f"UTMOS={_metric(row['utmos'])} "
                    f"delta={_metric(row['utmos_delta'])} SIM={_metric(row['sim'])}"
                )
        profile_outputs[profile_id] = outputs

    aggregate_dir = work_dir / "aggregate"
    source_aggregate = aggregate_dir / "source.wav"
    _concatenate([row["source_path"] for row in utterances], source_aggregate)
    print("\nTranscribing the concatenated raw-microphone utterances once for pseudo-WER...")
    source_asr = transcribe_for_evaluation(str(source_aggregate))

    summaries = []
    for profile in profiles:
        profile_id = profile["id"]
        converted_aggregate = aggregate_dir / f"{profile_id}.wav"
        _concatenate(profile_outputs[profile_id], converted_aggregate)
        aggregate = evaluate_conversion(
            str(converted_aggregate),
            str(target),
            source_path=str(source_aggregate),
            source_asr=source_asr,
            compute_intelligibility=True,
            compute_speaker_similarity=True,
            compute_utmos=False,
        )
        rows = [row for row in detail if row["profile_id"] == profile_id]
        summaries.append({
            "profile_id": profile_id,
            "utterances": len(rows),
            "utmos_median": _median([row["utmos"] for row in rows]),
            "utmos_delta_median": _median([row["utmos_delta"] for row in rows]),
            "sim_median": _median([row["sim"] for row in rows]),
            "aggregate_wer": aggregate.get("wer"),
            "aggregate_sim": aggregate.get("sim"),
            "render_rtf_median": _median([row["render_rtf"] for row in rows]),
            "wer_status": aggregate.get("wer_status"),
            "wer_error": aggregate.get("wer_error"),
        })
    return {"source_asr": source_asr, "utterance_scores": detail, "summaries": summaries}


def _print_report(session_id: str, utterances: list[dict], result: dict) -> None:
    profile_width = max(
        23, max(len(row["profile_id"]) for row in result["summaries"])
    )
    table_width = profile_width + 68
    raw_utmos = _median([
        row.get("raw_utmos") for row in result["utterance_scores"]
        if row["profile_id"] == result["summaries"][0]["profile_id"]
    ])
    print("\n" + "=" * table_width)
    print(f"X-VC PILOT CALIBRATION: {session_id}")
    print(
        f"Matched VC utterances: {len(utterances)} | "
        "boundaries: participant RMS events (estimated, route-aware)"
    )
    print(f"Raw-microphone median UTMOS: {_metric(raw_utmos)}")
    print("WER: ASR(converted aggregate) vs ASR(raw-microphone aggregate); free-speech proxy")
    print("-" * table_width)
    print(
        f"{'Profile':<{profile_width}} {'N':>3} {'UTMOS':>8} {'Delta':>8} "
        f"{'SIM-med':>8} {'WER':>8} {'SIM-all':>8} {'RTF-med':>9}"
    )
    print("-" * table_width)
    for row in result["summaries"]:
        print(
            f"{row['profile_id']:<{profile_width}} {row['utterances']:>3} "
            f"{_metric(row['utmos_median']):>8} "
            f"{_metric(row['utmos_delta_median']):>8} "
            f"{_metric(row['sim_median']):>8} "
            f"{_metric(row['aggregate_wer']):>8} "
            f"{_metric(row['aggregate_sim']):>8} "
            f"{_metric(row['render_rtf_median'], 2):>9}"
        )
    print("=" * table_width)
    print("Interpretation: UTMOS/SIM higher is better; WER/RTF lower is better.")
    print("RTF <= 1 is a live-feasibility check, not an end-to-end latency measurement.")
    if result["source_asr"].get("status") != "complete":
        print(f"Raw ASR unavailable: {result['source_asr'].get('error')}")
    unavailable = [
        row for row in result["summaries"]
        if row["aggregate_wer"] is None and row.get("wer_error")
    ]
    for row in unavailable:
        print(f"{row['profile_id']} WER unavailable: {row['wer_error']}")


def prepare_session_inputs(args: argparse.Namespace, work_dir: Path) -> dict:
    """Extract reusable route-aware pilot clips without changing the session."""
    data_root = Path(args.data_root).expanduser().resolve()
    session_dir = _find_session(data_root, args.session, args.session_dir)
    source = session_dir / "participant_raw.wav"
    observed = session_dir / "participant.wav"
    target = session_dir / "target.wav"
    events_path = session_dir / "events.jsonl"
    for required in (source, observed, target, events_path):
        if not required.exists():
            raise FileNotFoundError(f"required session artifact missing: {required}")

    events = read_events(events_path)
    utterances = select_vc_utterances(
        events,
        guard_s=args.guard_s,
        padding_s=args.padding_s,
        min_utterance_s=args.min_utterance_s,
    )
    if args.max_utterances is not None:
        utterances = utterances[:args.max_utterances]
    if not utterances:
        raise ValueError("no eligible participant speech intervals were found in a VC route")

    source_rate, source_audio = _read_mono_wav(source)
    observed_rate, observed_audio = _read_mono_wav(observed)
    for utterance in utterances:
        utterance_id = utterance["utterance_id"]
        source_path = work_dir / "source" / f"{utterance_id}.wav"
        observed_path = work_dir / "observed" / f"{utterance_id}.wav"
        _write_event_slice(
            source_path, source_rate, source_audio,
            utterance["source_start_sample"], utterance["source_end_sample"],
        )
        _write_event_slice(
            observed_path, observed_rate, observed_audio,
            utterance["observed_start_sample"], utterance["observed_end_sample"],
        )
        utterance["source_path"] = source_path
        utterance["observed_path"] = observed_path

    route_sources: dict[int, Path] = {}
    for utterance in utterances:
        region_index = utterance["region_index"]
        if region_index in route_sources:
            continue
        route_path = work_dir / "source_routes" / f"region_{region_index:02d}.wav"
        _write_event_slice(
            route_path,
            source_rate,
            source_audio,
            utterance["route_input_start_sample"],
            utterance["route_input_end_sample"],
        )
        route_sources[region_index] = route_path

    return {
        "session_dir": session_dir,
        "target": target,
        "utterances": utterances,
        "route_sources": route_sources,
    }


def _run(args: argparse.Namespace, work_dir: Path) -> dict:
    prepared = prepare_session_inputs(args, work_dir)
    session_dir = prepared["session_dir"]
    target = prepared["target"]
    utterances = prepared["utterances"]
    route_sources = prepared["route_sources"]

    profiles = diagnostic_profiles()
    render_rows = _render_profiles(
        profiles, utterances, route_sources, target, work_dir, args
    )
    result = _score_profiles(
        profiles, utterances, target, render_rows, work_dir, args.verbose
    )
    result.update({
        "schema": "hmo.xvc-pilot-calibration.v1",
        "session_id": args.session,
        "session_dir": str(session_dir),
        "profiles": profiles,
        "utterances": [
            {key: str(value) if isinstance(value, Path) else value
             for key, value in row.items()}
            for row in utterances
        ],
    })
    _print_report(args.session, utterances, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal-only X-VC calibration on a technical-pilot session."
    )
    parser.add_argument("--session", required=True, help="HMO session ID")
    parser.add_argument("--data-root", default="/workspace/data/media")
    parser.add_argument("--session-dir", help="explicit attempt directory override")
    parser.add_argument("--guard-s", type=float, default=0.5)
    parser.add_argument("--padding-s", type=float, default=0.2)
    parser.add_argument("--min-utterance-s", type=float, default=0.8)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--xvc-dir", default=str(_default_xvc_dir()))
    parser.add_argument("--xvc-config")
    parser.add_argument("--xvc-ckpt")
    parser.add_argument("--xvc-device", type=int, default=0)
    parser.add_argument("--quality-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--verbose", action="store_true", help="print every utterance score")
    parser.add_argument(
        "--out",
        help="retain temporary audio and result JSON here; directory must not exist",
    )
    args = parser.parse_args()
    if args.max_utterances is not None and args.max_utterances <= 0:
        parser.error("--max-utterances must be > 0")
    os.environ["VC_QUALITY_DEVICE"] = args.quality_device
    workspace = REPO_ROOT.parent
    os.environ.setdefault("SPEAKER_VERIFICATION_ROOT", str(workspace))
    os.environ.setdefault(
        "MEANVC_SV_CKPT",
        str(workspace / "models" / "meanvc-sv" / "wavlm_large_finetune.pth"),
    )

    if args.out:
        work_dir = Path(args.out).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=False)
        result = _run(args, work_dir)
        (work_dir / "calibration_results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(f"Retained calibration artifacts: {work_dir}")
    else:
        with tempfile.TemporaryDirectory(prefix="hmo-xvc-calibration-") as temporary:
            _run(args, Path(temporary))
        print("Temporary calibration audio removed; no study data or database rows were changed.")


if __name__ == "__main__":
    main()
