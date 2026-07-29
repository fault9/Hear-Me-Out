"""Rank candidate target WAVs with the frozen production X-VC stream."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

from pilot_calibration import (
    REPO_ROOT,
    _concatenate,
    _default_xvc_dir,
    _metric,
    _read_mono_wav,
    _write_event_slice,
    prepare_session_inputs,
    production_stream_profile,
)


def _median(values):
    usable = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.median(usable) if usable else None


def _candidate_wavs(directory: Path) -> list[Path]:
    return sorted(
        path.resolve() for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def _render_candidates(
    candidates: list[Path],
    utterances: list[dict],
    route_sources: dict[int, Path],
    work_dir: Path,
    args: argparse.Namespace,
) -> dict[tuple[str, str], dict]:
    xvc_dir = Path(args.xvc_dir).expanduser().resolve()
    config = Path(args.xvc_config or xvc_dir / "configs" / "xvc.yaml").resolve()
    checkpoint = Path(args.xvc_ckpt or xvc_dir / "ckpts" / "xvc.pt").resolve()
    for required in (xvc_dir, config, checkpoint):
        if not required.exists():
            raise FileNotFoundError(f"required X-VC path not found: {required}")

    profile = production_stream_profile()
    jobs = []
    candidate_ids = {str(path): f"c{index:03d}" for index, path in enumerate(candidates, 1)}
    for candidate in candidates:
        candidate_id = candidate_ids[str(candidate)]
        for region_index, source_path in route_sources.items():
            jobs.append({
                "job_id": f"{candidate_id}_region_{region_index:02d}",
                "source_path": str(source_path),
                "target_path": str(candidate),
                "outputs": {
                    profile["id"]: str(
                        work_dir / "route_renders" / candidate_id
                        / f"region_{region_index:02d}.wav"
                    )
                },
            })
    manifest = {
        "schema": "hmo.xvc-target-screen-render-request.v1",
        "xvc": {
            "directory": str(xvc_dir),
            "config": str(config),
            "checkpoint": str(checkpoint),
            "device": args.xvc_device,
            "ema_load": True,
        },
        "profiles": [profile],
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
    child_env = os.environ.copy()
    child_env.pop("VIRTUAL_ENV", None)
    print(
        f"Rendering {len(candidates)} targets over "
        f"{len(route_sources)} complete production-stream VC route(s)..."
    )
    subprocess.run(command, check=True, env=child_env)
    render_rows = json.loads(result_path.read_text(encoding="utf-8"))["rows"]
    raw_rows = {(row["job_id"], row["profile_id"]): row for row in render_rows}

    result = {}
    route_audio = {}
    for candidate in candidates:
        candidate_key = str(candidate)
        candidate_id = candidate_ids[candidate_key]
        for utterance in utterances:
            utterance_id = utterance["utterance_id"]
            region_index = utterance["region_index"]
            job_id = f"{candidate_id}_region_{region_index:02d}"
            route_row = raw_rows[(job_id, profile["id"])]
            cache_key = (candidate_key, region_index)
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
                work_dir / "utterance_renders" / candidate_id / f"{utterance_id}.wav"
            )
            _write_event_slice(destination, rate, audio, relative_start, relative_end)
            result[(candidate_key, utterance_id)] = {
                **route_row,
                "output_path": str(destination),
            }
    return result


def _score_candidates(
    candidates: list[Path],
    candidate_root: Path,
    utterances: list[dict],
    render_rows: dict[tuple[str, str], dict],
    work_dir: Path,
    verbose: bool,
) -> dict:
    from vc_quality import evaluate_conversion, naturalness, transcribe_for_evaluation

    print("\nScoring raw references and converted utterances...")
    raw_utmos = {
        utterance["utterance_id"]: naturalness(str(utterance["source_path"])).get("utmos")
        for utterance in utterances
    }
    source_aggregate = work_dir / "aggregate" / "source.wav"
    _concatenate([row["source_path"] for row in utterances], source_aggregate)
    source_asr = transcribe_for_evaluation(str(source_aggregate))

    detail = []
    summaries = []
    for candidate in candidates:
        candidate_key = str(candidate)
        label = str(candidate.relative_to(candidate_root))
        outputs = []
        for utterance in utterances:
            utterance_id = utterance["utterance_id"]
            render = render_rows[(candidate_key, utterance_id)]
            converted = Path(render["output_path"])
            outputs.append(converted)
            score = evaluate_conversion(
                str(converted), str(candidate),
                source_path=str(utterance["source_path"]),
                compute_intelligibility=False,
                compute_speaker_similarity=True,
                compute_utmos=True,
            )
            converted_utmos = score.get("utmos")
            raw_score = raw_utmos[utterance_id]
            row = {
                "candidate": label,
                "utterance_id": utterance_id,
                "utmos": converted_utmos,
                "utmos_delta": (
                    converted_utmos - raw_score
                    if converted_utmos is not None and raw_score is not None else None
                ),
                "sim": score.get("sim"),
                "render_rtf": render.get("render_rtf"),
            }
            detail.append(row)
            if verbose:
                print(
                    f"  {label:<32} {utterance_id} UTMOS={_metric(row['utmos'])} "
                    f"delta={_metric(row['utmos_delta'])} SIM={_metric(row['sim'])}"
                )

        converted_aggregate = work_dir / "aggregate" / f"candidate_{len(summaries) + 1:03d}.wav"
        _concatenate(outputs, converted_aggregate)
        aggregate = evaluate_conversion(
            str(converted_aggregate), str(candidate),
            source_path=str(source_aggregate), source_asr=source_asr,
            compute_intelligibility=True,
            compute_speaker_similarity=True,
            compute_utmos=False,
        )
        rows = [row for row in detail if row["candidate"] == label]
        summaries.append({
            "candidate": label,
            "utterances": len(rows),
            "target_utmos": naturalness(str(candidate)).get("utmos"),
            "converted_utmos": _median([row["utmos"] for row in rows]),
            "utmos_delta": _median([row["utmos_delta"] for row in rows]),
            "sim_median": _median([row["sim"] for row in rows]),
            "wer": aggregate.get("wer"),
            "sim_all": aggregate.get("sim"),
            "rtf_median": _median([row["render_rtf"] for row in rows]),
            "wer_error": aggregate.get("wer_error"),
        })

    summaries.sort(key=lambda row: (
        row["converted_utmos"] is None,
        -(row["converted_utmos"] or 0.0),
        row["wer"] is None,
        row["wer"] if row["wer"] is not None else float("inf"),
        -(row["sim_median"] or 0.0),
    ))
    for rank, row in enumerate(summaries, 1):
        row["rank"] = rank
    return {"source_asr": source_asr, "utterance_scores": detail, "summaries": summaries}


def _print_report(session_id: str, result: dict) -> None:
    candidate_width = max(24, max(len(row["candidate"]) for row in result["summaries"]))
    width = candidate_width + 78
    print("\n" + "=" * width)
    print(f"X-VC TARGET SCREENING: {session_id}")
    print("Production stream only: current=120, smooth=20, future=100, gate=0.008")
    print("-" * width)
    print(
        f"{'#':>2} {'Candidate':<{candidate_width}} {'N':>3} {'Target':>7} "
        f"{'UTMOS':>7} {'Delta':>7} {'SIM-med':>8} {'WER':>7} "
        f"{'SIM-all':>8} {'RTF':>6}"
    )
    print("-" * width)
    for row in result["summaries"]:
        print(
            f"{row['rank']:>2} {row['candidate']:<{candidate_width}} "
            f"{row['utterances']:>3} {_metric(row['target_utmos']):>7} "
            f"{_metric(row['converted_utmos']):>7} {_metric(row['utmos_delta']):>7} "
            f"{_metric(row['sim_median']):>8} {_metric(row['wer']):>7} "
            f"{_metric(row['sim_all']):>8} {_metric(row['rtf_median'], 2):>6}"
        )
    print("=" * width)
    print("Ranking is by converted UTMOS, then WER and SIM; confirm finalists by listening.")


def _run(args: argparse.Namespace, work_dir: Path) -> dict:
    prepared = prepare_session_inputs(args, work_dir / "source")
    candidate_root = Path(args.candidates).expanduser().resolve()
    if not candidate_root.is_dir():
        raise FileNotFoundError(f"candidate directory not found: {candidate_root}")
    candidates = _candidate_wavs(candidate_root)
    if args.max_candidates is not None:
        candidates = candidates[:args.max_candidates]
    if not candidates:
        raise ValueError(f"no WAV candidates found under {candidate_root}")

    render_rows = _render_candidates(
        candidates,
        prepared["utterances"],
        prepared["route_sources"],
        work_dir,
        args,
    )
    result = _score_candidates(
        candidates,
        candidate_root,
        prepared["utterances"],
        render_rows,
        work_dir,
        args.verbose,
    )
    result.update({
        "schema": "hmo.xvc-target-screening.v1",
        "session_id": args.session,
        "candidate_root": str(candidate_root),
    })
    _print_report(args.session, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank target WAVs using the frozen production X-VC stream."
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--data-root", default="/workspace/data/media")
    parser.add_argument("--session-dir")
    parser.add_argument("--guard-s", type=float, default=0.5)
    parser.add_argument("--padding-s", type=float, default=0.2)
    parser.add_argument("--min-utterance-s", type=float, default=0.8)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--xvc-dir", default=str(_default_xvc_dir()))
    parser.add_argument("--xvc-config")
    parser.add_argument("--xvc-ckpt")
    parser.add_argument("--xvc-device", type=int, default=0)
    parser.add_argument("--quality-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", help="retain renders and JSON here; directory must not exist")
    args = parser.parse_args()
    if args.max_candidates is not None and args.max_candidates <= 0:
        parser.error("--max-candidates must be > 0")
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
        (work_dir / "target_screening_results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(f"Retained target-screening artifacts: {work_dir}")
    else:
        with tempfile.TemporaryDirectory(prefix="hmo-target-screening-") as temporary:
            _run(args, Path(temporary))
        print("Temporary renders removed; study data and candidate WAVs were unchanged.")


if __name__ == "__main__":
    main()
