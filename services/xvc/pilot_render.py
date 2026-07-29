"""Render matched X-VC pilot variants outside the live study pipeline.

This script runs in the X-VC service environment. It accepts a JSON manifest
created by ``services/vc_quality/pilot_calibration.py`` and writes only to the
explicit output paths in that manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _fit_length(pcm, samples, np):
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if len(pcm) >= samples:
        return pcm[:samples]
    return np.pad(pcm, (0, samples - len(pcm)), mode="constant")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render X-VC pilot variants.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    xvc = manifest["xvc"]
    os.environ["XVC_DIR"] = xvc["directory"]
    os.environ["XVC_CONFIG"] = xvc["config"]
    os.environ["XVC_CKPT"] = xvc["checkpoint"]
    os.environ["XVC_DEVICE"] = str(xvc.get("device", 0))
    # X-VC config entries such as ``pretrained/...`` are intentionally
    # repository-relative. Match the live launcher before Hydra resolves them.
    os.chdir(xvc["directory"])

    # Imports must follow XVC_DIR setup because server.py resolves X-VC modules
    # while it is imported.
    import numpy as np
    import torch
    import torchaudio

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import server
    from bins.infer_utils import load_pair_as_tensors, load_xvc, run_offline

    server.cfg, server.model, server.device = load_xvc(
        xvc["config"], xvc["checkpoint"], int(xvc.get("device", 0)),
        bool(xvc.get("ema_load", True)),
    )
    server.SR = int(server.cfg["sample_rate"])
    server.HP_CUT = float(server.cfg.get("highpass_cutoff_freq", 0.0))
    server.MASK_TARGET_COND = bool(
        server.cfg.get("dataloader", {}).get("mask_target_condition", True)
    )

    default_target_path = manifest.get("target_path")
    target_conditions = {}
    profile_by_id = {row["id"]: row for row in manifest["profiles"]}
    results = []

    for job in manifest["jobs"]:
        source_path = job["source_path"]
        target_path = job.get("target_path", default_target_path)
        if not target_path:
            raise ValueError(f"target_path is missing for {job['job_id']}")
        source_wave, source_rate = torchaudio.load(source_path)
        if source_wave.numel() == 0:
            raise ValueError(f"source audio is empty: {source_path}")
        source_mono = source_wave.mean(dim=0)
        source_rate = int(source_rate)
        exact_samples = round(source_mono.shape[-1] * server.SR / source_rate)
        if source_rate != server.SR:
            source_mono = torchaudio.functional.resample(
                source_mono, source_rate, server.SR
            )
        source_16k = _fit_length(source_mono.numpy(), exact_samples, np)

        for profile_id, output_path in job["outputs"].items():
            profile = profile_by_id[profile_id]
            started = time.perf_counter()
            if profile["mode"] == "offline":
                source_tensor, target_tensor, target_condition = load_pair_as_tensors(
                    source_path,
                    target_path,
                    server.cfg,
                    server.device,
                    int(server.cfg["latent_hop_length"]),
                    server.MASK_TARGET_COND,
                )
                converted = run_offline(
                    server.model, source_tensor, target_tensor, target_condition
                )
                output = _fit_length(
                    converted.squeeze().detach().cpu().numpy(), exact_samples, np
                )
                inference_windows = 1
                silence_bypassed_windows = 0
                output_windows = 1
            elif profile["mode"] == "streaming":
                if target_path not in target_conditions:
                    _, speaker_condition, frame_condition = server._target_conditions(
                        target_path
                    )
                    target_conditions[target_path] = (
                        speaker_condition, frame_condition
                    )
                speaker_condition, frame_condition = target_conditions[target_path]
                session = server.XVCStreamSession(speaker_condition, frame_condition)
                current_samples = session.current_ms * server.SR // 1000
                window_count = max(1, int(np.ceil(exact_samples / current_samples)))
                required_samples = (
                    (window_count - 1) * session.current_ms
                    + session.current_ms
                    + session.smooth_ms
                    + session.future_ms
                ) * server.SR // 1000
                padded = _fit_length(source_16k, required_samples, np)
                chunks = session.feed(padded)
                if not chunks:
                    raise RuntimeError(
                        f"X-VC produced no windows for {job['job_id']}"
                    )
                output = _fit_length(np.concatenate(chunks), exact_samples, np)
                inference_windows = session.inference_windows
                silence_bypassed_windows = session.silence_bypassed_windows
                output_windows = len(chunks)
            else:
                raise ValueError(f"unsupported profile mode: {profile['mode']}")

            elapsed_s = time.perf_counter() - started
            output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            server._save_wav(str(destination), output, server.SR)
            duration_s = exact_samples / server.SR
            results.append({
                "job_id": job["job_id"],
                "profile_id": profile_id,
                "output_path": str(destination),
                "sample_rate_hz": server.SR,
                "samples": exact_samples,
                "duration_s": duration_s,
                "render_seconds": elapsed_s,
                "render_rtf": elapsed_s / duration_s if duration_s else None,
                "output_windows": output_windows,
                "inference_windows": inference_windows,
                "silence_bypassed_windows": silence_bypassed_windows,
            })
            print(
                f"[pilot-render] {job['job_id']} {profile_id}: "
                f"{elapsed_s:.2f}s (RTF {elapsed_s / duration_s:.2f})",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    Path(args.results).write_text(
        json.dumps({"schema": "hmo.xvc-pilot-render.v1", "rows": results}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
