"""Post-process soundboard-audit runs into analysis-ready tables.

Run as:  python -m study.audit_postprocess [--run NAME] [--force]
(cwd = services/app_api, under the app-api venv; STUDY_DATA_ROOT may point at
a copied data root.)

For every run directory under STUDY_DATA_ROOT/audit/ this:
  1. verifies complete input delivery per presentation (the logged
     sent-clip SHA-256 must equal the manifest's frozen bake hash);
  2. verifies matched pairing (presentations grouped by raw-source hash;
     each item must appear in a natural and a converted manipulation);
  3. transcribes each presentation's PersonaPlex response WAV (Whisper via
     the existing metrics stack) into transcripts/NNN_slot.json;
  4. computes overlap-during-clip and premature-onset candidates from the
     logged PP speech events and the clip window;
  5. writes audit_summary.csv (one row per item x condition) and
     audit_summary.json next to the run's results.zip.

Everything derives from the frozen manifest + immutable run log; reruns with
--force only ever ADD files, never mutate the originals.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Transcriber = Callable[[str], dict]


def _default_transcriber(path: str) -> dict:
    from metrics import get_transcript_result

    return get_transcript_result(path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _load_run_inputs(run_dir: Path) -> tuple[dict, dict] | None:
    """Manifest + run log, from the unpacked copies or from the zip."""
    manifest = _read_json(run_dir / "manifest.json")
    log = _read_json(run_dir / "run_log.json")
    if manifest is None or log is None:
        zip_path = run_dir / "results.zip"
        if not zip_path.exists():
            return None
        try:
            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
                if manifest is None and "manifest.json" in names:
                    manifest = json.loads(archive.read("manifest.json"))
                if log is None and "run_log.json" in names:
                    log = json.loads(archive.read("run_log.json"))
        except zipfile.BadZipFile:
            return None
    if manifest is None or log is None:
        return None
    return manifest, log


def _overlap_stats(record: dict) -> dict:
    """Overlap of PP speech with the clip window, from the logged events."""
    start = record.get("t_play_start_ms")
    end = record.get("t_play_end_ms")
    if start is None or end is None:
        return {"overlap_ms": None, "pp_onsets_during_clip": None}
    events = record.get("pp_speech_events") or []
    # Reconstruct speech runs from the event stream.
    runs: list[tuple[float, float]] = []
    open_start: float | None = None
    for event in events:
        ts = float(event.get("timestampMs") or 0)
        if event.get("type") == "pp_speech_start":
            open_start = ts
        elif event.get("type") == "pp_speech_end" and open_start is not None:
            runs.append((open_start, ts))
            open_start = None
    if open_start is not None:
        runs.append((open_start, float(end)))
    overlap = sum(max(0.0, min(b, end) - max(a, start)) for a, b in runs)
    onsets = sum(1 for a, _ in runs if start <= a < end)
    return {"overlap_ms": round(overlap, 1), "pp_onsets_during_clip": onsets}


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(mean, 1), round(sd, 1)


def process_run(run_dir: Path, transcriber: Transcriber | None = None,
                force: bool = False) -> dict | None:
    summary_path = run_dir / "audit_summary.json"
    if summary_path.exists() and not force:
        return None
    loaded = _load_run_inputs(run_dir)
    if loaded is None:
        return {"run": run_dir.name, "error": "manifest or run log unavailable"}
    manifest, log = loaded
    presentations = {int(p["index"]): p for p in manifest.get("presentations") or []}
    records = list(log.get("records") or [])
    transcribe = transcriber or _default_transcriber

    # ---- per-presentation table -------------------------------------------
    zip_path = run_dir / "results.zip"
    transcripts_dir = run_dir / "transcripts"
    rows: list[dict] = []
    archive_cm = (zipfile.ZipFile(zip_path) if zip_path.exists()
                  else zipfile.ZipFile(io.BytesIO(), "w"))
    with archive_cm as archive:
        names = set(archive.namelist())
        for record in records:
            presentation = presentations.get(int(record.get("index") or 0), {})
            prefix = (f"runs/{int(record['index']):03d}_"
                      f"{record.get('slot_id')}")
            # 1. Delivery: logged sent bytes vs frozen bake hash.
            sent = record.get("sent_clip_sha256")
            frozen = presentation.get("clip_sha256")
            delivery_verified = (sent is not None and frozen is not None
                                 and sent == frozen)
            # 3. PP response transcript.
            transcript_text = None
            wav_name = f"{prefix}/personaplex.wav"
            transcript_path = transcripts_dir / f"{prefix.split('/', 1)[1]}.json"
            if transcript_path.exists() and not force:
                cached = _read_json(transcript_path)
                transcript_text = (cached or {}).get("text")
            elif wav_name in names:
                transcripts_dir.mkdir(parents=True, exist_ok=True)
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
                    handle.write(archive.read(wav_name))
                    handle.flush()
                    try:
                        asr = transcribe(handle.name)
                        transcript_text = str(asr.get("text") or "").strip()
                        transcript_path.write_text(json.dumps({
                            "text": transcript_text,
                            "status": asr.get("status"),
                            "source": wav_name,
                        }, indent=2))
                    except Exception as exc:  # noqa: BLE001 - keep processing
                        transcript_text = None
                        transcript_path.write_text(json.dumps({
                            "text": None, "status": "failed",
                            "error": str(exc), "source": wav_name,
                        }, indent=2))
            rows.append({
                "index": record.get("index"),
                "label": record.get("label"),
                "manipulation": presentation.get("manipulation"),
                "engine": presentation.get("engine"),
                "raw_sha256": presentation.get("raw_sha256"),
                "presentation_mode": presentation.get("presentation_mode"),
                "status": record.get("status"),
                "delivery_verified": delivery_verified,
                "response_latency_ms": record.get("response_latency_ms"),
                "pp_yield_latency_ms": record.get("pp_yield_latency_ms"),
                **_overlap_stats(record),
                "pp_response_transcript": transcript_text,
            })

    # ---- 2. matched-pair verification --------------------------------------
    by_raw: dict[str, set[str]] = {}
    for presentation in presentations.values():
        raw = presentation.get("raw_sha256")
        if raw:
            by_raw.setdefault(raw, set()).add(str(presentation.get("manipulation")))
    pairing_failures = [
        {"raw_sha256": raw, "manipulations": sorted(manipulations)}
        for raw, manipulations in sorted(by_raw.items())
        if not ("vc" in manipulations and len(manipulations) > 1)
    ]

    # ---- item x condition summary -------------------------------------------
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("raw_sha256") or row["label"]),
               str(row.get("manipulation")))
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for (item, manipulation), items in sorted(grouped.items()):
        latencies = [r["response_latency_ms"] for r in items
                     if r.get("response_latency_ms") is not None]
        yields = [r["pp_yield_latency_ms"] for r in items
                  if r.get("pp_yield_latency_ms") is not None]
        latency_mean, latency_sd = _mean_sd(latencies)
        yield_mean, _ = _mean_sd(yields)
        summary_rows.append({
            "item_raw_sha256": item[:12],
            "labels": "; ".join(sorted({r["label"] for r in items})),
            "manipulation": manipulation,
            "n": len(items),
            "n_ok": sum(1 for r in items if r["status"] == "ok"),
            "n_no_response": sum(1 for r in items if r["status"] == "no_response"),
            "n_delivery_verified": sum(1 for r in items if r["delivery_verified"]),
            "response_latency_mean_ms": latency_mean,
            "response_latency_sd_ms": latency_sd,
            "yield_latency_mean_ms": yield_mean,
            "overlap_ms_total": round(sum(r["overlap_ms"] or 0 for r in items), 1),
            "pp_onsets_during_clip": sum(r["pp_onsets_during_clip"] or 0 for r in items),
        })

    summary = {
        "schema": "hmo.soundboard-audit-summary.v1",
        "run": run_dir.name,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_schema": manifest.get("schema"),
        "generated_at_unix_s": time.time(),
        "presentations": rows,
        "pairing_failures": pairing_failures,
        "item_condition_summary": summary_rows,
        "counts": {
            "presentations": len(rows),
            "ok": sum(1 for r in rows if r["status"] == "ok"),
            "delivery_verified": sum(1 for r in rows if r["delivery_verified"]),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    if summary_rows:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
        (run_dir / "audit_summary.csv").write_text(buffer.getvalue())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m study.audit_postprocess")
    parser.add_argument("--run", default=None,
                        help="process only this run directory name")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    data_root = Path(os.path.expanduser(
        os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))
    audit_root = data_root / "audit"
    if not audit_root.exists():
        print(json.dumps({"error": f"no audit directory at {audit_root}"}))
        return
    processed, skipped = [], []
    for run_dir in sorted(audit_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if args.run and run_dir.name != args.run:
            continue
        result = process_run(run_dir, force=args.force)
        if result is None:
            skipped.append(run_dir.name)
        else:
            processed.append({"run": run_dir.name,
                              "counts": result.get("counts"),
                              "pairing_failures": len(result.get("pairing_failures") or []),
                              "error": result.get("error")})
    print(json.dumps({"processed": processed, "skipped_existing": skipped},
                     indent=2))


if __name__ == "__main__":
    main()
