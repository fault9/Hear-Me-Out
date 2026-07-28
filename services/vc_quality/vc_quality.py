"""
VC-quality evaluation (post-hoc).

This module measures *voice-conversion quality* — i.e. how good the converted
audio is — which is distinct from tools/metrics.py, which measures how the LLM
*responded* (sentiment / semantic / aesthetics of responses). The two are
complementary: VC-quality here tells you whether a difference in LLM behaviour
could be attributed to degraded converted audio rather than to voice identity.

Designed to run AFTER an interaction (or as an offline batch over a folder of
recordings), never in the real-time path. The ASR and speaker-embedding passes
are too heavy for the latency-critical loop, and the study evaluates stable VC
regions rather than individual streaming frames.

The objective battery follows the X-VC evaluation:
  - intelligibility : WER   (jiwer, using source-ASR for free speech)
  - speaker identity: SIM   (cosine, WavLM-large + SV head)
  - naturalness     : UTMOS (torch.hub, CPU)

Streaming latency and RTF come from runtime telemetry, not this post-hoc audio
scorer. Route-switch boundary measures remain in study/transition_analysis.py.

Segment-level evaluation: pass `segment_mode="fixed"` (or "word"/"vad") to
`evaluate_conversion` to additionally produce per-segment scores and a
list of anomaly windows alongside the whole-clip headline numbers. Heavy
intermediates are computed once per call and reused. Target speaker embeddings
are also cached across calls in a batch.

Runtime defaults (deployed server has a 97%-full RTX 3090 + 3 live services):
  - CPU-only by default. Override only on a dedicated eval host via env
    VC_QUALITY_DEVICE=cuda.
  - All heavy / optional deps (jiwer, SV module, UTMOS via torch.hub) are
    lazy-imported. Each metric degrades to None with
    a printed reason if a dep or resource is missing, so the module imports
    cleanly even on a fresh clone and individual metrics are testable
    incrementally.

Reuses the project's existing choices: Whisper-small for ASR (as in
tools/metrics.py) and the WavLM-large speaker model via init_model (as in
src.runtime.speaker_verification.verification on the deployed server).

UTMOS note: loaded via torch.hub from tarepan/SpeechMOS (pinned tag), runs on
CPU. First call downloads ~400MB into ~/.cache/torch/hub; subsequent calls are
fully offline. The predictor takes a (batch, time) waveform tensor and the
source sample rate and returns a scalar MOS per item.
"""

from __future__ import annotations

import hashlib
import json
import os
import string
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import librosa
import torch

# Speaker similarity model. This is the SAME init_model the running MeanVC
# server imports and that MeanVC/src/eval/verification.py uses: a WavLM-large
# backbone with an ECAPA-TDNN head, loaded from wavlm_large_finetune.pth.
# Importing it keeps the WavLM-based SIM implementation consistent.
#   from src.runtime.speaker_verification.verification import init_model
# Imported lazily inside _get_sv_model so this module loads even when the SV
# checkpoint / path isn't configured.

SAMPLE_RATE = 16000

# CPU by default. The eval is post-hoc and must NOT compete with the live LLM
# (which holds ~97% of the GPU). Override on a dedicated eval host with
# VC_QUALITY_DEVICE=cuda.
DEVICE = os.environ.get("VC_QUALITY_DEVICE", "cpu")

# torch.hub spec for UTMOS. Pinned tag so future repo edits upstream don't
# change scores under us. Override with UTMOS_HUB_SPEC if needed.
UTMOS_HUB_SPEC = os.environ.get("UTMOS_HUB_SPEC", "tarepan/SpeechMOS:v1.2.0")
UTMOS_MODEL_NAME = os.environ.get("UTMOS_MODEL_NAME", "utmos22_strong")

# Module-level singletons so batch runs don't reload models per file.
_asr_pipeline = None
_sv_model = None
_utmos_predictor = None
_target_sv_embeddings: dict[str, torch.Tensor] = {}

# Minimum segment duration (seconds) for each segment-aware metric. Shorter
# slices are skipped (score=None) because the underlying models become brittle:
#   - WavLM-large SV: trained on >=1s; <0.5s embeddings are noisy
#   - UTMOS22:        trained on ~5-10s; <0.8s scores are noisy
# Segment-level scoring is diagnostic only; the study worker uses whole regions.
_MIN_SEG_S = {"sim": 0.5, "utmos": 0.8}


# --------------------------------------------------------------------------
# Model loaders (lazy, cached)
# --------------------------------------------------------------------------
def _get_asr():
    global _asr_pipeline
    if _asr_pipeline is None:
        try:
            from transformers import pipeline
        except ImportError as e:
            print(f"[vc_quality] transformers not available ({e}); "
                  f"transcription will be skipped.")
            return None
        _asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=DEVICE,
        )
    return _asr_pipeline


def _get_sv_model(sv_ckpt_path=None, sv_root=None):
    """Load the project's WavLM-large speaker-verification model once.

    sv_ckpt_path : path to wavlm_large_finetune.pth
                   (default: env MEANVC_SV_CKPT)
    sv_root      : repo root so `from src.runtime...` resolves
                   (default: env SPEAKER_VERIFICATION_ROOT)
    """
    global _sv_model
    if _sv_model is not None:
        return _sv_model

    sv_ckpt_path = sv_ckpt_path or os.environ.get("MEANVC_SV_CKPT")
    sv_root = sv_root or os.environ.get("SPEAKER_VERIFICATION_ROOT", os.getcwd())

    if not sv_ckpt_path or not os.path.exists(sv_ckpt_path):
        print(f"[vc_quality] SV checkpoint not found ({sv_ckpt_path}); "
              f"SIM will be skipped.")
        return None

    try:
        import sys
        if sv_root not in sys.path:
            sys.path.insert(0, sv_root)
        from src.runtime.speaker_verification.verification import init_model
        model = init_model("wavlm_large", sv_ckpt_path)
        model.eval().to(DEVICE)
        _sv_model = model
        return _sv_model
    except Exception as e:
        print(f"[vc_quality] SV load failed ({e}); SIM will be skipped.")
        return None


def _get_utmos():
    """Load the UTMOS22 predictor from torch.hub once, on CPU. Returns the
    nn.Module, or None if torch.hub can't fetch / load it (UTMOS then degrades
    to None). First call downloads ~400MB into
    ~/.cache/torch/hub; subsequent calls are offline."""
    global _utmos_predictor
    if _utmos_predictor is not None:
        return _utmos_predictor

    try:
        predictor = torch.hub.load(
            UTMOS_HUB_SPEC, UTMOS_MODEL_NAME, trust_repo=True
        )
        predictor.eval()
        # Pin to CPU regardless of VC_QUALITY_DEVICE so post-hoc scoring does
        # not compete with the live conversation services.
        predictor.to("cpu")
        _utmos_predictor = predictor
        return _utmos_predictor
    except Exception as e:
        print(f"[vc_quality] UTMOS load failed ({e}); UTMOS will be skipped.")
        return None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _load_16k_mono(path):
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return wav


def _normalize_text(s):
    s = s.lower().strip()
    return s.translate(str.maketrans("", "", string.punctuation))


def _asr_run(path):
    """Run Whisper on a file, requesting word-level timestamps. Returns the
    full HF pipeline dict ({"text": str, "chunks": [{"text", "timestamp"}, ...]}),
    or a minimal dict on failure / missing model. Word timestamps may be absent
    (older HF versions or test fakes) — callers must tolerate empty chunks."""
    try:
        asr = _get_asr()
        if asr is None:
            return {"text": "", "chunks": []}
        try:
            res = asr(path, chunk_length_s=30, batch_size=8,
                      return_timestamps="word")
        except (TypeError, ValueError):
            # Fakes / older HF: fall back to a plain call.
            res = asr(path, chunk_length_s=30, batch_size=8)
        if not isinstance(res, dict):
            return {"text": "", "chunks": []}
        res.setdefault("chunks", [])
        res.setdefault("text", "")
        return res
    except Exception as e:
        print(f"[vc_quality] transcription failed for {path}: {e}")
        return {"text": "", "chunks": []}


def _transcribe(path):
    """Back-compat: return just the text."""
    return _asr_run(path).get("text", "")


def _slice_wav(wav, start_s, end_s):
    s = max(0, int(start_s * SAMPLE_RATE))
    e = min(len(wav), int(end_s * SAMPLE_RATE))
    return wav[s:e]


# --------------------------------------------------------------------------
# Per-clip data container (load + cache intermediates ONCE per evaluation)
# --------------------------------------------------------------------------
@dataclass
class _ClipData:
    """Lazy cache of everything that's expensive per clip and reused across
    whole-clip and per-segment passes. Each `_ensure_*` helper populates a
    field on first access and returns it; subsequent calls are free.

    Holding these on a per-clip object (rather than module-level) is what
    lets evaluate_conversion guarantee 'load once, score many' without
    threading nullable parameters through every metric signature.
    """
    path: str
    _wav: np.ndarray | None = None
    _duration: float | None = None
    _asr: dict | None = None
    _sv_emb: torch.Tensor | None = None
    # Per-segment SV embeddings keyed by (start, end) so repeated segment
    # lookups in different metrics share the same embedding.
    _seg_sv_emb: dict[tuple[float, float], torch.Tensor | None] = field(
        default_factory=dict
    )


def _ensure_wav(cd: _ClipData) -> np.ndarray:
    if cd._wav is None:
        cd._wav = _load_16k_mono(cd.path)
        cd._duration = len(cd._wav) / SAMPLE_RATE
    return cd._wav


def _duration_s(cd: _ClipData) -> float:
    if cd._duration is None:
        _ensure_wav(cd)
    return cd._duration or 0.0


def _ensure_asr(cd: _ClipData) -> dict:
    if cd._asr is None:
        cd._asr = _asr_run(cd.path)
    return cd._asr


def _ensure_sv_emb(cd: _ClipData, model) -> torch.Tensor | None:
    if cd._sv_emb is None:
        try:
            wav = _ensure_wav(cd)
            t = torch.from_numpy(wav).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                cd._sv_emb = model(t)
        except Exception as e:
            print(f"[vc_quality] SV embed failed for {cd.path}: {e}")
            return None
    return cd._sv_emb


def _ensure_target_sv_emb(cd: _ClipData, model) -> torch.Tensor | None:
    """Reuse immutable target embeddings across manifest rows in one batch.

    Sessions freeze separate copies of the same two target WAVs, so content
    hashes deduplicate those copies while preserving artifact immutability.
    """
    try:
        with open(cd.path, "rb") as stream:
            key = hashlib.sha256(stream.read()).hexdigest()
    except OSError:
        return _ensure_sv_emb(cd, model)
    if key not in _target_sv_embeddings:
        embedding = _ensure_sv_emb(cd, model)
        if embedding is None:
            return None
        _target_sv_embeddings[key] = embedding.detach()
    return _target_sv_embeddings[key]


def _segment_sv_emb(cd: _ClipData, model, start_s, end_s) -> torch.Tensor | None:
    """Embed a slice of the converted clip via the SV model. Cached on the
    _ClipData so repeated lookups across metrics don't recompute."""
    key = (round(start_s, 3), round(end_s, 3))
    if key in cd._seg_sv_emb:
        return cd._seg_sv_emb[key]
    if (end_s - start_s) < _MIN_SEG_S["sim"]:
        cd._seg_sv_emb[key] = None
        return None
    try:
        slice_wav = _slice_wav(_ensure_wav(cd), start_s, end_s)
        t = torch.from_numpy(slice_wav).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            emb = model(t)
        cd._seg_sv_emb[key] = emb
        return emb
    except Exception as e:
        print(f"[vc_quality] SV segment embed failed [{start_s:.2f},{end_s:.2f}]: {e}")
        cd._seg_sv_emb[key] = None
        return None

# --------------------------------------------------------------------------
# Segmentation strategies + anomaly flagging
# --------------------------------------------------------------------------
def _segments_fixed(duration: float, win: float = 2.0,
                    hop: float = 1.0) -> list[tuple[float, float]]:
    """Sliding-window segments. Returns a single (0, duration) window for
    clips shorter than `win`. Always includes a tail window so the last frames
    are covered."""
    if duration <= 0:
        return []
    if duration <= win:
        return [(0.0, duration)]
    segs: list[tuple[float, float]] = []
    t = 0.0
    while t + win <= duration:
        segs.append((round(t, 3), round(t + win, 3)))
        t += hop
    if segs[-1][1] < duration - 1e-6:
        segs.append((round(max(0.0, duration - win), 3), round(duration, 3)))
    return segs


def _segments_from_whisper(asr_result: dict) -> list[tuple[float, float]]:
    """One window per word from a Whisper word-timestamp result. Chunks with
    missing/None timestamps are skipped (Whisper sometimes emits these on
    very short or noisy clips)."""
    out: list[tuple[float, float]] = []
    for chunk in asr_result.get("chunks", []) or []:
        ts = chunk.get("timestamp")
        if not ts or len(ts) != 2:
            continue
        s, e = ts
        if s is None or e is None or e <= s:
            continue
        out.append((float(s), float(e)))
    return out


def _segments_vad(wav: np.ndarray) -> list[tuple[float, float]]:
    """Speech segments via silero-vad (torch.hub, lazy). Returns [] if
    silero-vad isn't reachable so the caller can fall back."""
    try:
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
    except Exception as e:
        print(f"[vc_quality] silero-vad unavailable ({e}); VAD segmentation skipped.")
        return []
    try:
        get_speech_timestamps = utils[0]
        wav_t = torch.from_numpy(wav).float()
        ts = get_speech_timestamps(wav_t, model, sampling_rate=SAMPLE_RATE)
        return [(t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE) for t in ts]
    except Exception as e:
        print(f"[vc_quality] silero-vad call failed ({e}); VAD segmentation skipped.")
        return []


def _words_in_range(asr_result: dict, start_s: float, end_s: float) -> str:
    """Join whisper words whose midpoint falls in [start_s, end_s]. Empty
    string if the result has no usable word timestamps."""
    parts: list[str] = []
    for chunk in asr_result.get("chunks", []) or []:
        ts = chunk.get("timestamp")
        if not ts or len(ts) != 2 or ts[0] is None or ts[1] is None:
            continue
        mid = (float(ts[0]) + float(ts[1])) / 2.0
        if start_s <= mid <= end_s:
            parts.append(str(chunk.get("text", "")).strip())
    return _normalize_text(" ".join(p for p in parts if p))


def _flag_anomalies(
    segments_with_scores: list[dict],
    z_threshold: float = 2.0,
    hard_floor: dict | None = None,
    higher_is_worse: tuple[str, ...] = ("wer",),
) -> list[dict]:
    """Flag segments where a metric is an outlier vs the per-clip distribution
    (|z| >= z_threshold) OR violates an absolute floor. Returns one entry per
    (segment, offending metric) so a single bad segment can be flagged for
    multiple reasons.

    `hard_floor` keys map a metric name to an absolute threshold under which
    the segment is flagged regardless of z-score. For metrics in
    `higher_is_worse` the comparison is reversed (the floor is treated as a
    ceiling and z>+threshold is the outlier direction)."""
    if hard_floor is None:
        hard_floor = {
            "sim": 0.5, "utmos": 2.5,
            # WER ceiling: anything above 0.5 in a segment is suspicious
            "wer": 0.5,
        }
    if not segments_with_scores:
        return []

    # Discover numeric keys present in at least one segment.
    keys: set[str] = set()
    for seg in segments_with_scores:
        for k, v in seg.items():
            if k in ("start", "end"):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)

    flagged: list[dict] = []
    for metric in keys:
        scores = np.array(
            [s.get(metric) if isinstance(s.get(metric), (int, float))
             else np.nan for s in segments_with_scores],
            dtype=float,
        )
        if np.all(np.isnan(scores)):
            continue
        mean = float(np.nanmean(scores))
        std = float(np.nanstd(scores))
        floor = hard_floor.get(metric)
        for i, seg in enumerate(segments_with_scores):
            v = seg.get(metric)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            score = float(v)
            z = (score - mean) / (std + 1e-9)
            if metric in higher_is_worse:
                bad = z > z_threshold or (floor is not None and score > floor)
            else:
                bad = z < -z_threshold or (floor is not None and score < floor)
            if bad:
                flagged.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "metric": metric,
                    "score": score,
                    "z": float(z),
                })
    return flagged


# --------------------------------------------------------------------------
# Core eval functions — take precomputed _ClipData + optional segment list
# --------------------------------------------------------------------------
def _intelligibility_eval(
    conv: _ClipData,
    ref_text: str | None,
    src: _ClipData | None,
    segments: list[tuple[float, float]] | None = None,
) -> dict:
    """WER on the whole clip. If `segments` is given, also compute per-
    segment WER using Whisper word timestamps on the converted clip and
    (when available) word-aligned reference text from `src`."""
    conv_asr = _ensure_asr(conv)
    vc_text = _normalize_text(conv_asr.get("text", ""))

    if ref_text is not None:
        ref = _normalize_text(ref_text)
        ref_kind = "ground_truth"
        src_asr = _ensure_asr(src) if (src is not None and segments) else None
    elif src is not None:
        src_asr = _ensure_asr(src)
        ref = _normalize_text(src_asr.get("text", ""))
        ref_kind = "source_asr_free_speech"
    else:
        return {
            "wer": None,
            "vc_transcript": vc_text, "ref_transcript": None, "ref_kind": None,
            "wer_segments": None,
        }

    try:
        import jiwer
    except ImportError:
        print("[vc_quality] jiwer not installed; WER skipped "
              "(install on a non-shared venv to enable).")
        return {
            "wer": None,
            "vc_transcript": vc_text, "ref_transcript": ref, "ref_kind": ref_kind,
            "wer_segments": None,
        }

    try:
        wer = jiwer.wer(ref, vc_text) if (ref or vc_text) else 0.0
    except Exception as e:
        print(f"[vc_quality] WER failed: {e}")
        wer = None

    wer_segments: list[dict] | None = None
    if segments is not None:
        wer_segments = []
        for s, e in segments:
            vc_slice = _words_in_range(conv_asr, s, e)
            ref_slice = (_words_in_range(src_asr, s, e)
                         if src_asr is not None else None)
            seg_wer: float | None
            if ref_slice is None or (not vc_slice and not ref_slice):
                seg_wer = None
            else:
                try:
                    seg_wer = jiwer.wer(ref_slice, vc_slice)
                except Exception:
                    seg_wer = None
            wer_segments.append({
                "start": s, "end": e, "wer": seg_wer,
                "vc": vc_slice, "ref": ref_slice,
            })

    reference_note = (
        "WER compares converted-audio ASR against raw-source ASR because the "
        "interaction is unscripted."
        if ref_kind == "source_asr_free_speech" else None
    )
    return {
        "wer": wer,
        "vc_transcript": vc_text, "ref_transcript": ref, "ref_kind": ref_kind,
        "wer_reference_note": reference_note,
        "wer_segments": wer_segments,
    }


def _speaker_similarity_eval(
    conv: _ClipData,
    tgt: _ClipData,
    sv_ckpt_path: str | None = None,
    sv_root: str | None = None,
    segments: list[tuple[float, float]] | None = None,
) -> dict:
    """X-VC SIM for the whole clip and optional diagnostic segments.
    The target embedding is computed exactly once and reused for every
    per-segment cosine."""
    model = _get_sv_model(sv_ckpt_path, sv_root)
    if model is None:
        return {"sim": None, "sim_segments": None}

    e_conv = _ensure_sv_emb(conv, model)
    e_tgt = _ensure_target_sv_emb(tgt, model)
    if e_conv is None or e_tgt is None:
        return {"sim": None, "sim_segments": None}

    try:
        whole = float(torch.nn.functional.cosine_similarity(
            e_conv, e_tgt, dim=-1).mean().item())
    except Exception as e:
        print(f"[vc_quality] SIM cosine failed: {e}")
        whole = None

    sim_segments: list[dict] | None = None
    if segments is not None:
        sim_segments = []
        for s, e in segments:
            seg_emb = _segment_sv_emb(conv, model, s, e)
            if seg_emb is None:
                sim_segments.append({"start": s, "end": e, "sim": None})
                continue
            try:
                v = float(torch.nn.functional.cosine_similarity(
                    seg_emb, e_tgt, dim=-1).mean().item())
            except Exception:
                v = None
            sim_segments.append({"start": s, "end": e, "sim": v})

    return {"sim": whole, "sim_segments": sim_segments}


def _utmos_eval(
    conv: _ClipData,
    segments: list[tuple[float, float]] | None = None,
) -> dict:
    """X-VC's reference-free naturalness score."""
    out: dict[str, Any] = {"utmos": None}

    def _f(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    predictor = _get_utmos()

    if predictor is not None:
        try:
            wav = _ensure_wav(conv)
            wav_t = torch.from_numpy(wav).unsqueeze(0).float()
            with torch.no_grad():
                score = predictor(wav_t, SAMPLE_RATE)
            out["utmos"] = _f(score.mean().item())
        except Exception as e:
            print(f"[vc_quality] UTMOS compute failed for {conv.path}: {e}")

    if segments is None:
        out["utmos_segments"] = None
        return out

    wav = _ensure_wav(conv)
    seg_rows: list[dict] = []
    for s, e in segments:
        row: dict[str, Any] = {"start": s, "end": e, "utmos": None}
        slice_wav = _slice_wav(wav, s, e)
        eligible = len(slice_wav) / SAMPLE_RATE >= _MIN_SEG_S["utmos"]
        if predictor is not None and eligible:
            try:
                tensor = torch.from_numpy(slice_wav).unsqueeze(0).float()
                with torch.no_grad():
                    score = predictor(tensor, SAMPLE_RATE)
                row["utmos"] = _f(score.mean().item())
            except Exception:
                pass
        seg_rows.append(row)

    out["utmos_segments"] = seg_rows
    return out


# --------------------------------------------------------------------------
# Public file-path wrappers (backward compatible)
# --------------------------------------------------------------------------
def intelligibility(converted_path, source_transcript=None,
                    source_path=None):
    """WER of the converted audio against the source content.

    Prefer a known source_transcript (ground truth). If absent, fall back to
    ASR on the raw source audio and record that provenance explicitly.
    """
    conv = _ClipData(path=converted_path)
    src = _ClipData(path=source_path) if source_path else None
    res = _intelligibility_eval(conv, source_transcript, src, segments=None)
    res.pop("wer_segments", None)
    return res


def speaker_similarity(converted_path, target_path,
                       sv_ckpt_path=None, sv_root=None):
    """SIM: cosine similarity between converted and target speaker
    embeddings (WavLM-large). Returns None if the SV model isn't available."""
    conv = _ClipData(path=converted_path)
    tgt = _ClipData(path=target_path)
    res = _speaker_similarity_eval(conv, tgt, sv_ckpt_path, sv_root,
                                   segments=None)
    res.pop("sim_segments", None)
    return res


def naturalness(converted_path):
    """UTMOS naturalness score used by X-VC."""
    conv = _ClipData(path=converted_path)
    res = _utmos_eval(conv, segments=None)
    res.pop("utmos_segments", None)
    return res


# --------------------------------------------------------------------------
# Top-level orchestrator: load everything once, score whole + segments
# --------------------------------------------------------------------------
_SEGMENT_MODES = ("fixed", "word", "vad", None)


def _choose_segments(
    mode: str | None,
    conv: _ClipData,
    fixed_win: float = 2.0,
    fixed_hop: float = 1.0,
) -> list[tuple[float, float]] | None:
    """Resolve the requested mode into a concrete (start, end) list, falling
    back to fixed windows if the chosen mode produces no usable segments."""
    if mode is None:
        return None
    dur = _duration_s(conv)
    if dur <= 0:
        return None
    if mode == "fixed":
        return _segments_fixed(dur, fixed_win, fixed_hop)
    if mode == "word":
        segs = _segments_from_whisper(_ensure_asr(conv))
        if not segs:
            print("[vc_quality] word-level segments unavailable "
                  "(no Whisper word timestamps); falling back to fixed.")
            return _segments_fixed(dur, fixed_win, fixed_hop)
        return segs
    if mode == "vad":
        segs = _segments_vad(_ensure_wav(conv))
        if not segs:
            print("[vc_quality] VAD segments empty/unavailable; "
                  "falling back to fixed.")
            return _segments_fixed(dur, fixed_win, fixed_hop)
        return segs
    raise ValueError(f"unknown segment_mode {mode!r}, expected one of "
                     f"{[m for m in _SEGMENT_MODES if m]}")


def _merge_segment_scores(
    segments: list[tuple[float, float]],
    *per_metric_lists: list[dict] | None,
) -> list[dict]:
    """Fold per-metric segment lists into one segment-keyed list. Each input
    list has shape [{"start", "end", <metric_keys>}, ...]. Output preserves
    the segment grid and merges metric keys across the inputs."""
    out: list[dict] = []
    for idx, (s, e) in enumerate(segments):
        row: dict[str, Any] = {"start": s, "end": e}
        for lst in per_metric_lists:
            if not lst or idx >= len(lst):
                continue
            item = lst[idx]
            for k, v in item.items():
                if k in ("start", "end"):
                    continue
                row[k] = v
        out.append(row)
    return out


def evaluate_conversion(converted_path, target_path,
                        source_transcript=None, source_path=None,
                        compute_intelligibility=True,
                        compute_speaker_similarity=True,
                        compute_utmos=True,
                        sv_ckpt_path=None, sv_root=None,
                        metadata=None,
                        segment_mode: str | None = None,
                        segment_win: float = 2.0,
                        segment_hop: float = 1.0,
                        anomaly_z: float = 2.0,
                        anomaly_floor: dict | None = None):
    """Score a single (converted, target[, source]) example using X-VC's three
    dimensions. `metadata` (vc_system, llm, target_id, timestamp, code_version,
    ...) is passed through untouched so each row is self-describing for the
    snapshot.

    segment_mode:
      None     — whole-clip only (default; backward-compatible).
      "fixed"  — sliding window (segment_win=2.0s, segment_hop=1.0s by default).
      "word"   — one window per Whisper word; falls back to "fixed" if word
                 timestamps are unavailable.
      "vad"    — silero-VAD speech regions; falls back to "fixed" on failure.

    When segment_mode is set, the returned row gains:
      - segments[]    : per-segment {start, end, wer, sim, utmos},
      - anomalies[]   : per-(segment, metric) outliers via z-score+floor.

    Segment mode is diagnostic and substantially slower than whole-region
    scoring because SIM and UTMOS require an additional pass per window.
    """
    if segment_mode not in _SEGMENT_MODES:
        raise ValueError(f"segment_mode={segment_mode!r} invalid; expected "
                         f"one of {[m for m in _SEGMENT_MODES if m]} or None")

    conv = _ClipData(path=converted_path)
    tgt = _ClipData(path=target_path)
    src = _ClipData(path=source_path) if source_path else None

    segments = _choose_segments(segment_mode, conv, segment_win, segment_hop)

    row: dict[str, Any] = {
        "converted_path": converted_path,
        "target_path": target_path,
        "source_path": source_path,
    }

    wer_segs = sim_segs = utmos_segs = None

    if compute_intelligibility:
        intel = _intelligibility_eval(conv, source_transcript, src, segments)
        wer_segs = intel.pop("wer_segments", None)
        row.update(intel)
    if compute_speaker_similarity:
        sim = _speaker_similarity_eval(conv, tgt, sv_ckpt_path, sv_root, segments)
        sim_segs = sim.pop("sim_segments", None)
        row.update(sim)
    if compute_utmos:
        naturalness = _utmos_eval(conv, segments)
        utmos_segs = naturalness.pop("utmos_segments", None)
        row.update(naturalness)

    if segments is not None:
        seg_rows = _merge_segment_scores(segments, wer_segs, sim_segs, utmos_segs)
        row["segment_mode"] = (
            f"{segment_mode}_{segment_win:g}s_hop{segment_hop:g}s"
            if segment_mode == "fixed" else segment_mode
        )
        row["segments"] = seg_rows
        row["anomalies"] = _flag_anomalies(
            seg_rows, z_threshold=anomaly_z, hard_floor=anomaly_floor,
        )

    if metadata:
        row.update(metadata)
    return row


# --------------------------------------------------------------------------
# Batch over a manifest -> snapshot (JSONL)
# --------------------------------------------------------------------------
def evaluate_manifest(manifest_path, out_path,
                      segment_mode: str | None = None,
                      segment_win: float = 2.0,
                      segment_hop: float = 1.0):
    """manifest_path: JSONL, one object per line with at least
        {"converted_path", "target_path"} and optionally
        {"source_path", "source_transcript", plus any metadata}.
    Writes one result object per line to out_path (append-friendly snapshot).
    Pass segment_mode to enable per-segment scoring + anomaly flagging
    (see evaluate_conversion)."""
    n = 0
    with open(manifest_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            meta = {k: v for k, v in item.items()
                    if k not in {"converted_path", "target_path",
                                 "source_path", "source_transcript"}}
            res = evaluate_conversion(
                converted_path=item["converted_path"],
                target_path=item["target_path"],
                source_transcript=item.get("source_transcript"),
                source_path=item.get("source_path"),
                metadata=meta,
                segment_mode=segment_mode,
                segment_win=segment_win,
                segment_hop=segment_hop,
            )
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            print(f"[vc_quality] scored {n}: {os.path.basename(item['converted_path'])}")
    print(f"[vc_quality] wrote {n} rows -> {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Post-hoc VC-quality evaluation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("one", help="score a single triplet")
    one.add_argument("--converted", required=True)
    one.add_argument("--target", required=True)
    one.add_argument("--source")
    one.add_argument("--source-transcript")
    one.add_argument("--no-intelligibility", action="store_true",
                     help="skip WER (faster Whisper-free run)")
    one.add_argument("--no-speaker-similarity", action="store_true",
                     help="skip SIM (WavLM-large)")
    one.add_argument("--no-utmos", action="store_true",
                     help="skip UTMOS")
    one.add_argument("--segment-mode", choices=["fixed", "word", "vad"],
                     default=None,
                     help="enable per-segment scoring + anomaly flagging")
    one.add_argument("--segment-win", type=float, default=2.0)
    one.add_argument("--segment-hop", type=float, default=1.0)

    batch = sub.add_parser("batch", help="score a JSONL manifest")
    batch.add_argument("--manifest", required=True)
    batch.add_argument("--out", required=True)
    batch.add_argument("--segment-mode", choices=["fixed", "word", "vad"],
                       default=None)
    batch.add_argument("--segment-win", type=float, default=2.0)
    batch.add_argument("--segment-hop", type=float, default=1.0)

    args = p.parse_args()
    if args.cmd == "one":
        res = evaluate_conversion(
            converted_path=args.converted,
            target_path=args.target,
            source_transcript=args.source_transcript,
            source_path=args.source,
            compute_intelligibility=not args.no_intelligibility,
            compute_speaker_similarity=not args.no_speaker_similarity,
            compute_utmos=not args.no_utmos,
            segment_mode=args.segment_mode,
            segment_win=args.segment_win,
            segment_hop=args.segment_hop,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        evaluate_manifest(
            args.manifest, args.out,
            segment_mode=args.segment_mode,
            segment_win=args.segment_win,
            segment_hop=args.segment_hop,
        )
