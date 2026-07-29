"""
Offline pitch + formant shift for the soundboard.

Uses the WORLD vocoder (pyworld) to decompose speech into F0, spectral
envelope (sp), and aperiodicity (ap), then re-synthesizes after independently
scaling F0 and warping the spectral envelope's frequency axis.

Why WORLD and not librosa.pitch_shift:
    librosa.effects.pitch_shift is phase-vocoder based — shifting pitch with
    it ALSO shifts the formants (because it just retunes the whole spectrum).
    For research where the experimental contrast is gender cues, you usually
    want pitch and formants as INDEPENDENT axes. WORLD gives us that, and it
    preserves duration by construction (synth uses the same frame grid as
    analysis, so total samples in == total samples out, modulo small framing).

CRITICAL — format integrity:
    This function MUST NOT change the sample rate. Input SR == output SR.
    Caller is responsible for delivering the WAV at the PP target SR
    (PP_SAMPLE_RATE on the frontend). We assert no silent resampling here.
    Duration is preserved to within a few ms (WORLD frame quantisation);
    caller logs and flags any larger drift.
"""

from __future__ import annotations

import io
import numpy as np
import soundfile as sf

# WORLD pitch range. Defaults from pyworld docs; widened a little so the
# analyser doesn't clip near male/child extremes.
_F0_FLOOR = 60.0
_F0_CEILING = 800.0


def shift_pitch_formant(
    pcm: np.ndarray,
    sr: int,
    semitones: float = 0.0,
    formant_shift: float = 1.0,
) -> np.ndarray:
    """Shift pitch (semitones) and formants (multiplicative ratio).

    formant_shift > 1.0 → higher formants (more "feminine"/child).
    formant_shift < 1.0 → lower formants (more "masculine"/adult).
    formant_shift = 1.0 → unchanged.

    Returns float32 PCM at the same SR and same length (within WORLD's
    framing quantum, which is typically <5ms for short clips).
    """
    import pyworld as pw

    if pcm.ndim != 1:
        raise ValueError(f"shift_pitch_formant expects mono PCM, got shape {pcm.shape}")

    x = pcm.astype(np.float64)
    in_len = len(x)

    # WORLD analysis: F0, spectral envelope (formants live here), aperiodicity.
    f0, t = pw.dio(x, sr, f0_floor=_F0_FLOOR, f0_ceil=_F0_CEILING)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)

    # Pitch: scale F0 multiplicatively (semitones → ratio of 2^(n/12)).
    # Voiced frames (f0 > 0) get scaled; unvoiced (f0 == 0) stays 0.
    if semitones != 0.0:
        ratio = 2.0 ** (semitones / 12.0)
        f0 = np.where(f0 > 0, f0 * ratio, f0)

    # Formant: warp the frequency axis of the spectral envelope. We don't
    # touch sample rate — we just re-interpolate sp's bins.
    if formant_shift != 1.0:
        n_bins = sp.shape[1]
        # For each target bin i, source bin is i / formant_shift.
        # formant_shift > 1.0 stretches features to higher frequencies.
        target_idx = np.arange(n_bins)
        src_idx = target_idx / float(formant_shift)
        s0 = np.floor(src_idx).astype(np.int64)
        s1 = np.clip(s0 + 1, 0, n_bins - 1)
        s0 = np.clip(s0, 0, n_bins - 1)
        frac = (src_idx - s0).astype(np.float64)

        # Vectorized linear interp across the frame axis.
        sp = sp[:, s0] * (1.0 - frac) + sp[:, s1] * frac

    # Synthesize at the same SR — WORLD never resamples on its own.
    y = pw.synthesize(f0, sp, ap, sr)

    # Defensive: if WORLD's framing drifted the length by >0, pad/trim to
    # match the input EXACTLY. Soundboard timing depends on length stability.
    if len(y) != in_len:
        if len(y) > in_len:
            y = y[:in_len]
        else:
            pad = np.zeros(in_len - len(y), dtype=y.dtype)
            y = np.concatenate([y, pad])

    return y.astype(np.float32)


def wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV blob → (mono float32 PCM, sample rate).

    Stereo inputs are downmixed to mono. We do NOT resample here — caller
    has already ensured the bake input matches the target SR.
    """
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    return data.astype(np.float32, copy=False), int(sr)


def pcm_to_wav_bytes(pcm: np.ndarray, sr: int, bit_depth: int = 16) -> bytes:
    """Encode mono float32 PCM → WAV bytes at the EXACT SR and bit depth.

    No dithering, no soft-clipping. We clip to [-1, 1] before quantisation;
    callers should normalize upstream if they want headroom.
    """
    subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}[bit_depth]
    pcm = np.clip(pcm, -1.0, 1.0).astype(np.float32 if bit_depth == 32 else np.float64)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype=subtype)
    return buf.getvalue()
