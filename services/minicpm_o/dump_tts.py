#!/usr/bin/env python3
"""Diagnostic: inspect the raw Token2Wav output wavs and bundle them into one file.

The bridge writes each TTS chunk to <output_dir>/tts_wav/wav_N.wav (24kHz). This script
prints per-wav stats (sample rate / frames / duration / peak) and concatenates them into
turn_dump.wav so you can LISTEN to exactly what Token2Wav produced -- bypassing our Opus
streaming and the browser entirely. That localizes audio problems:

  raw wavs sound clean & complete  -> Token2Wav is fine; bug is in our streaming/playback
  raw wavs are gibberish / silent  -> Token2Wav (the Q4 vocoder) itself

Run from the repo root with the minicpm_o venv (it already has soundfile + numpy):

  uv run --project services/minicpm_o python services/minicpm_o/dump_tts.py
  # optional: limit to an index range
  uv run --project services/minicpm_o python services/minicpm_o/dump_tts.py 70 98

Then download services/minicpm_o/_omni_out/turn_dump.wav and play it.
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("MINICPM_O_OUTPUT_DIR", os.path.join(HERE, "_omni_out"))
WAV_DIR = os.path.join(OUT_DIR, "tts_wav")


def _idx(path):
    digits = "".join(filter(str.isdigit, os.path.basename(path)))
    return int(digits) if digits else 0


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else None
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else None

    files = sorted(glob.glob(os.path.join(WAV_DIR, "wav_*.wav")), key=_idx)
    if lo is not None:
        files = [f for f in files if lo <= _idx(f) <= (hi if hi is not None else lo)]
    if not files:
        print(f"no wav_*.wav found in {WAV_DIR}")
        return

    chunks, sr = [], 24000
    total_silent = 0
    for f in files:
        a, sr = sf.read(f, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        peak = float(np.abs(a).max()) if len(a) else 0.0
        flag = "  <-- near-silent" if peak < 0.01 else ("  <-- clipping" if peak >= 0.999 else "")
        if peak < 0.01:
            total_silent += 1
        print("%-16s sr=%d frames=%6d dur=%4.2fs peak=%.3f%s"
              % (os.path.basename(f), sr, len(a), len(a) / sr if sr else 0, peak, flag))
        chunks.append(a)

    out = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    op = os.path.join(OUT_DIR, "turn_dump.wav")
    sf.write(op, out, sr)
    rates = {int(sf.info(f).samplerate) for f in files}
    print("\n%d wavs | sample rates seen: %s | near-silent wavs: %d"
          % (len(files), sorted(rates), total_silent))
    print("wrote %s : %d frames = %.1fs @ %dHz" % (op, len(out), len(out) / sr if sr else 0, sr))


if __name__ == "__main__":
    main()
