// ============================================================================
//  AUDIO FORMAT HELPERS — bake-path integrity
// ----------------------------------------------------------------------------
//  The whole soundboard depends on bakes coming out at exactly PP's expected
//  sample rate, channel count, and bit depth. Resampling silently anywhere
//  along the bake path would invalidate the experiment (timing/pitch drift).
//
//  Rules:
//    - Recording captures at PP_SAMPLE_RATE directly via getUserMedia
//      constraints + AudioContext({sampleRate}) so the browser hands us PCM
//      already at the right rate. We do NOT resample after the fact.
//    - WAVs are loaded via decodeAudioData (browser-native), which may decode
//      at the original file SR. If the file is at a different SR than PP, we
//      EXPLICITLY resample via OfflineAudioContext at the boundary and assert
//      the result is at PP_SAMPLE_RATE before storing/uploading.
//    - Stereo inputs are downmixed to mono by averaging channels.
//    - All baked WAVs are 16-bit PCM mono via createWavFile (audio.ts).
// ============================================================================

import {
  PP_SAMPLE_RATE,
  PP_CHANNELS,
  DURATION_DRIFT_TOLERANCE_MS,
  RECORDING_TRIM_RMS,
  RECORDING_TRIM_WINDOW_MS,
  RECORDING_TRIM_GUARD_MS,
} from "@/lib/soundboardConfig"
import { createWavFile } from "@shared/lib/audio"

export interface PcmBuffer {
  pcm: Float32Array
  sampleRate: number
  durationMs: number
}

// Decode any browser-supported audio blob to a Float32Array. Does NOT resample.
// Returns whatever SR the file was encoded at.
export async function decodeToPcm(blob: Blob): Promise<PcmBuffer> {
  const buf = await blob.arrayBuffer()
  // Use a one-shot AudioContext to decode; closed immediately to free resources.
  const ctx = new AudioContext()
  try {
    const audio = await ctx.decodeAudioData(buf.slice(0))
    // Downmix stereo → mono by averaging. PP only sees one channel.
    let mono: Float32Array
    if (audio.numberOfChannels === 1) {
      mono = new Float32Array(audio.getChannelData(0))
    } else {
      const len = audio.length
      mono = new Float32Array(len)
      for (let c = 0; c < audio.numberOfChannels; c++) {
        const ch = audio.getChannelData(c)
        for (let i = 0; i < len; i++) mono[i] += ch[i]
      }
      for (let i = 0; i < len; i++) mono[i] /= audio.numberOfChannels
    }
    return {
      pcm: mono,
      sampleRate: audio.sampleRate,
      durationMs: (mono.length / audio.sampleRate) * 1000,
    }
  } finally {
    await ctx.close()
  }
}

// Resample a PCM buffer to a target SR via OfflineAudioContext. This is the
// ONLY place in the bake path where resampling is allowed, and callers must
// be deliberate about invoking it. Pass-through if already at target SR.
export async function resampleTo(
  pcm: Float32Array,
  fromSr: number,
  toSr: number,
): Promise<Float32Array> {
  if (fromSr === toSr) return pcm
  // OfflineAudioContext expects a length in TARGET sample frames.
  const outLen = Math.round((pcm.length * toSr) / fromSr)
  const offline = new OfflineAudioContext(1, outLen, toSr)
  const buffer = offline.createBuffer(1, pcm.length, fromSr)
  buffer.copyToChannel(pcm, 0)
  const src = offline.createBufferSource()
  src.buffer = buffer
  src.connect(offline.destination)
  src.start(0)
  const rendered = await offline.startRendering()
  return new Float32Array(rendered.getChannelData(0))
}

// Trim silence from the START and END of a take only; interior silence (a
// scripted within-turn pause) is left untouched. Returns the original buffer
// when nothing crosses the threshold, so a silent take is never emptied.
export function trimEdgeSilence(
  pcm: Float32Array,
  sampleRate: number,
  {
    thresholdRms = RECORDING_TRIM_RMS,
    windowMs = RECORDING_TRIM_WINDOW_MS,
    guardMs = RECORDING_TRIM_GUARD_MS,
  } = {},
): { pcm: Float32Array; trimmedStartMs: number; trimmedEndMs: number } {
  const window = Math.max(1, Math.round((windowMs / 1000) * sampleRate))
  const loud: boolean[] = []
  for (let start = 0; start < pcm.length; start += window) {
    let sum = 0
    const end = Math.min(start + window, pcm.length)
    for (let i = start; i < end; i++) sum += pcm[i] * pcm[i]
    loud.push(Math.sqrt(sum / (end - start)) >= thresholdRms)
  }
  const first = loud.indexOf(true)
  const last = loud.lastIndexOf(true)
  if (first === -1) return { pcm, trimmedStartMs: 0, trimmedEndMs: 0 }

  const guard = Math.round((guardMs / 1000) * sampleRate)
  const from = Math.max(0, first * window - guard)
  const to = Math.min(pcm.length, (last + 1) * window + guard)
  return {
    pcm: pcm.slice(from, to),
    trimmedStartMs: (from / sampleRate) * 1000,
    trimmedEndMs: ((pcm.length - to) / sampleRate) * 1000,
  }
}

// Decode → downmix → resample to PP_SAMPLE_RATE. Use this for ANY external
// WAV (uploaded targets, imported clips) before storing it as a slot/target.
// The returned PcmBuffer is guaranteed to be mono at PP_SAMPLE_RATE.
export async function decodeAndConformToPp(blob: Blob): Promise<PcmBuffer> {
  const decoded = await decodeToPcm(blob)
  const pcm = await resampleTo(decoded.pcm, decoded.sampleRate, PP_SAMPLE_RATE)
  return {
    pcm,
    sampleRate: PP_SAMPLE_RATE,
    durationMs: (pcm.length / PP_SAMPLE_RATE) * 1000,
  }
}

// Encode mono Float32 PCM → 16-bit mono WAV Blob at the given SR.
// Asserts the SR matches PP's expected rate (catch bugs in callers early).
export function encodeWavAtPpRate(pcm: Float32Array, sampleRate: number): Blob {
  if (sampleRate !== PP_SAMPLE_RATE) {
    throw new Error(
      `encodeWavAtPpRate: refusing to encode at ${sampleRate} Hz, ` +
        `PP expects ${PP_SAMPLE_RATE} Hz. Resample first via resampleTo().`,
    )
  }
  if (PP_CHANNELS !== 1) {
    // createWavFile only writes mono. If we ever need stereo, change BOTH.
    throw new Error(`encodeWavAtPpRate: only mono is supported (PP_CHANNELS=${PP_CHANNELS}).`)
  }
  return createWavFile(pcm, sampleRate)
}

// Check that two clips have the same duration within tolerance. Used after
// every bake (VC or pitch/formant) to detect silent re-timing. Returns the
// drift in ms (signed) so callers can log it.
export interface DurationCheck {
  inMs: number
  outMs: number
  driftMs: number
  ok: boolean
}

export function checkDuration(inMs: number, outMs: number): DurationCheck {
  const driftMs = +(outMs - inMs).toFixed(2)
  return {
    inMs: +inMs.toFixed(2),
    outMs: +outMs.toFixed(2),
    driftMs,
    ok: Math.abs(driftMs) <= DURATION_DRIFT_TOLERANCE_MS,
  }
}

// Convenience: blob → ms duration via decodeAudioData (no resampling).
export async function blobDurationMs(blob: Blob): Promise<number> {
  const { durationMs } = await decodeToPcm(blob)
  return durationMs
}
