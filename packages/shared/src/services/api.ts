import { createWavFile } from "@shared/lib/audio"

// Both apps are served same-origin by app-api, so API calls are relative.
const API_BASE = ""

export interface TranscriptionWord {
  start: number
  end: number
  word: string
}

export interface TranscriptionSegment {
  start: number
  end: number
  text: string
  words?: TranscriptionWord[]
}

export interface TranscriptionResult {
  text: string
  segments: TranscriptionSegment[]
}

export async function transcribeRecording(
  chunks: Blob[]
): Promise<TranscriptionResult> {
  const blob = new Blob(chunks, { type: "audio/webm" })
  const arrayBuffer = await blob.arrayBuffer()
  const ctx = new AudioContext()
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer)
  ctx.close()

  const wavBlob = createWavFile(audioBuffer.getChannelData(0), audioBuffer.sampleRate)
  const formData = new FormData()
  formData.append("audio", wavBlob, "recording.wav")

  const resp = await fetch(`${API_BASE}/api/transcribe`, { method: "POST", body: formData })
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`)
  return resp.json()
}

// Transcribe a WAV blob directly (no webm decode/re-encode round-trip).
// Safer for long clips than transcribeRecording, which rebuilds the audio.
export async function transcribeWavBlob(
  wav: Blob
): Promise<TranscriptionResult> {
  const formData = new FormData()
  formData.append("audio", wav, "audio.wav")
  const resp = await fetch(`${API_BASE}/api/transcribe`, { method: "POST", body: formData })
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`)
  return resp.json()
}

export async function convertVoice(sourceFile: File, targetFile: File): Promise<Blob> {
  const fd = new FormData()
  fd.append("source_audio", sourceFile)
  fd.append("target_audio", targetFile)
  const resp = await fetch(`${API_BASE}/api/voice-conversion`, { method: "POST", body: fd })
  if (!resp.ok) throw new Error(await resp.text())
  return resp.blob()
}

export interface XvcFileBake {
  wav: Blob
  engine: "xvc"
  inMs: number
  outMs: number
  driftMs: number
  inputSamples: number
  outputSamples: number
  sampleRate: number
  inferenceWindows: number
}

// Offline X-VC conversion for pre-baked soundboard clips. The X-VC service
// keeps the complete source timeline and returns a sample-exact 24 kHz WAV, so
// the browser must store this blob directly rather than resampling it again.
export async function xvcFileBake(
  source: Blob,
  target: Blob,
  outputSr: number,
): Promise<XvcFileBake> {
  const fd = new FormData()
  fd.append("source_audio", source, "source.wav")
  fd.append("target_audio", target, "target.wav")
  fd.append("output_sr", String(outputSr))
  const resp = await fetch(`${API_BASE}/api/xvc/file-conversion`, {
    method: "POST",
    body: fd,
  })
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`)

  const engine = resp.headers.get("X-VC-Engine")
  const inMs = Number(resp.headers.get("X-Input-Duration-Ms") || "0")
  const outMs = Number(resp.headers.get("X-Output-Duration-Ms") || "0")
  const sampleRate = Number(resp.headers.get("X-Sample-Rate") || "0")
  const inputSamples = Number(resp.headers.get("X-Input-Samples") || "0")
  const outputSamples = Number(resp.headers.get("X-Output-Samples") || "0")
  const inferenceWindows = Number(
    resp.headers.get("X-XVC-Inference-Windows") || "0",
  )
  if (engine !== "xvc") {
    throw new Error(`X-VC bake returned unexpected engine ${engine || "unknown"}.`)
  }
  if (![inMs, outMs, sampleRate, inputSamples, outputSamples].every(Number.isFinite)) {
    throw new Error("X-VC bake returned invalid audio audit headers.")
  }
  return {
    wav: await resp.blob(),
    engine,
    inMs,
    outMs,
    driftMs: +(outMs - inMs).toFixed(2),
    inputSamples,
    outputSamples,
    sampleRate,
    inferenceWindows,
  }
}

// Offline Seed-VC conversion (the Convert tab's engine) for soundboard bakes.
// NOT duration-preserving: the endpoint VAD-trims the source and Seed-VC may
// re-time it, and the output sample rate is the model's own — callers must
// conform the result to PP's rate themselves and expect duration drift.
export async function seedVcFileConversion(source: Blob, target: Blob): Promise<Blob> {
  const fd = new FormData()
  fd.append("source_audio", source, "source.wav")
  fd.append("target_audio", target, "target.wav")
  const resp = await fetch(`${API_BASE}/api/voice-conversion`, { method: "POST", body: fd })
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`)
  return resp.blob()
}

export async function compareMetrics(sourceFile: File, targetFile: File): Promise<Blob> {
  const fd = new FormData()
  fd.append("source_audio", sourceFile)
  fd.append("target_audio", targetFile)
  const resp = await fetch(`${API_BASE}/api/metrics-comparison`, { method: "POST", body: fd })
  if (!resp.ok) throw new Error(await resp.text())
  return resp.blob()
}

export interface ResponseMetrics {
  speech_rate: number | null
  sentiment: string | null
  mean_pitch: number | null
  std_pitch: number | null
  transcript?: string | null
  duration?: number | null
}

export interface AestheticMetrics {
  production_quality: number | null
  content_usefulness: number | null
  content_enjoyment: number | null
  production_complexity: number | null
}

export interface MetricsResult {
  response_a: ResponseMetrics
  response_b: ResponseMetrics
  comparison: { semantic_similarity: number | null }
  aesthetics: { response_a: AestheticMetrics; response_b: AestheticMetrics }
}

// JSON variant — returns the raw metrics so the UI renders them as HTML/CSS
// (radar chart + cards) instead of a server-rendered PNG.
export async function compareMetricsData(source: Blob, target: Blob): Promise<MetricsResult> {
  const fd = new FormData()
  // Explicit .wav filenames so the backend's extension check passes for raw Blobs.
  fd.append("source_audio", source, "source.wav")
  fd.append("target_audio", target, "target.wav")
  const resp = await fetch(`${API_BASE}/api/metrics-comparison?output=json`, { method: "POST", body: fd })
  if (!resp.ok) throw new Error(await resp.text())
  return resp.json()
}

// X-VC objective quality profile: WER, WavLM SIM, and UTMOS. Optional
// segmentation is for diagnostics; study runs score whole stable VC regions.
export interface VcQualitySegment {
  start: number
  end: number
  sim?: number | null
  wer?: number | null
  utmos?: number | null
  vc?: string
  ref?: string
}

export interface VcQualityAnomaly {
  start: number
  end: number
  metric: string
  score: number
  z: number
}

export interface VcQualityResult {
  converted_path: string
  target_path: string
  source_path: string | null
  wer: number | null
  sim: number | null
  utmos: number | null
  vc_transcript?: string
  ref_transcript?: string | null
  ref_kind?: string | null
  wer_reference_note?: string | null
  segment_mode?: string
  segments?: VcQualitySegment[]
  anomalies?: VcQualityAnomaly[]
}

// Names match the backend's _SKIPPABLE_METRICS keys.
export type VcQualityMetric = "intelligibility" | "speaker_similarity" | "utmos"

// Offline pitch/formant shift for soundboard bakes. Preserves duration.
// `targetSr` MUST equal PP's expected SR (PP_SAMPLE_RATE); the server refuses
// to bake at any other rate. Returns the WAV blob + the bake's measured input
// vs output duration (from response headers) so the caller can flag drift.
export interface PitchFormantBake {
  wav: Blob
  inMs: number
  outMs: number
  driftMs: number
  sampleRate: number
}

export async function pitchFormantBake(
  source: Blob,
  semitones: number,
  formantShift: number,
  targetSr: number,
): Promise<PitchFormantBake> {
  const fd = new FormData()
  fd.append("source_audio", source, "source.wav")
  fd.append("semitones", String(semitones))
  fd.append("formant_shift", String(formantShift))
  fd.append("target_sr", String(targetSr))
  const resp = await fetch(`${API_BASE}/api/pitch-formant`, { method: "POST", body: fd })
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`)
  const inMs = Number(resp.headers.get("X-Input-Duration-Ms") || "0")
  const outMs = Number(resp.headers.get("X-Output-Duration-Ms") || "0")
  const sampleRate = Number(resp.headers.get("X-Sample-Rate") || "0")
  return {
    wav: await resp.blob(),
    inMs,
    outMs,
    driftMs: +(outMs - inMs).toFixed(2),
    sampleRate,
  }
}

// EBU R128 / LUFS loudness normalization for soundboard bakes (P3). Preserves
// SR + duration; `targetSr` must equal PP's expected rate (the server refuses
// to resample). Returns the normalized WAV + measured in/out loudness so the
// caller can audit the level match.
export interface LoudnessNormalizeResult {
  wav: Blob
  sampleRate: number
  inLufs: number | null    // null when the clip was too short/silent to meter
  outLufs: number | null
  peak: number
  durationMs: number
}

export async function loudnessNormalize(
  source: Blob,
  targetLufs: number,
  targetSr: number,
): Promise<LoudnessNormalizeResult> {
  const fd = new FormData()
  fd.append("audio", source, "audio.wav")
  fd.append("target_lufs", String(targetLufs))
  fd.append("target_sr", String(targetSr))
  const resp = await fetch(`${API_BASE}/api/loudness-normalize`, { method: "POST", body: fd })
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`)
  const parseNum = (h: string | null): number | null => {
    if (h == null || h === "nan") return null
    const n = Number(h)
    return Number.isFinite(n) ? n : null
  }
  const sampleRate = Number(resp.headers.get("X-Sample-Rate") || "0")
  const inLufs = parseNum(resp.headers.get("X-Input-Lufs"))
  const outLufs = parseNum(resp.headers.get("X-Output-Lufs"))
  const peak = Number(resp.headers.get("X-Peak") || "0")
  const durationMs = Number(resp.headers.get("X-Duration-Ms") || "0")
  return { wav: await resp.blob(), sampleRate, inLufs, outLufs, peak, durationMs }
}

export async function vcQuality(
  source: Blob, target: Blob, converted: Blob,
  opts?: {
    sourceTranscript?: string
    segmentMode?: "fixed" | "word" | "vad"
    // Fixed-window resolution for explicit diagnostic runs.
    segmentWin?: number
    segmentHop?: number
    skipMetrics?: VcQualityMetric[]
  },
): Promise<VcQualityResult> {
  const fd = new FormData()
  fd.append("source_audio", source, "source.wav")
  fd.append("target_audio", target, "target.wav")
  fd.append("converted_audio", converted, "converted.wav")
  if (opts?.sourceTranscript) fd.append("source_transcript", opts.sourceTranscript)
  if (opts?.segmentMode) fd.append("segment_mode", opts.segmentMode)
  if (opts?.segmentWin != null) fd.append("segment_win", String(opts.segmentWin))
  if (opts?.segmentHop != null) fd.append("segment_hop", String(opts.segmentHop))
  if (opts?.skipMetrics?.length) fd.append("skip_metrics", opts.skipMetrics.join(","))
  const resp = await fetch(`${API_BASE}/api/vc-quality`, { method: "POST", body: fd })
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`)
  return resp.json()
}
