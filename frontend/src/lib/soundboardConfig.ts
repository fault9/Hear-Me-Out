// ============================================================================
//  SOUNDBOARD CONFIG — single source of truth
// ----------------------------------------------------------------------------
//  EVERY audio constant the soundboard touches lives here. Do not hardcode
//  sample rates / channels / bit depths anywhere else. If PP's expected input
//  format changes, you change it HERE and only here.
//
//  CRITICAL — these MUST match what PersonaPlex expects on its WebSocket.
//  The live-mic path (useRecorder.ts) currently encodes Opus at 24000 Hz
//  mono with 80-sample frames, 1 frame per Ogg page. If you change PP_SAMPLE_RATE
//  here, make sure the live-mic encoder config in useRecorder.ts matches.
//
//  Verify end-to-end after any change:
//    1. Bake a known clip in VC mode (e.g. a 3.0s sine).
//    2. Run the bake again — output should be the same bytes (deterministic).
//    3. Compare X-Input-Duration-Ms vs X-Output-Duration-Ms response headers
//       — drift > DURATION_DRIFT_TOLERANCE_MS is a bug.
//    4. Play the slot through the conversation soundboard. Confirm PP responds
//       as if you'd spoken the same words live with VC off.
// ============================================================================

// PP's input sample rate. The live Opus encoder runs at this rate too; bakes
// MUST be delivered at exactly this SR or PP gets timing/pitch distortion.
export const PP_SAMPLE_RATE = 24000

// PP is mono on input. Stereo bakes are downmixed.
export const PP_CHANNELS = 1

// We always store baked WAVs at 16-bit PCM. Frontend's createWavFile already
// does 16-bit; keep them consistent. Anything else would require touching
// createWavFile too.
export const PP_BIT_DEPTH = 16

// Opus encoder config — must match useRecorder.ts so soundboard playback
// produces the same Opus framing as live mic input. PP very likely depends
// on the 1-frame-per-page granularity for low-latency processing.
export const OPUS_ENCODER_CONFIG = {
  encoderPath:
    "https://cdn.jsdelivr.net/npm/opus-recorder@8.0.5/dist/encoderWorker.min.js",
  streamPages: true,
  encoderApplication: 2049,        // OPUS_APPLICATION_VOIP
  encoderFrameSize: 80,            // ms per frame (NOT samples)
  encoderSampleRate: PP_SAMPLE_RATE,
  maxFramesPerPage: 1,             // one Opus frame per Ogg page — PP expects this
  numberOfChannels: PP_CHANNELS,
} as const

// Defaults for the pitch/formant bake mode. Researchers override per slot.
// Semitones is additive (0 = no change); formant_shift is multiplicative
// (1.0 = no change, > 1.0 raises formants = more "feminine" perception).
export const PITCH_FORMANT_DEFAULTS = {
  semitones: 0.0,
  formantShift: 1.0,
} as const

// Two common gender-shift presets. These are STARTING POINTS — perception
// varies; tune per voice. Used only as quick-pick buttons in Configure.
export const PITCH_FORMANT_PRESETS = {
  more_masculine: { semitones: -4, formantShift: 0.85 },
  more_feminine:  { semitones: +4, formantShift: 1.15 },
} as const

// Any baked clip whose duration drifts more than this from the raw input
// is flagged in the Configure UI. WORLD synthesis is duration-preserving;
// the server pads/trims to exact input length. VC pipelines can re-time
// audio, which would invalidate timing-sensitive experiments — so we want
// to know about drift.
export const DURATION_DRIFT_TOLERANCE_MS = 5

// Loudness normalization (EBU R128 / LUFS) target for soundboard bakes (P3).
// -23 LUFS is the EBU R128 broadcast reference; the exact value matters less
// than every condition landing at the SAME level so playback loudness isn't a
// confound. Applied as a bake step (see useSoundboard.bakeSlot) when enabled.
export const LOUDNESS_TARGET_LUFS = -23
// Default the toggle ON — level-matched conditions are the safer research
// default; the operator can turn it off per bake if needed.
export const LOUDNESS_NORMALIZE_DEFAULT = true

// Recording limits in Configure. Slot recordings are short scripted utterances;
// 30s is plenty and keeps IndexedDB blobs small.
export const MAX_RECORDING_SECONDS = 30
export const MIN_RECORDING_SECONDS = 0.3

// Leading/trailing silence is trimmed from a fresh take: audit latencies are
// measured from the clip's end, so trailing room tone would place the measured
// offset after the speech actually stopped. Only the edges are trimmed —
// within-turn pauses are part of the utterance and must survive.
export const RECORDING_TRIM_RMS = 0.01
export const RECORDING_TRIM_WINDOW_MS = 10
export const RECORDING_TRIM_GUARD_MS = 40

// IndexedDB names — version bumps trigger migrations (none defined yet).
export const SOUNDBOARD_DB_NAME = "hmo_soundboard"
export const SOUNDBOARD_DB_VERSION = 1

// Built-in default target voice — the same one the rest of the app uses for
// PP's voice prompt. The Configure tab seeds the target library with this.
export const DEFAULT_TARGET = {
  id: "natf2",
  label: "NATF2 (default female)",
  // Targets uploaded by the user are stored as Blobs in IndexedDB. The
  // built-in target is referenced by name and served by the VC service.
  builtin: true,
} as const
