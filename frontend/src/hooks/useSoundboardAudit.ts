// ============================================================================
//  SOUNDBOARD AUDIT RUNNER — automated matched-input presentations
// ----------------------------------------------------------------------------
//  Implements the prespecified soundboard audit (method §Technical Pilot and
//  Controlled System Audit): matched natural/converted recordings, N
//  repetitions per item, randomized order, and a FRESH conversation state for
//  every presentation. The manifest (items, bake hashes, engines, order,
//  seed, prompt, timing parameters) is generated and hash-frozen BEFORE the
//  run starts; results bundle into one zip (manifest + per-presentation PP
//  audio, sent audio, transcript, timing log).
//
//  The runner drives the WebSocket directly (connect → handshake → wait for
//  PP's greeting to settle → play slot → wait for PP's response run to end →
//  collect → disconnect). It deliberately does NOT go through
//  useConversation: no mic recorder, no post-conversation Whisper calls —
//  each presentation's evidence is the capture buffer (sent bytes), PP's
//  Opus packets, and the live PP transcript.
//
//  Production-engine discipline: VC slots baked with a non-X-VC engine are
//  flagged; the run refuses them unless the operator explicitly marks the
//  run exploratory (recorded in the manifest).
// ============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  getPersonaplexAuditWsURL,
  PERSONAPLEX_AUDIT_VOICE_PROMPT,
} from "@/lib/config"
import {
  DURATION_DRIFT_TOLERANCE_MS,
  OPUS_ENCODER_CONFIG,
  PP_SAMPLE_RATE,
} from "@/lib/soundboardConfig"
import { checkpointSoundboardAudit, uploadSoundboardAudit } from "@shared/services/api"
import { makeZip } from "@/lib/soundboardZip"
import { assembleSentWav, assembleSentTimelineWav, getCapturedClips, resetCapture } from "@/lib/soundboardCapture"
import {
  analyzeAuditInput,
  type AuditInputTiming,
} from "@/lib/soundboardAuditTiming"
import type { Slot, Target } from "@/lib/soundboardDb"
import type {
  AudioDeliveryAuditResult,
  useWebSocket,
} from "@shared/hooks/useWebSocket"
import type { useSoundboardPlayback } from "@/hooks/useSoundboardPlayback"

type WsState = ReturnType<typeof useWebSocket>
type PlaybackState = ReturnType<typeof useSoundboardPlayback>

// opus-recorder is loaded globally via <script> (see useSoundboardPlayback).
interface RecorderInstance {
  start(): Promise<void>
  stop(): Promise<void> | void
  ondataavailable: ((buf: ArrayBuffer) => void) | null
}
type RecorderConstructor = new (opts: Record<string, unknown>) => RecorderInstance

// PersonaPlex is full-duplex: its generation loop is CLOCKED by incoming
// audio frames — with no input stream it produces nothing (no greeting, no
// response). A live conversation gets this clock from the mic; the audit
// runner has no mic, so it must stream continuous SILENCE, gated off while a
// clip plays (the same isMicMuted gate the live mic uses, so the clip's
// Opus stream is the only input during playback — one coherent stream).
async function startSilenceFeed(ws: WsState): Promise<{ stop: () => Promise<void> }> {
  const RecorderCtor = (
    window as unknown as { Recorder?: RecorderConstructor }
  ).Recorder
  if (!RecorderCtor) {
    throw new Error("Opus encoder not available (window.Recorder missing)")
  }
  const ctx = new AudioContext({ sampleRate: PP_SAMPLE_RATE })
  if (ctx.sampleRate !== PP_SAMPLE_RATE) {
    await ctx.close()
    throw new Error(`AudioContext refused ${PP_SAMPLE_RATE} Hz for the silence feed`)
  }
  const source = ctx.createConstantSource()
  source.offset.value = 0
  // Keep the graph pulled: a zero-gain tap to the destination guarantees the
  // encoder's ScriptProcessor keeps firing in all browsers.
  const sink = ctx.createGain()
  sink.gain.value = 0
  source.connect(sink)
  sink.connect(ctx.destination)
  const recorder = new RecorderCtor({ ...OPUS_ENCODER_CONFIG, sourceNode: source })
  recorder.ondataavailable = (buf: ArrayBuffer) => {
    if (!ws.isMicMuted()) ws.sendAudio(buf)
  }
  await recorder.start()
  source.start()
  return {
    stop: async () => {
      try { source.stop() } catch { /* already stopped */ }
      try { await recorder.stop() } catch { /* noop */ }
      try { await ctx.close() } catch { /* noop */ }
    },
  }
}

// "matched": §3.3 audit — shuffled single-turn presentations, matched
// natural/converted pairs. "script": a full ordered multi-turn conversation
// (all slots, soundboard order) replayed as-is, N times in fresh
// conversations, with a fixed quiet gap between turns.
export type AuditMode = "matched" | "script"

export interface AuditConfig {
  mode: AuditMode
  reps: number                 // matched: reps per slot · script: whole-script replays
  seed: number
  prompt: string
  greetingSettleMs: number     // PP silent this long after its greeting → play
  greetingCapMs: number        // play anyway after this long post-handshake
  responseTimeoutMs: number    // no PP speech after clip end → "no_response"
  responseLingerMs: number     // PP silent this long after response → done
  interTurnGapMs: number       // script: quiet gap after PP replies → next line
  // Script mode: build one script per condition TAG and cycle through the
  // tags A,B,…,A,B,… so condition is not confounded with time / server
  // drift. reps = replays PER condition.
  interleaveByCondition: boolean
  handshakeTimeoutMs: number
  cooldownMs: number           // between conversations (PP teardown/reset)
  allowNonProductionEngines: boolean
}

export const DEFAULT_AUDIT_CONFIG: AuditConfig = {
  mode: "script",
  reps: 3,
  seed: 0,
  prompt: "You enjoy having a good conversation.",
  greetingSettleMs: 900,
  greetingCapMs: 10_000,
  responseTimeoutMs: 20_000,
  responseLingerMs: 1_500,
  interTurnGapMs: 500,
  interleaveByCondition: false,
  handshakeTimeoutMs: 45_000,
  // PP (moshi) is single-connection and needs time to reset between
  // conversations; too short and the next connection never handshakes. Human
  // stop→start gaps in normal use are several seconds — match that.
  cooldownMs: 8_000,
  allowNonProductionEngines: false,
}

// Interrupt presentations fire this far into PP's ongoing speech run, so
// every corrective interruption lands at a comparable point of a PP turn.
export const INTERRUPT_FIRE_DELAY_MS = 800
// Includes the browser's encoder-start and local relay-control round trip.
// The exact realized offset is retained and used for the final validity check.
export const INTERRUPT_FIRE_TOLERANCE_MS = 300
// A decoded PP packet counts as speech (not silence) at/above this RMS; PP's
// idle frames sit near zero (observed ~0.01 mean silence vs ~0.25 peak speech).
export const PP_ENERGY_THRESHOLD = 0.02
// PP is "speaking now" if an energetic packet arrived within this window.
export const PP_SPEAKING_GAP_MS = 350
// Minimum warm time after handshake before playing, so PP's brief greeting
// isn't clipped and the encoder/graph is settled.
export const MIN_PRE_CLIP_MS = 1200
// One energetic 20-ms assistant packet overlapping annotated source speech is
// the minimum audit overlap event. Participant-study overlap uses its separate
// manually verified 200-ms definition.
export const AUDIT_OVERLAP_EVENT_MIN_MS = 20

export type PresentationMode = "after_silence" | "during_pp_speech"

export interface AuditPresentation {
  index: number                 // 1-based, in frozen (shuffled) order
  slot_id: string
  label: string
  condition: string
  manipulation: string
  route: "natural" | "converted" | "pitch_formant"
  source_speaker: string | null
  engine: string | null
  target_id: string | null
  target_label: string | null
  target_sha256: string | null
  clip_sha256: string
  // Hash of the slot's RAW source take. Matched natural/converted pairs share
  // one source WAV, so raw_sha256 equality across a pair IS the proof that
  // linguistic content and source timing are held constant.
  raw_sha256: string | null
  clip_duration_ms: number
  input_timing_source_sha256: string
  input_timing: AuditInputTiming
  clip_audio: AuditClipAudioMetadata
  // after_silence: wait for PP to finish speaking, then play (default).
  // during_pp_speech: fire the clip WHILE PP is speaking (corrective-
  // interruption items) and measure yielding.
  presentation_mode: PresentationMode
}

// One line of a scripted conversation (script mode). The script is the
// eligible slots in soundboard order; reps replay the whole script.
export interface AuditScriptTurn {
  turn: number                  // 1-based position in the script
  slot_id: string
  label: string
  condition: string
  manipulation: string
  route: "natural" | "converted" | "pitch_formant"
  source_speaker: string | null
  engine: string | null
  target_id: string | null
  target_label: string | null
  target_sha256: string | null
  clip_sha256: string
  raw_sha256: string | null
  clip_duration_ms: number
  input_timing_source_sha256: string
  input_timing: AuditInputTiming
  clip_audio: AuditClipAudioMetadata
}

export interface AuditClipAudioMetadata {
  sample_rate_hz: number
  normalized: boolean
  target_lufs: number | null
  measured_lufs: number | null
  duration_drift_ms: number
}

export interface AuditManifest {
  schema: "hmo.soundboard-audit-manifest.v3"
  mode: AuditMode
  created_at: string
  seed: number
  reps: number
  prompt: string
  assistant: {
    engine: "personaplex"
    voice_prompt: string
    text_prompt: string
  }
  audio_format: {
    personaplex_sample_rate_hz: number
    opus_encoder: typeof OPUS_ENCODER_CONFIG
  }
  inter_turn_gap_ms?: number    // script mode
  // Every parameter that DEFINES a measurement is frozen here — including the
  // energy-detection constants (what counts as PP "speaking") — so a run's
  // artifacts fully specify how their own numbers were produced.
  timing: Pick<AuditConfig, "greetingSettleMs" | "greetingCapMs"
    | "responseTimeoutMs" | "responseLingerMs" | "cooldownMs">
    & {
      interruptFireDelayMs: number
      interruptFireToleranceMs: number
      interTurnGapMs?: number
      ppEnergyThresholdRms: number
      ppSpeakingGapMs: number
      minPreClipMs: number
      overlapEventMinimumMs: number
    }
  production_engine_only: boolean
  engine_warnings: string[]
  // Matched mode only: raw_sha256 -> manipulations present; single-condition
  // items are warned about.
  pairing?: {
    raw_sha256: string
    labels: string[]
    manipulations: string[]
    source_speakers: string[]
  }[]
  pairing_warnings?: string[]
  presentations?: AuditPresentation[]   // matched mode
  script?: AuditScriptTurn[]            // script mode, single script
  // Script mode, interleaved: one script per condition tag + the alternating
  // replay plan (condition not confounded with time).
  interleaved?: boolean
  scripts?: { condition: string; turns: AuditScriptTurn[] }[]
  replay_plan?: { rep: number; condition: string; cycle: number }[]
  manifest_sha256?: string      // hash of the manifest WITHOUT this field
}

// Script mode records: one AuditSessionRecord per replay, each holding a
// per-turn AuditTurnRecord. The whole-conversation PP energy events let the
// post-processor slice per-turn overlap, onset, and response windows.
export interface AuditTurnRecord {
  turn: number
  slot_id: string
  label: string
  status: "ok" | "no_response" | "clip_error"
  t_play_start_ms: number | null
  t_play_end_ms: number | null
  t_pp_response_start_ms: number | null
  t_pp_response_end_ms: number | null
  // Live-runner diagnostic. The scheduled playback timeline and input-sample
  // annotations produce the authoritative post-processed response gap.
  response_latency_ms: number | null
  // Coarse live-runner flag retained for diagnostics. Packet-scheduled timing
  // in post-processing is authoritative for overlap and premature onset.
  pp_spoke_during_clip: boolean
  sent_clip_sha256: string | null
  delivery_audit: AudioDeliveryAuditResult | null
  notes: string[]
}

export interface AuditSessionRecord {
  rep: number
  condition?: string | null     // interleaved script runs: this replay's script
  status: "ok" | "handshake_timeout" | "aborted" | "error"
  error?: string
  t_connect_ms: number
  t_handshake_ms: number | null
  greeted: boolean
  turns: AuditTurnRecord[]
  pp_speech_events: { type: string; timestampMs: number; rms?: number }[]
  pp_transcript: { text: string; speaker: string }[]
  _ppWav?: Blob | null
  _sentWav?: Blob | null
  _playbackTimeline?: unknown
}

export interface AuditRunRecord {
  index: number
  slot_id: string
  label: string
  presentation_mode: PresentationMode
  status: "ok" | "no_response" | "handshake_timeout" | "clip_error" | "aborted" | "error"
  error?: string
  // performance.now() marks (same clock as pp_speech_events).
  t_connect_ms: number
  t_handshake_ms: number | null
  t_play_start_ms: number | null
  t_play_end_ms: number | null
  t_pp_response_start_ms: number | null
  t_pp_response_end_ms: number | null
  // Live-runner clip-end-to-energy diagnostic. Post-processing derives the
  // authoritative signed onset offset from scheduled playback packets.
  response_latency_ms: number | null
  // Live-runner interruption diagnostics retained for troubleshooting. The
  // packet timeline supplies authoritative fire offset and stop latency.
  fire_offset_into_pp_speech_ms: number | null
  pp_yield_latency_ms: number | null
  notes: string[]
  // Legacy/live diagnostics on performance.now(). New analyses use the
  // separately archived playback_timeline.json packet schedule.
  pp_speech_events: { type: string; timestampMs: number }[]
  pp_transcript: { text: string; speaker: string }[]
  sent_clip_sha256: string | null
  delivery_audit: AudioDeliveryAuditResult | null
  // Internal carriers between runPresentation and the zip bundler; stripped
  // from the record before it lands in run_log.json.
  _ppWav?: Blob | null
  _sentWav?: Blob | null
  _playbackTimeline?: unknown
}

export interface AuditProgress {
  running: boolean
  currentIndex: number          // 1-based presentation/rep being run (0 = none)
  total: number
  phase: string                 // human-readable current step
  records: (AuditRunRecord | AuditSessionRecord)[]
}

// Deterministic PRNG (mulberry32) + Fisher-Yates so the presentation order is
// reproducible from the seed recorded in the manifest.
function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function shuffled<T>(items: T[], seed: number): T[] {
  const rng = mulberry32(seed)
  const out = items.slice()
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

async function sha256Hex(blob: Blob): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer())
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0")).join("")
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

const routeForSlot = (slot: Slot): AuditPresentation["route"] => {
  if (slot.manipulation === "vc") return "converted"
  if (slot.manipulation === "unconverted") return "natural"
  return "pitch_formant"
}

const clipAudioForSlot = (slot: Slot): AuditClipAudioMetadata => ({
  sample_rate_hz: slot.sampleRate,
  normalized: slot.normalized === true,
  target_lufs: slot.targetLufs ?? null,
  measured_lufs: slot.measuredLufs ?? null,
  duration_drift_ms: slot.driftMs,
})

export function useSoundboardAudit(opts: {
  ws: WsState
  playback: PlaybackState
  slots: Slot[]
  targets: Target[]
  onBeforeRun?: () => void      // e.g. panel turns its autoplay off
}) {
  const { ws, playback, slots, targets, onBeforeRun } = opts
  const [config, setConfig] = useState<AuditConfig>(DEFAULT_AUDIT_CONFIG)
  // Slot ids presented in interrupt mode (corrective-interruption items).
  const [interruptSlotIds, setInterruptSlotIds] = useState<Set<string>>(new Set())
  const toggleInterruptSlot = useCallback((slotId: string) => {
    setInterruptSlotIds((prev) => {
      const next = new Set(prev)
      if (next.has(slotId)) next.delete(slotId)
      else next.add(slotId)
      return next
    })
  }, [])
  const [manifest, setManifest] = useState<AuditManifest | null>(null)
  const [progress, setProgress] = useState<AuditProgress>({
    running: false, currentIndex: 0, total: 0, phase: "idle", records: [],
  })
  const [error, setError] = useState<string | null>(null)
  const [resultsZip, setResultsZip] = useState<Blob | null>(null)
  // Server persistence: runs are auto-uploaded to STUDY_DATA_ROOT/audit/ so
  // audit artifacts live on the study data volume; the local download stays
  // available as a fallback.
  const [uploadState, setUploadState] = useState<
    { status: "idle" | "uploading" | "done" | "failed"; detail?: string }>({ status: "idle" })

  // ---- live mirrors (poll-friendly refs over React state) ----------------
  const handshakeRef = useRef(false)
  handshakeRef.current = ws.handshakeReceived
  const playingRef = useRef<string | null>(null)
  playingRef.current = playback.playingSlotId
  const transcriptsRef = useRef(ws.transcripts)
  transcriptsRef.current = ws.transcripts

  // PP speech-run tracking for the whole audit run, plus the COMPLETE event
  // list per presentation (overlap during the clip, premature onsets in
  // ambiguous pauses, and yielding are all derived from it downstream).
  // Speech-vs-silence is detected by ENERGY, not packet arrival: PP streams
  // continuous frames (mostly silence) while a conversation is open, so the
  // arrival-based pp-speech events would report PP "speaking" forever. We key
  // off per-packet RMS instead. Events are still logged for reference.
  const ppEventsRef = useRef<{ type: string; timestampMs: number; rms?: number }[]>([])
  const lastEnergeticMsRef = useRef(0)   // perf.now() of last rms>threshold packet
  const energeticRunStartMsRef = useRef(0)
  const sawEnergyRef = useRef(false)
  const speakingNow = useCallback(
    () => lastEnergeticMsRef.current > 0
      && performance.now() - lastEnergeticMsRef.current < PP_SPEAKING_GAP_MS,
    [])
  const { registerPpSpeechListener, registerAssistantAudioListener } = ws
  useEffect(() => {
    const offEvents = registerPpSpeechListener((e) => {
      ppEventsRef.current.push({ type: e.type, timestampMs: e.timestampMs })
    })
    const offAudio = registerAssistantAudioListener((rms, startMs, endMs) => {
      if (rms >= PP_ENERGY_THRESHOLD) {
        if (lastEnergeticMsRef.current === 0
            || startMs - lastEnergeticMsRef.current >= PP_SPEAKING_GAP_MS) {
          energeticRunStartMsRef.current = startMs
        }
        lastEnergeticMsRef.current = endMs
        sawEnergyRef.current = true
        ppEventsRef.current.push({
          type: "pp_energy",
          timestampMs: endMs,
          rms: +rms.toFixed(4),
        })
      }
    })
    return () => { offEvents(); offAudio() }
  }, [registerPpSpeechListener, registerAssistantAudioListener])

  const abortRef = useRef(false)

  const waitFor = useCallback(async (predicate: () => boolean, timeoutMs: number) => {
    const deadline = performance.now() + timeoutMs
    while (performance.now() < deadline) {
      if (abortRef.current) throw new Error("aborted")
      if (predicate()) return true
      await sleep(100)
    }
    return false
  }, [])

  // ---- manifest ----------------------------------------------------------
  const eligibleSlots = useMemo(
    () => slots.filter((slot) => !!(slot.baked ?? slot.raw)),
    [slots],
  )
  const engineWarnings = useMemo(
    () => eligibleSlots
      .filter((slot) => slot.manipulation === "vc"
        && slot.engine && slot.engine !== "xvc")
      .map((slot) => `${slot.label}: VC bake uses non-production engine "${slot.engine}"`),
    [eligibleSlots],
  )
  const configurationErrors = useMemo(() => {
    const errors = eligibleSlots.flatMap((slot) => {
      const slotErrors: string[] = []
      if (!slot.sourceSpeaker) {
        slotErrors.push(`${slot.label}: source speaker is missing`)
      }
      if (slot.sampleRate !== PP_SAMPLE_RATE) {
        slotErrors.push(
          `${slot.label}: sample rate is ${slot.sampleRate}, expected ${PP_SAMPLE_RATE}`,
        )
      }
      if (!slot.normalized || slot.measuredLufs == null) {
        slotErrors.push(`${slot.label}: final clip is not loudness-normalized`)
      }
      if (Math.abs(slot.driftMs) > DURATION_DRIFT_TOLERANCE_MS) {
        slotErrors.push(`${slot.label}: duration drift is ${slot.driftMs} ms`)
      }
      if (slot.manipulation === "vc") {
        const target = targets.find((candidate) => candidate.id === slot.targetId)
        if (!slot.targetId || !target?.wav) {
          slotErrors.push(`${slot.label}: converted clip has no uploaded target WAV`)
        }
      }
      return slotErrors
    })
    const targetLevels = new Set(
      eligibleSlots
        .map((slot) => slot.targetLufs)
        .filter((value): value is number => value != null),
    )
    if (targetLevels.size > 1) {
      errors.push(
        `clips use different loudness targets: ${[...targetLevels].sort().join(", ")} LUFS`,
      )
    }
    return errors
  }, [eligibleSlots, targets])

  const generateManifest = useCallback(async () => {
    setError(null)
    setResultsZip(null)
    if (eligibleSlots.length === 0) {
      setError("No playable slots (record/bake in Configure Soundboard first).")
      return null
    }
    if (engineWarnings.length > 0 && !config.allowNonProductionEngines) {
      setError("Manifest contains non-X-VC VC bakes. Re-bake with X-VC, or "
        + "explicitly mark this run exploratory.")
      return null
    }
    if (configurationErrors.length > 0) {
      setError(`Audit configuration is incomplete: ${configurationErrors.join("; ")}`)
      return null
    }

    const baseTiming = {
      greetingSettleMs: config.greetingSettleMs,
      greetingCapMs: config.greetingCapMs,
      responseTimeoutMs: config.responseTimeoutMs,
      responseLingerMs: config.responseLingerMs,
      cooldownMs: config.cooldownMs,
      interruptFireDelayMs: INTERRUPT_FIRE_DELAY_MS,
      interruptFireToleranceMs: INTERRUPT_FIRE_TOLERANCE_MS,
      // Detection constants define onset, overlap, and speaking, so
      // they belong in the frozen record, not only in code.
      ppEnergyThresholdRms: PP_ENERGY_THRESHOLD,
      ppSpeakingGapMs: PP_SPEAKING_GAP_MS,
      minPreClipMs: MIN_PRE_CLIP_MS,
      overlapEventMinimumMs: AUDIT_OVERLAP_EVENT_MIN_MS,
    }
    const inputTimingCache = new Map<string, AuditInputTiming>()
    const targetHashCache = new Map<string, string>()
    const targetMetadataFor = async (slot: Slot) => {
      if (slot.manipulation !== "vc" || !slot.targetId) {
        return { id: null, label: null, sha256: null }
      }
      const target = targets.find((candidate) => candidate.id === slot.targetId)
      if (!target?.wav) return { id: slot.targetId, label: target?.label ?? null, sha256: null }
      const cached = targetHashCache.get(slot.targetId)
      const sha256 = cached ?? await sha256Hex(target.wav)
      targetHashCache.set(slot.targetId, sha256)
      return { id: slot.targetId, label: target.label, sha256 }
    }
    const inputTimingFor = async (
      slot: Slot,
      clip: Blob,
      clipHash: string,
      rawHash: string | null,
    ) => {
      // Matched pairs share the raw take and therefore share one annotation.
      // This prevents route-specific VC residue from shifting the nominal user
      // onset, offset, or pause windows.
      const source = slot.raw ?? clip
      const sourceHash = rawHash ?? clipHash
      let annotation = inputTimingCache.get(sourceHash)
      if (!annotation) {
        annotation = await analyzeAuditInput(source)
        inputTimingCache.set(sourceHash, annotation)
      }
      return { sourceHash, annotation }
    }

    // ---- script mode: ordered turns, replayed `reps` times --------------
    if (config.mode === "script") {
      const buildTurns = async (group: Slot[]): Promise<AuditScriptTurn[]> => {
        const turns: AuditScriptTurn[] = []
        for (const slot of group) {
          const clip = slot.baked ?? slot.raw!
          const clipHash = await sha256Hex(clip)
          const rawHash = slot.raw ? await sha256Hex(slot.raw) : null
          const inputTiming = await inputTimingFor(
            slot, clip, clipHash, rawHash
          )
          const target = await targetMetadataFor(slot)
          turns.push({
            turn: turns.length + 1,
            slot_id: slot.id,
            label: slot.label,
            condition: slot.condition,
            manipulation: slot.manipulation,
            route: routeForSlot(slot),
            source_speaker: slot.sourceSpeaker ?? null,
            engine: slot.manipulation === "vc" ? (slot.engine ?? "xvc") : null,
            target_id: target.id,
            target_label: target.label,
            target_sha256: target.sha256,
            clip_sha256: clipHash,
            raw_sha256: rawHash,
            clip_duration_ms: Math.round(slot.bakedDurationMs || slot.rawDurationMs),
            input_timing_source_sha256: inputTiming.sourceHash,
            input_timing: inputTiming.annotation,
            clip_audio: clipAudioForSlot(slot),
          })
        }
        return turns
      }
      const common = {
        schema: "hmo.soundboard-audit-manifest.v3" as const,
        mode: "script" as const,
        created_at: new Date().toISOString(),
        seed: config.seed,
        reps: config.reps,
        prompt: config.prompt,
        assistant: {
          engine: "personaplex" as const,
          voice_prompt: PERSONAPLEX_AUDIT_VOICE_PROMPT,
          text_prompt: config.prompt,
        },
        audio_format: {
          personaplex_sample_rate_hz: PP_SAMPLE_RATE,
          opus_encoder: OPUS_ENCODER_CONFIG,
        },
        inter_turn_gap_ms: config.interTurnGapMs,
        timing: { ...baseTiming, interTurnGapMs: config.interTurnGapMs },
        production_engine_only: engineWarnings.length === 0,
        engine_warnings: engineWarnings,
      }
      let draft: AuditManifest
      if (config.interleaveByCondition) {
        // One script per condition TAG, replays cycling through every tag
        // each round so no condition is confounded with time/server drift.
        const groups = new Map<string, Slot[]>()
        for (const slot of eligibleSlots) {
          const tag = slot.condition || "(untagged)"
          groups.set(tag, [...(groups.get(tag) ?? []), slot])
        }
        if (groups.size < 2) {
          setError("Interleave needs at least two condition tags among "
            + `playable slots (found ${groups.size}: `
            + `${[...groups.keys()].join(", ") || "none"}). Tag each slot's `
            + "condition in Configure Soundboard (e.g. natural / converted).")
          return null
        }
        const scripts = []
        for (const [condition, group] of groups) {
          scripts.push({ condition, turns: await buildTurns(group) })
        }
        const order = scripts.map((s) => s.condition)
        const plan: { rep: number; condition: string; cycle: number }[] = []
        for (let cycle = 1; cycle <= config.reps; cycle++) {
          for (const condition of order) {
            plan.push({ rep: plan.length + 1, condition, cycle })
          }
        }
        draft = { ...common, interleaved: true, scripts, replay_plan: plan }
      } else {
        draft = { ...common, script: await buildTurns(eligibleSlots) }
      }
      const digest = await sha256Hex(new Blob([JSON.stringify(draft)]))
      const frozen = { ...draft, manifest_sha256: digest }
      setManifest(frozen)
      return frozen
    }

    // ---- matched mode (§3.3): shuffled single-turn presentations --------
    const base: Omit<AuditPresentation, "index">[] = []
    const pairing = new Map<string, {
      labels: string[]
      manipulations: string[]
      source_speakers: string[]
    }>()
    for (const slot of eligibleSlots) {
      const clip = slot.baked ?? slot.raw!
      const hash = await sha256Hex(clip)
      const rawHash = slot.raw ? await sha256Hex(slot.raw) : null
      const inputTiming = await inputTimingFor(slot, clip, hash, rawHash)
      const target = await targetMetadataFor(slot)
      if (rawHash) {
        const entry = pairing.get(rawHash) ?? {
          labels: [],
          manipulations: [],
          source_speakers: [],
        }
        entry.labels.push(slot.label)
        if (!entry.manipulations.includes(slot.manipulation)) {
          entry.manipulations.push(slot.manipulation)
        }
        if (slot.sourceSpeaker
            && !entry.source_speakers.includes(slot.sourceSpeaker)) {
          entry.source_speakers.push(slot.sourceSpeaker)
        }
        pairing.set(rawHash, entry)
      }
      for (let r = 0; r < config.reps; r++) {
        base.push({
          slot_id: slot.id,
          label: slot.label,
          condition: slot.condition,
          manipulation: slot.manipulation,
          route: routeForSlot(slot),
          source_speaker: slot.sourceSpeaker ?? null,
          engine: slot.manipulation === "vc" ? (slot.engine ?? "xvc") : null,
          target_id: target.id,
          target_label: target.label,
          target_sha256: target.sha256,
          clip_sha256: hash,
          raw_sha256: rawHash,
          clip_duration_ms: Math.round(slot.bakedDurationMs || slot.rawDurationMs),
          input_timing_source_sha256: inputTiming.sourceHash,
          input_timing: inputTiming.annotation,
          clip_audio: clipAudioForSlot(slot),
          presentation_mode: interruptSlotIds.has(slot.id)
            ? "during_pp_speech" : "after_silence",
        })
      }
    }
    // Matched-pair accounting: a natural/converted item pair shares one raw
    // source, so a raw hash present in only one manipulation is unmatched.
    const pairingRows = Array.from(pairing.entries()).map(
      ([raw_sha256, entry]) => ({ raw_sha256, ...entry }))
    const pairingWarnings = pairingRows
      .flatMap((row) => {
        const warnings = []
        if (!(row.manipulations.includes("vc")
            && row.manipulations.includes("unconverted"))) {
          warnings.push(`unmatched natural/converted item: ${row.labels.join(", ")}`)
        }
        if (row.source_speakers.length !== 1) {
          warnings.push(`source-speaker mismatch: ${row.labels.join(", ")}`)
        }
        return warnings
      })
    const ordered = shuffled(base, config.seed)
      .map((p, i) => ({ index: i + 1, ...p }))
    const draft: AuditManifest = {
      schema: "hmo.soundboard-audit-manifest.v3",
      mode: "matched",
      created_at: new Date().toISOString(),
      seed: config.seed,
      reps: config.reps,
      prompt: config.prompt,
      assistant: {
        engine: "personaplex" as const,
        voice_prompt: PERSONAPLEX_AUDIT_VOICE_PROMPT,
        text_prompt: config.prompt,
      },
      audio_format: {
        personaplex_sample_rate_hz: PP_SAMPLE_RATE,
        opus_encoder: OPUS_ENCODER_CONFIG,
      },
      timing: baseTiming,
      production_engine_only: engineWarnings.length === 0,
      engine_warnings: engineWarnings,
      pairing: pairingRows,
      pairing_warnings: pairingWarnings,
      presentations: ordered,
    }
    const digest = await sha256Hex(new Blob([JSON.stringify(draft)]))
    const frozen = { ...draft, manifest_sha256: digest }
    setManifest(frozen)
    return frozen
  }, [
    eligibleSlots,
    engineWarnings,
    configurationErrors,
    config,
    interruptSlotIds,
    targets,
  ])

  // ---- the run -----------------------------------------------------------
  const runPresentation = useCallback(async (
    presentation: AuditPresentation,
  ): Promise<AuditRunRecord> => {
    const slot = slots.find((s) => s.id === presentation.slot_id)
    const record: AuditRunRecord = {
      index: presentation.index,
      slot_id: presentation.slot_id,
      label: presentation.label,
      presentation_mode: presentation.presentation_mode,
      status: "error",
      t_connect_ms: performance.now(),
      t_handshake_ms: null,
      t_play_start_ms: null,
      t_play_end_ms: null,
      t_pp_response_start_ms: null,
      t_pp_response_end_ms: null,
      response_latency_ms: null,
      fire_offset_into_pp_speech_ms: null,
      pp_yield_latency_ms: null,
      notes: [],
      pp_speech_events: [],
      pp_transcript: [],
      sent_clip_sha256: null,
      delivery_audit: null,
    }
    if (!slot || !(slot.baked ?? slot.raw)) {
      record.status = "clip_error"
      record.error = "slot missing or has no audio"
      return record
    }

    // Fresh conversation state for every presentation.
    ws.clearTranscripts()
    ws.clearResponseChunks()
    resetCapture()
    lastEnergeticMsRef.current = 0
    energeticRunStartMsRef.current = 0
    sawEnergyRef.current = false
    ppEventsRef.current = []

    ws.connect(getPersonaplexAuditWsURL(config.prompt))
    let silence: { stop: () => Promise<void> } | null = null
    try {
      if (!(await waitFor(() => handshakeRef.current, config.handshakeTimeoutMs))) {
        record.status = "handshake_timeout"
        return record
      }
      // Use the WebSocket hook's exact handshake epoch. playback_timeline.json
      // is relative to this same performance.now() value, so post-processing
      // can place scheduled assistant packets on the audit record's clock
      // without estimating an offset from React state propagation.
      record.t_handshake_ms = ws.getConversationStartPerformanceMs()

      // Feed PP its input clock (continuous silence) — without this PP never
      // generates: no greeting, and responses freeze when the clip ends.
      ws.setMicMuted(false)
      silence = await startSilenceFeed(ws)

      const handshakeAt = record.t_handshake_ms
      let interruptRunStart: number | null = null
      if (presentation.presentation_mode === "during_pp_speech") {
        // Corrective interruption: fire the clip WHILE PP is speaking
        // (energetically), a fixed delay into its speech run.
        const spoke = await waitFor(() => speakingNow(), config.greetingCapMs)
        if (spoke) {
          interruptRunStart = energeticRunStartMsRef.current
          const fireAt = interruptRunStart + INTERRUPT_FIRE_DELAY_MS
          await waitFor(() => performance.now() >= fireAt || !speakingNow(),
            INTERRUPT_FIRE_DELAY_MS + 1_000)
          if (!speakingNow()) {
            record.notes.push("pp_stopped_before_interrupt_fire")
          }
        } else {
          record.notes.push("pp_never_spoke_before_interrupt_cap")
        }
      } else {
        // Play only AFTER PP has actually greeted (PP always does) and then
        // gone energetically quiet for greetingSettleMs — not on a bare timer.
        // The cap is a safety net; if it fires without PP ever speaking, note
        // it (PP wedged / didn't greet).
        const greetingDeadline = performance.now() + config.greetingCapMs
        await waitFor(
          () => (sawEnergyRef.current
                  && performance.now() - handshakeAt >= MIN_PRE_CLIP_MS
                  && !speakingNow()
                  && performance.now() - lastEnergeticMsRef.current >= config.greetingSettleMs)
                || performance.now() >= greetingDeadline,
          config.greetingCapMs + 1_000,
        )
        if (!sawEnergyRef.current) record.notes.push("pp_never_greeted")
      }

      const playbackStart = await playback.playSlot(slot, { auditDelivery: true })
      if (!playbackStart) {
        record.status = "clip_error"
        record.error = playback.error || "clip did not start"
        return record
      }
      record.t_play_start_ms = playbackStart.startMs
      const wasSpeakingAtFire = speakingNow()
      if (presentation.presentation_mode === "during_pp_speech"
          && interruptRunStart != null) {
        record.fire_offset_into_pp_speech_ms = +(
          playbackStart.startMs - interruptRunStart
        ).toFixed(1)
      }
      // Yielding (interrupt mode): how long PP kept speaking after the clip
      // started. Resolved as soon as PP's ongoing run ends, checked below
      // after the clip finishes (events carry the exact end time).
      if (presentation.presentation_mode === "during_pp_speech" && !wasSpeakingAtFire) {
        record.notes.push("pp_not_speaking_at_clip_start")
      }
      // playSlot resolves once streaming starts; wait for the clip to finish
      // (playingSlotId returns to null in useSoundboardPlayback.stop()).
      await waitFor(() => playingRef.current === presentation.slot_id, 3_000)
      const clipBudget = presentation.clip_duration_ms + 10_000
      if (!(await waitFor(() => playingRef.current === null, clipBudget))) {
        record.status = "clip_error"
        record.error = "clip did not finish within budget"
        return record
      }
      const playbackEnd = playback.getLastPlaybackRecord()
      record.delivery_audit = playbackEnd?.deliveryAudit ?? null
      const playEnd = playbackEnd?.slotId === presentation.slot_id
        ? playbackEnd.endMs
        : playbackStart.startMs + playbackStart.clipDurationMs
      record.t_play_end_ms = playEnd

      // PP's response: first ENERGETIC packet after the clip ended. (The
      // authoritative response latency is recomputed from packet RMS in the
      // post-processor; this live value drives the runner and the UI.)
      const responded = await waitFor(
        () => lastEnergeticMsRef.current > playEnd,
        config.responseTimeoutMs,
      )
      if (!responded) {
        record.status = "no_response"
      } else {
        record.t_pp_response_start_ms = lastEnergeticMsRef.current
        record.response_latency_ms = +(lastEnergeticMsRef.current - playEnd).toFixed(1)
        // Response is over once PP has been energetically silent for
        // responseLingerMs (silence frames don't reset the timer).
        await waitFor(
          () => !speakingNow()
            && lastEnergeticMsRef.current > playEnd
            && performance.now() - lastEnergeticMsRef.current >= config.responseLingerMs,
          120_000,
        )
        record.t_pp_response_end_ms = lastEnergeticMsRef.current || null
        record.status = "ok"
      }

      record.pp_transcript = transcriptsRef.current.map(
        (t) => ({ text: t.text, speaker: t.speaker }))
      const captured = getCapturedClips()
      if (captured.length > 0) {
        record.sent_clip_sha256 = await sha256Hex(captured[0].sentBlob)
      }
      return record
    } finally {
      // Runs on every exit path (success, timeout, abort) so even failed
      // presentations carry their full evidence.
      record.pp_speech_events = ppEventsRef.current.slice()
      if (record.presentation_mode === "during_pp_speech"
          && record.t_play_start_ms != null
          && record.fire_offset_into_pp_speech_ms != null) {
        // Yielding: from clip onset until PP's post-clip speech run ends —
        // i.e. the last energetic packet before the first gap > the speaking
        // window. (Silence frames alone don't count as still speaking.)
        const playStart = record.t_play_start_ms
        const energy = record.pp_speech_events
          .filter((e) => e.type === "pp_energy" && e.timestampMs >= playStart)
          .map((e) => e.timestampMs)
          .sort((a, b) => a - b)
        let yieldAt: number | null = null
        for (let i = 0; i < energy.length; i++) {
          if (energy[i + 1] == null || energy[i + 1] - energy[i] > PP_SPEAKING_GAP_MS) {
            yieldAt = energy[i]
            break
          }
        }
        record.pp_yield_latency_ms = yieldAt != null
          ? +(yieldAt - playStart).toFixed(1) : null
      }
      // Stop the silence feed first, then collect PP audio + the packet-level
      // playback timeline BEFORE disconnect resets state, then tear down.
      if (silence) await silence.stop().catch(() => undefined)
      record._ppWav = await ws.getPersonaplexWav().catch(() => null)
      record._sentWav = await assembleSentWav().catch(() => null)
      record._playbackTimeline = await ws.getClientPlaybackTimeline()
        .catch(() => null)
      ws.disconnect()
    }
  }, [slots, ws, playback, config, waitFor, speakingNow])

  // ---- script mode: one full ordered conversation, replayed per rep ------
  const runScriptSession = useCallback(async (
    turns: AuditScriptTurn[], rep: number,
  ): Promise<AuditSessionRecord> => {
    const record: AuditSessionRecord = {
      rep, status: "error", t_connect_ms: performance.now(), t_handshake_ms: null,
      greeted: false, turns: [], pp_speech_events: [], pp_transcript: [],
    }
    ws.clearTranscripts()
    ws.clearResponseChunks()
    resetCapture()
    lastEnergeticMsRef.current = 0
    energeticRunStartMsRef.current = 0
    sawEnergyRef.current = false
    ppEventsRef.current = []

    ws.connect(getPersonaplexAuditWsURL(config.prompt))
    let silence: { stop: () => Promise<void> } | null = null
    const gap = config.interTurnGapMs
    try {
      if (!(await waitFor(() => handshakeRef.current, config.handshakeTimeoutMs))) {
        record.status = "handshake_timeout"
        return record
      }
      record.t_handshake_ms = ws.getConversationStartPerformanceMs()
      ws.setMicMuted(false)
      silence = await startSilenceFeed(ws)

      // Wait for PP's greeting to FINISH (audible quiet ≥ greetingSettleMs —
      // energy timestamps are playback-aligned), then the protocol gap.
      const handshakeAt = record.t_handshake_ms
      const greetingDeadline = performance.now() + config.greetingCapMs
      await waitFor(
        () => (sawEnergyRef.current
                && performance.now() - handshakeAt >= MIN_PRE_CLIP_MS
                && !speakingNow()
                && performance.now() - lastEnergeticMsRef.current >= config.greetingSettleMs)
              || performance.now() >= greetingDeadline,
        config.greetingCapMs + 1_000,
      )
      record.greeted = sawEnergyRef.current
      await sleep(gap)

      for (const turn of turns) {
        if (abortRef.current) throw new Error("aborted")
        const slot = slots.find((s) => s.id === turn.slot_id)
        const tr: AuditTurnRecord = {
          turn: turn.turn, slot_id: turn.slot_id, label: turn.label,
          status: "clip_error",
          t_play_start_ms: null, t_play_end_ms: null,
          t_pp_response_start_ms: null, t_pp_response_end_ms: null,
          response_latency_ms: null, pp_spoke_during_clip: false,
          sent_clip_sha256: null, notes: [],
          delivery_audit: null,
        }
        if (!slot || !(slot.baked ?? slot.raw)) {
          tr.notes.push("slot missing or has no audio")
          record.turns.push(tr)
          continue
        }
        const playbackStart = await playback.playSlot(slot, { auditDelivery: true })
        if (!playbackStart) {
          tr.status = "clip_error"
          tr.notes.push(playback.error || "clip did not start")
          record.turns.push(tr)
          continue
        }
        tr.t_play_start_ms = playbackStart.startMs
        await waitFor(() => playingRef.current === turn.slot_id, 3_000)
        if (!(await waitFor(() => playingRef.current === null,
                            turn.clip_duration_ms + 10_000))) {
          tr.status = "clip_error"
          tr.notes.push("clip did not finish within budget")
          record.turns.push(tr)
          continue
        }
        const playbackEnd = playback.getLastPlaybackRecord()
        tr.delivery_audit = playbackEnd?.deliveryAudit ?? null
        const playEnd = playbackEnd?.slotId === turn.slot_id
          ? playbackEnd.endMs
          : playbackStart.startMs + playbackStart.clipDurationMs
        tr.t_play_end_ms = playEnd
        // Response: first energetic PP packet after the clip ends. PP's
        // utterance is FINISHED only after responseLingerMs of audible quiet
        // (sentence pauses are shorter than that; the timestamps are
        // playback-aligned so this is silence at the speakers, not at the
        // network). THEN the protocol inter-turn gap runs before the next
        // line — "PP finishes → 500 ms → next line".
        if (await waitFor(() => lastEnergeticMsRef.current > playEnd, config.responseTimeoutMs)) {
          tr.t_pp_response_start_ms = lastEnergeticMsRef.current
          tr.response_latency_ms = +(lastEnergeticMsRef.current - playEnd).toFixed(1)
          await waitFor(
            () => !speakingNow()
              && lastEnergeticMsRef.current > playEnd
              && performance.now() - lastEnergeticMsRef.current >= config.responseLingerMs,
            120_000,
          )
          tr.t_pp_response_end_ms = lastEnergeticMsRef.current || null
          tr.status = "ok"
        } else {
          tr.status = "no_response"
        }
        await sleep(gap)
        const captured = getCapturedClips()
        if (captured.length > 0) {
          tr.sent_clip_sha256 = await sha256Hex(captured[captured.length - 1].sentBlob)
        }
        record.turns.push(tr)
      }
      record.status = "ok"
      record.pp_transcript = transcriptsRef.current.map(
        (t) => ({ text: t.text, speaker: t.speaker }))
      return record
    } finally {
      record.pp_speech_events = ppEventsRef.current.slice()
      // Coarse live overlap flag; scheduled packet timing is authoritative.
      for (const tr of record.turns) {
        if (tr.t_play_start_ms != null && tr.t_play_end_ms != null) {
          tr.pp_spoke_during_clip = record.pp_speech_events.some(
            (e) => e.type === "pp_energy"
              && e.timestampMs >= tr.t_play_start_ms!
              && e.timestampMs <= tr.t_play_end_ms!)
        }
      }
      if (silence) await silence.stop().catch(() => undefined)
      record._ppWav = await ws.getPersonaplexWav().catch(() => null)
      record._sentWav = await assembleSentTimelineWav().catch(() => null)
      record._playbackTimeline = await ws.getClientPlaybackTimeline().catch(() => null)
      ws.disconnect()
    }
  }, [slots, ws, playback, config, waitFor, speakingNow])

  const start = useCallback(async () => {
    if (!manifest) {
      setError("Generate and freeze the manifest first.")
      return
    }
    if (ws.connected) {
      setError("A conversation is already open — stop it before running the audit.")
      return
    }
    onBeforeRun?.()
    abortRef.current = false
    setError(null)
    setResultsZip(null)
    const records: (AuditRunRecord | AuditSessionRecord)[] = []
    const audio: { name: string; blob: Blob }[] = []
    // Every conversation's WAVs are shipped as they complete and dropped from
    // memory: holding a whole run's audio in the tab exhausts it (a 240-replay
    // run died at replay 7), and a crashed tab cannot bundle anything.
    const runId = `${new Date().toISOString().replace(/[-:]/g, "").slice(0, 15)}Z`
      + `_${(manifest.manifest_sha256 ?? "nohash").slice(0, 8)}`
    let checkpointFailed = false
    const checkpoint = async (
      entries: { name: string; blob: Blob }[], done: boolean,
    ) => {
      try {
        await checkpointSoundboardAudit(runId, {
          part: entries.length
            ? makeZip(await Promise.all(entries.map(async (e) => ({
                name: e.name, data: new Uint8Array(await e.blob.arrayBuffer()),
              }))))
            : undefined,
          runLog: { completed_at: new Date().toISOString(),
                    aborted: abortRef.current, complete: done, records },
        })
      } catch (e) {
        // Keep running: the local bundle at the end is still a fallback.
        checkpointFailed = true
        setUploadState({ status: "failed", detail: String((e as Error).message ?? e) })
      }
    }
    await checkpoint([], false)
    try {
      await checkpointSoundboardAudit(runId, { manifest })
    } catch { /* reported by the next checkpoint */ }
    // Script mode runs a replay PLAN: either N single-script replays, or the
    // interleaved A,B,A,B… plan from the manifest.
    const plan = manifest.mode !== "script" ? [] : (
      manifest.interleaved && manifest.replay_plan && manifest.scripts
        ? manifest.replay_plan.map((entry) => ({
            rep: entry.rep,
            condition: entry.condition as string | null,
            turns: manifest.scripts!.find(
              (s) => s.condition === entry.condition)?.turns ?? [],
          }))
        : Array.from({ length: manifest.reps }, (_, i) => ({
            rep: i + 1, condition: null as string | null,
            turns: manifest.script ?? [],
          }))
    )
    const total = manifest.mode === "script"
      ? plan.length : (manifest.presentations?.length ?? 0)
    setProgress({ running: true, currentIndex: 0, total, phase: "starting", records: [] })
    try {
      if (manifest.mode === "script") {
        for (const entry of plan) {
          const { rep, condition, turns } = entry
          if (abortRef.current) break
          setProgress((p) => ({ ...p, currentIndex: rep,
            phase: `replay ${rep}/${plan.length}`
              + `${condition ? ` [${condition}]` : ""} — ${turns.length}-turn script` }))
          let record: AuditSessionRecord
          try {
            record = await runScriptSession(turns, rep)
          } catch (e) {
            record = {
              rep, status: abortRef.current ? "aborted" : "error",
              error: String((e as Error).message ?? e),
              t_connect_ms: performance.now(), t_handshake_ms: null,
              greeted: false, turns: [],
              pp_speech_events: ppEventsRef.current.slice(), pp_transcript: [],
            }
            try { ws.disconnect() } catch { /* already down */ }
          }
          record.condition = condition
          const suffix = condition
            ? `_${condition.toLowerCase().replace(/[^a-z0-9]+/g, "-")}` : ""
          const prefix = `runs/rep_${String(rep).padStart(3, "0")}${suffix}`
          const ppWav = record._ppWav
          const sentWav = record._sentWav
          const timeline = record._playbackTimeline
          delete record._ppWav
          delete record._sentWav
          delete record._playbackTimeline
          const shipped: { name: string; blob: Blob }[] = []
          if (ppWav) shipped.push({ name: `${prefix}/personaplex.wav`, blob: ppWav })
          if (sentWav) shipped.push({ name: `${prefix}/sent.wav`, blob: sentWav })
          if (timeline) {
            shipped.push({ name: `${prefix}/playback_timeline.json`,
              blob: new Blob([JSON.stringify(timeline)]) })
          }
          records.push(record)
          setProgress((p) => ({ ...p, records: records.slice() }))
          // Ship and release: only fall back to accumulating in the tab if the
          // server is not taking checkpoints.
          await checkpoint(shipped, false)
          if (checkpointFailed) audio.push(...shipped)
          if (abortRef.current) break
          await sleep(config.cooldownMs)
        }
      } else {
        for (const presentation of (manifest.presentations ?? [])) {
          if (abortRef.current) break
          setProgress((p) => ({ ...p, currentIndex: presentation.index,
                                phase: `run ${presentation.index}/${total}: ${presentation.label}` }))
          let record: AuditRunRecord
          try {
            record = await runPresentation(presentation)
          } catch (e) {
            record = {
              index: presentation.index, slot_id: presentation.slot_id,
              label: presentation.label,
              presentation_mode: presentation.presentation_mode,
              status: abortRef.current ? "aborted" : "error",
              error: String((e as Error).message ?? e),
              t_connect_ms: performance.now(), t_handshake_ms: null,
              t_play_start_ms: null, t_play_end_ms: null,
              t_pp_response_start_ms: null, t_pp_response_end_ms: null,
              response_latency_ms: null,
              fire_offset_into_pp_speech_ms: null, pp_yield_latency_ms: null,
              notes: [], pp_speech_events: ppEventsRef.current.slice(),
              pp_transcript: [], sent_clip_sha256: null, delivery_audit: null,
            }
            try { ws.disconnect() } catch { /* already down */ }
          }
          const prefix = `runs/${String(presentation.index).padStart(3, "0")}_${presentation.slot_id}`
          const ppWav = record._ppWav
          const sentWav = record._sentWav
          const timeline = record._playbackTimeline
          delete record._ppWav
          delete record._sentWav
          delete record._playbackTimeline
          if (ppWav) audio.push({ name: `${prefix}/personaplex.wav`, blob: ppWav })
          if (sentWav) audio.push({ name: `${prefix}/sent.wav`, blob: sentWav })
          if (timeline) {
            audio.push({
              name: `${prefix}/playback_timeline.json`,
              blob: new Blob([JSON.stringify(timeline)]),
            })
          }
          records.push(record)
          setProgress((p) => ({ ...p, records: records.slice() }))
          if (abortRef.current) break
          await sleep(config.cooldownMs)
        }
      }
    } finally {
      setProgress((p) => ({ ...p, running: false, phase: abortRef.current ? "aborted" : "finished" }))
      // The frozen STIMULI go into the bundle too: the manifest's hashes prove
      // which bytes ran, but the clips themselves live only in this browser's
      // IndexedDB — without embedding them the audit artifacts reference audio
      // that may no longer exist anywhere. Deduped by hash; raw sources
      // included so matched pairs can be re-verified from the zip alone.
      const referenced = manifest.mode === "script"
        ? (manifest.interleaved
            ? (manifest.scripts ?? []).flatMap((s) => s.turns)
            : (manifest.script ?? []))
        : (manifest.presentations ?? [])
      const seenHashes = new Set<string>()
      const slug = (t: string) => t.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 40)
      for (const item of referenced) {
        const slot = slots.find((s) => s.id === item.slot_id)
        if (!slot) continue
        const clip = slot.baked ?? slot.raw
        if (clip && !seenHashes.has(item.clip_sha256)) {
          seenHashes.add(item.clip_sha256)
          audio.push({
            name: `stimuli/${item.clip_sha256.slice(0, 8)}_${slug(item.label)}.wav`,
            blob: clip,
          })
        }
        if (slot.raw && item.raw_sha256 && !seenHashes.has(item.raw_sha256)) {
          seenHashes.add(item.raw_sha256)
          audio.push({
            name: `stimuli/raw_${item.raw_sha256.slice(0, 8)}_${slug(item.label)}.wav`,
            blob: slot.raw,
          })
        }
        if (item.target_id && item.target_sha256
            && !seenHashes.has(item.target_sha256)) {
          const target = targets.find((candidate) => candidate.id === item.target_id)
          if (target?.wav) {
            seenHashes.add(item.target_sha256)
            audio.push({
              name: `stimuli/target_${item.target_sha256.slice(0, 8)}_`
                + `${slug(target.label)}.wav`,
              blob: target.wav,
            })
          }
        }
      }
      // Bundle everything, even a partial/aborted run — partial data beats none.
      const entries = [
        { name: "manifest.json",
          data: new TextEncoder().encode(JSON.stringify(manifest, null, 2)) },
        { name: "run_log.json",
          data: new TextEncoder().encode(JSON.stringify({
            completed_at: new Date().toISOString(),
            aborted: abortRef.current,
            records,
          }, null, 2)) },
        ...await Promise.all(audio.map(async (a) => ({
          name: a.name, data: new Uint8Array(await a.blob.arrayBuffer()),
        }))),
      ]
      const zip = makeZip(entries)
      setResultsZip(zip)
      // Close the checkpointed run out; only re-upload the whole bundle when
      // checkpointing failed, since the audio is already on the volume.
      setUploadState({ status: "uploading" })
      try {
        if (checkpointFailed) {
          const stored = await uploadSoundboardAudit(zip, manifest.manifest_sha256 ?? "")
          setUploadState({ status: "done", detail: stored.path })
        } else {
          const stored = await checkpointSoundboardAudit(runId, {
            manifest,
            runLog: { completed_at: new Date().toISOString(),
                      aborted: abortRef.current, complete: true, records },
          })
          setUploadState({ status: "done", detail: stored.path })
        }
      } catch (e) {
        setUploadState({ status: "failed", detail: String((e as Error).message ?? e) })
      }
    }
  }, [
    manifest,
    ws,
    config.cooldownMs,
    runPresentation,
    runScriptSession,
    onBeforeRun,
    slots,
    targets,
  ])

  const retryUpload = useCallback(async () => {
    if (!resultsZip || !manifest) return
    setUploadState({ status: "uploading" })
    try {
      const stored = await uploadSoundboardAudit(resultsZip, manifest.manifest_sha256 ?? "")
      setUploadState({ status: "done", detail: stored.path })
    } catch (e) {
      setUploadState({ status: "failed", detail: String((e as Error).message ?? e) })
    }
  }, [resultsZip, manifest])

  const abort = useCallback(() => { abortRef.current = true }, [])

  const downloadResults = useCallback(() => {
    if (!resultsZip || !manifest) return
    const a = document.createElement("a")
    a.href = URL.createObjectURL(resultsZip)
    a.download = `soundboard_audit_${manifest.manifest_sha256?.slice(0, 8)}.zip`
    a.click()
  }, [resultsZip, manifest])

  const downloadManifest = useCallback(() => {
    if (!manifest) return
    const a = document.createElement("a")
    a.href = URL.createObjectURL(new Blob(
      [JSON.stringify(manifest, null, 2)], { type: "application/json" }))
    a.download = `soundboard_audit_manifest_${manifest.manifest_sha256?.slice(0, 8)}.json`
    a.click()
  }, [manifest])

  return {
    config, setConfig,
    manifest, generateManifest, downloadManifest,
    eligibleCount: eligibleSlots.length,
    engineWarnings,
    progress, error,
    start, abort,
    resultsZip, downloadResults,
    uploadState, retryUpload,
    interruptSlotIds, toggleInterruptSlot,
    eligibleSlots,
  }
}
