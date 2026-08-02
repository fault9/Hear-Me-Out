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

import { useCallback, useEffect, useRef, useState } from "react"
import { getPersonaplexWsURL } from "@/lib/config"
import { uploadSoundboardAudit } from "@shared/services/api"
import { makeZip } from "@/lib/soundboardZip"
import { assembleSentWav, getCapturedClips, resetCapture } from "@/lib/soundboardCapture"
import type { Slot } from "@/lib/soundboardDb"
import type { useWebSocket } from "@shared/hooks/useWebSocket"
import type { useSoundboardPlayback } from "@/hooks/useSoundboardPlayback"

type WsState = ReturnType<typeof useWebSocket>
type PlaybackState = ReturnType<typeof useSoundboardPlayback>

export interface AuditConfig {
  reps: number
  seed: number
  prompt: string
  greetingSettleMs: number     // PP silent this long after its greeting → play
  greetingCapMs: number        // play anyway after this long post-handshake
  responseTimeoutMs: number    // no PP speech after clip end → "no_response"
  responseLingerMs: number     // PP silent this long after response → done
  handshakeTimeoutMs: number
  cooldownMs: number           // between presentations (PP teardown)
  allowNonProductionEngines: boolean
}

export const DEFAULT_AUDIT_CONFIG: AuditConfig = {
  reps: 2,
  seed: 0,
  prompt: "You enjoy having a good conversation.",
  greetingSettleMs: 900,
  greetingCapMs: 10_000,
  responseTimeoutMs: 20_000,
  responseLingerMs: 1_500,
  handshakeTimeoutMs: 45_000,
  cooldownMs: 3_000,
  allowNonProductionEngines: false,
}

// Interrupt presentations fire this far into PP's ongoing speech run, so
// every corrective interruption lands at a comparable point of a PP turn.
export const INTERRUPT_FIRE_DELAY_MS = 800

export type PresentationMode = "after_silence" | "during_pp_speech"

export interface AuditPresentation {
  index: number                 // 1-based, in frozen (shuffled) order
  slot_id: string
  label: string
  condition: string
  manipulation: string
  engine: string | null
  clip_sha256: string
  // Hash of the slot's RAW source take. Matched natural/converted pairs share
  // one source WAV, so raw_sha256 equality across a pair IS the proof that
  // linguistic content and source timing are held constant.
  raw_sha256: string | null
  clip_duration_ms: number
  // after_silence: wait for PP to finish speaking, then play (default).
  // during_pp_speech: fire the clip WHILE PP is speaking (corrective-
  // interruption items) and measure yielding.
  presentation_mode: PresentationMode
}

export interface AuditManifest {
  schema: "hmo.soundboard-audit-manifest.v2"
  created_at: string
  seed: number
  reps: number
  prompt: string
  timing: Pick<AuditConfig, "greetingSettleMs" | "greetingCapMs"
    | "responseTimeoutMs" | "responseLingerMs" | "cooldownMs">
    & { interruptFireDelayMs: number }
  production_engine_only: boolean
  engine_warnings: string[]
  // Matched-pair accounting: raw_sha256 -> the manipulations present. Items
  // whose source appears in only one condition are listed as warnings.
  pairing: { raw_sha256: string; labels: string[]; manipulations: string[] }[]
  pairing_warnings: string[]
  presentations: AuditPresentation[]
  manifest_sha256?: string      // hash of the manifest WITHOUT this field
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
  response_latency_ms: number | null   // clip end → PP speech start
  // Interrupt presentations: how far into PP's speech run the clip fired, and
  // how long PP kept speaking after the clip started (yielding). null when PP
  // was not actually speaking at fire time (see notes).
  fire_offset_into_pp_speech_ms: number | null
  pp_yield_latency_ms: number | null
  notes: string[]
  // EVERY PP speech event during this presentation, on performance.now().
  // Overlap during the clip window, premature onsets in ambiguous pauses, and
  // yielding are all computed from this list downstream.
  pp_speech_events: { type: string; timestampMs: number }[]
  pp_transcript: { text: string; speaker: string }[]
  sent_clip_sha256: string | null
  // Internal carriers between runPresentation and the zip bundler; stripped
  // from the record before it lands in run_log.json.
  _ppWav?: Blob | null
  _sentWav?: Blob | null
  _playbackTimeline?: unknown
}

export interface AuditProgress {
  running: boolean
  currentIndex: number          // 1-based presentation being run (0 = none)
  total: number
  phase: string                 // human-readable current step
  records: AuditRunRecord[]
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

export function useSoundboardAudit(opts: {
  ws: WsState
  playback: PlaybackState
  slots: Slot[]
  onBeforeRun?: () => void      // e.g. panel turns its autoplay off
}) {
  const { ws, playback, slots, onBeforeRun } = opts
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
  const ppSpeakingRef = useRef(false)
  const ppLastStartRef = useRef(0)
  const ppLastEndRef = useRef(0)
  const ppSawAnyRef = useRef(false)
  const ppEventsRef = useRef<{ type: string; timestampMs: number }[]>([])
  const { registerPpSpeechListener } = ws
  useEffect(() => {
    return registerPpSpeechListener((e) => {
      ppEventsRef.current.push({ type: e.type, timestampMs: e.timestampMs })
      if (e.type === "pp_speech_start") {
        ppSpeakingRef.current = true
        ppSawAnyRef.current = true
        ppLastStartRef.current = e.timestampMs
      } else {
        ppSpeakingRef.current = false
        ppLastEndRef.current = e.timestampMs
      }
    })
  }, [registerPpSpeechListener])

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
  const eligibleSlots = slots.filter((s) => !!(s.baked ?? s.raw))
  const engineWarnings = eligibleSlots
    .filter((s) => s.manipulation === "vc" && s.engine && s.engine !== "xvc")
    .map((s) => `${s.label}: VC bake uses non-production engine "${s.engine}"`)

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
    const base: Omit<AuditPresentation, "index">[] = []
    const pairing = new Map<string, { labels: string[]; manipulations: string[] }>()
    for (const slot of eligibleSlots) {
      const clip = slot.baked ?? slot.raw!
      const hash = await sha256Hex(clip)
      const rawHash = slot.raw ? await sha256Hex(slot.raw) : null
      if (rawHash) {
        const entry = pairing.get(rawHash) ?? { labels: [], manipulations: [] }
        entry.labels.push(slot.label)
        if (!entry.manipulations.includes(slot.manipulation)) {
          entry.manipulations.push(slot.manipulation)
        }
        pairing.set(rawHash, entry)
      }
      for (let r = 0; r < config.reps; r++) {
        base.push({
          slot_id: slot.id,
          label: slot.label,
          condition: slot.condition,
          manipulation: slot.manipulation,
          engine: slot.manipulation === "vc" ? (slot.engine ?? "xvc") : null,
          clip_sha256: hash,
          raw_sha256: rawHash,
          clip_duration_ms: Math.round(slot.bakedDurationMs || slot.rawDurationMs),
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
      .filter((row) => !(row.manipulations.includes("vc")
        && (row.manipulations.includes("unconverted")
          || row.manipulations.includes("pitch_formant"))))
      .map((row) => `unmatched item (single condition): ${row.labels.join(", ")}`)
    const ordered = shuffled(base, config.seed)
      .map((p, i) => ({ index: i + 1, ...p }))
    const draft: AuditManifest = {
      schema: "hmo.soundboard-audit-manifest.v2",
      created_at: new Date().toISOString(),
      seed: config.seed,
      reps: config.reps,
      prompt: config.prompt,
      timing: {
        greetingSettleMs: config.greetingSettleMs,
        greetingCapMs: config.greetingCapMs,
        responseTimeoutMs: config.responseTimeoutMs,
        responseLingerMs: config.responseLingerMs,
        cooldownMs: config.cooldownMs,
        interruptFireDelayMs: INTERRUPT_FIRE_DELAY_MS,
      },
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
  }, [eligibleSlots, engineWarnings, config, interruptSlotIds])

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
    ppSpeakingRef.current = false
    ppSawAnyRef.current = false
    ppLastStartRef.current = 0
    ppLastEndRef.current = 0
    ppEventsRef.current = []

    ws.connect(getPersonaplexWsURL(config.prompt))
    try {
      if (!(await waitFor(() => handshakeRef.current, config.handshakeTimeoutMs))) {
        record.status = "handshake_timeout"
        return record
      }
      record.t_handshake_ms = performance.now()

      if (presentation.presentation_mode === "during_pp_speech") {
        // Corrective interruption: fire the clip WHILE PP is speaking, a fixed
        // delay into its speech run, so every interruption lands at a
        // comparable point of a PP turn.
        const spoke = await waitFor(
          () => ppSpeakingRef.current, config.greetingCapMs)
        if (spoke) {
          const runStart = ppLastStartRef.current
          const fireAt = runStart + INTERRUPT_FIRE_DELAY_MS
          await waitFor(() => performance.now() >= fireAt
            || !ppSpeakingRef.current, INTERRUPT_FIRE_DELAY_MS + 1_000)
          if (ppSpeakingRef.current) {
            record.fire_offset_into_pp_speech_ms =
              +(performance.now() - runStart).toFixed(1)
          } else {
            record.notes.push("pp_stopped_before_interrupt_fire")
          }
        } else {
          record.notes.push("pp_never_spoke_before_interrupt_cap")
        }
      } else {
        // Let PP's opening greeting run its course: play once PP has spoken
        // and then stayed silent for greetingSettleMs, or after greetingCapMs.
        const greetingDeadline = performance.now() + config.greetingCapMs
        await waitFor(
          () => (ppSawAnyRef.current
                  && !ppSpeakingRef.current
                  && performance.now() - ppLastEndRef.current >= config.greetingSettleMs)
                || performance.now() >= greetingDeadline,
          config.greetingCapMs + 1_000,
        )
      }

      const wasSpeakingAtFire = ppSpeakingRef.current
      record.t_play_start_ms = performance.now()
      await playback.playSlot(slot)
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
      const playEnd = performance.now()
      record.t_play_end_ms = playEnd

      // PP's response: first speech run STARTING after the clip ended.
      const responded = await waitFor(
        () => ppLastStartRef.current >= playEnd,
        config.responseTimeoutMs,
      )
      if (!responded) {
        record.status = "no_response"
      } else {
        record.t_pp_response_start_ms = ppLastStartRef.current
        record.response_latency_ms = +(ppLastStartRef.current - playEnd).toFixed(1)
        // Response is over once PP has been silent for responseLingerMs.
        await waitFor(
          () => !ppSpeakingRef.current
            && ppLastEndRef.current >= playEnd
            && performance.now() - ppLastEndRef.current >= config.responseLingerMs,
          120_000,
        )
        record.t_pp_response_end_ms = ppLastEndRef.current || null
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
        // Yielding: first PP speech END at/after the clip's start.
        const playStart = record.t_play_start_ms
        const yieldEnd = record.pp_speech_events.find(
          (e) => e.type === "pp_speech_end" && e.timestampMs >= playStart)
        record.pp_yield_latency_ms = yieldEnd
          ? +(yieldEnd.timestampMs - playStart).toFixed(1) : null
      }
      // Collect PP audio + packet-level playback timeline BEFORE disconnect
      // resets state, then tear down.
      record._ppWav = await ws.getPersonaplexWav().catch(() => null)
      record._sentWav = await assembleSentWav().catch(() => null)
      record._playbackTimeline = await ws.getClientPlaybackTimeline()
        .catch(() => null)
      ws.disconnect()
    }
  }, [slots, ws, playback, config, waitFor])

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
    const records: AuditRunRecord[] = []
    const audio: { name: string; blob: Blob }[] = []
    setProgress({ running: true, currentIndex: 0,
                  total: manifest.presentations.length, phase: "starting", records: [] })
    try {
      for (const presentation of manifest.presentations) {
        if (abortRef.current) break
        setProgress((p) => ({ ...p, currentIndex: presentation.index,
                              phase: `run ${presentation.index}/${manifest.presentations.length}: ${presentation.label}` }))
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
            pp_transcript: [], sent_clip_sha256: null,
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
    } finally {
      setProgress((p) => ({ ...p, running: false, phase: abortRef.current ? "aborted" : "finished" }))
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
      // Persist to the server data volume (best-effort; the researcher can
      // retry from the UI or fall back to the local download).
      setUploadState({ status: "uploading" })
      try {
        const stored = await uploadSoundboardAudit(zip, manifest.manifest_sha256 ?? "")
        setUploadState({ status: "done", detail: stored.path })
      } catch (e) {
        setUploadState({ status: "failed", detail: String((e as Error).message ?? e) })
      }
    }
  }, [manifest, ws, config.cooldownMs, runPresentation, onBeforeRun])

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
