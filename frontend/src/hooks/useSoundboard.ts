// ============================================================================
//  USE SOUNDBOARD — orchestration hook
// ----------------------------------------------------------------------------
//  Owns the slot/target/session state. The Configure tab consumes most of
//  this; the runtime panel consumes only the read parts + session-log writes.
//
//  Recording, bake (VC / pitch-formant), preview, session-log, export/import.
// ============================================================================

import { useCallback, useEffect, useRef, useState } from "react"
import {
  PP_SAMPLE_RATE,
  PP_CHANNELS,
  MAX_RECORDING_SECONDS,
  MIN_RECORDING_SECONDS,
  PITCH_FORMANT_DEFAULTS,
  LOUDNESS_TARGET_LUFS,
  DEFAULT_TARGET,
} from "@/lib/soundboardConfig"
import {
  decodeAndConformToPp,
  encodeWavAtPpRate,
  checkDuration,
} from "@/lib/audioFormat"
import {
  type Slot,
  type Target,
  type SessionRow,
  type ManipulationMode,
  ensureBuiltinTarget,
  listSlots,
  listTargets,
  putSlot,
  putTarget,
  deleteSlot,
  deleteTarget,
  appendSessionRow,
  listSessionRows,
  clearSessionRows,
  bulkExport,
  bulkReplace,
  newId,
} from "@/lib/soundboardDb"
import {
  convertVoice,
  pitchFormantBake as apiPitchFormantBake,
  loudnessNormalize,
  transcribeWavBlob,
} from "@/services/api"
import { exportSoundboard, importSoundboard } from "@/lib/soundboardZip"

export interface SessionContext {
  sessionId: string
  conditionContext: string
  // Human-readable counted-session number (P4). Undefined for the initial
  // empty context before any conversation has started.
  sessionNumber?: number
}

export function makeSessionContext(label: string = ""): SessionContext {
  return {
    sessionId: newId("sess"),
    conditionContext: label,
  }
}

// Cross-instance sync bus. Multiple useSoundboard() hooks (one in the
// Configure tab, one in the runtime SoundboardPanel) each hold their own
// React state; without a broadcast, mutations in one don't propagate to the
// other. Any write op calls notifyChange(), which triggers refresh() on
// every mounted instance.
const SOUNDBOARD_EVENT = "hmo-soundboard-changed"
function notifyChange() {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(SOUNDBOARD_EVENT))
  }
}

// Auto-Whisper on the given audio; returns null on failure (never throws) so
// callers can attach the transcript opportunistically without failing the
// bake/record on transcription hiccups.
async function safeTranscribe(blob: Blob): Promise<string | null> {
  try {
    const result = await transcribeWavBlob(blob)
    return (result.text || "").trim() || null
  } catch (e) {
    console.warn("[soundboard] transcript failed:", e)
    return null
  }
}

export function useSoundboard() {
  const [slots, setSlots] = useState<Slot[]>([])
  const [targets, setTargets] = useState<Target[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // ---------- bootstrap ----------
  const refresh = useCallback(async () => {
    const [s, t] = await Promise.all([listSlots(), listTargets()])
    setSlots(s)
    setTargets(t)
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        await ensureBuiltinTarget()
        if (cancelled) return
        await refresh()
    notifyChange()
      } catch (e) {
        setError((e as Error).message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [refresh])

  // Sync with any other useSoundboard() instances in the page (e.g. Configure
  // tab writes → runtime SoundboardPanel refreshes without a tab switch).
  useEffect(() => {
    const onChange = () => { void refresh() }
    window.addEventListener(SOUNDBOARD_EVENT, onChange)
    return () => window.removeEventListener(SOUNDBOARD_EVENT, onChange)
  }, [refresh])

  // ---------- slot CRUD ----------
  const createSlot = useCallback(async (label: string, condition: string): Promise<Slot> => {
    const now = Date.now()
    const slot: Slot = {
      id: newId("slot"),
      label,
      condition,
      manipulation: "unconverted",
      raw: null,
      baked: null,
      sampleRate: PP_SAMPLE_RATE,
      rawDurationMs: 0,
      bakedDurationMs: 0,
      driftMs: 0,
      createdAt: now,
      updatedAt: now,
    }
    await putSlot(slot)
    await refresh()
    notifyChange()
    return slot
  }, [refresh])

  const updateSlot = useCallback(async (slot: Slot) => {
    await putSlot(slot)
    await refresh()
    notifyChange()
  }, [refresh])

  const removeSlot = useCallback(async (id: string) => {
    await deleteSlot(id)
    await refresh()
    notifyChange()
  }, [refresh])

  // Move a slot one position up/down in the Configure list (and therefore in
  // the runtime panel, which uses the same order). Swaps the two slots' sort
  // keys — `order ?? createdAt`, on the createdAt scale — so the change is
  // consistent whether or not the slots had an explicit order yet.
  const moveSlot = useCallback(async (id: string, dir: "up" | "down") => {
    const ordered = [...slots] // already sorted by listSlots
    const i = ordered.findIndex((s) => s.id === id)
    if (i < 0) return
    const j = dir === "up" ? i - 1 : i + 1
    if (j < 0 || j >= ordered.length) return
    const a = ordered[i]
    const b = ordered[j]
    const aKey = a.order ?? a.createdAt
    const bKey = b.order ?? b.createdAt
    await putSlot({ ...a, order: bKey })
    await putSlot({ ...b, order: aKey })
    await refresh()
    notifyChange()
  }, [slots, refresh])

  // ---------- target CRUD ----------
  const addUploadedTarget = useCallback(async (label: string, file: File): Promise<Target> => {
    // Conform the upload to PP's expected format BEFORE storing. We don't
    // want the bake path to be the first thing that notices SR mismatches.
    const conformed = await decodeAndConformToPp(file)
    const wav = encodeWavAtPpRate(conformed.pcm, PP_SAMPLE_RATE)
    const target: Target = {
      id: newId("tgt"),
      label,
      builtin: false,
      wav,
      sampleRate: PP_SAMPLE_RATE,
      createdAt: Date.now(),
    }
    await putTarget(target)
    await refresh()
    notifyChange()
    return target
  }, [refresh])

  const removeTarget = useCallback(async (id: string) => {
    if (id === DEFAULT_TARGET.id) {
      throw new Error("Cannot delete the built-in default target.")
    }
    await deleteTarget(id)
    await refresh()
    notifyChange()
  }, [refresh])

  // ---------- recording ----------
  // Short-lived getUserMedia for one slot recording. Independent from the
  // conversation mic stream; we acquire and release a fresh stream each time
  // so recording can't interfere with an active conversation.
  const recordingRef = useRef<{
    stream: MediaStream
    ctx: AudioContext
    processor: ScriptProcessorNode
    source: MediaStreamAudioSourceNode
    chunks: Float32Array[]
    startedAt: number
    actualSr: number
  } | null>(null)

  const startRecording = useCallback(async (): Promise<void> => {
    if (recordingRef.current) throw new Error("Recording already in progress.")
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: PP_SAMPLE_RATE,
        channelCount: PP_CHANNELS,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })
    // AudioContext at PP_SAMPLE_RATE — getUserMedia honors the constraint as
    // a hint, but if the device's true rate differs we let the context
    // resample once at the device-boundary (this is the ONLY allowed
    // boundary resample on the record path).
    const ctx = new AudioContext({ sampleRate: PP_SAMPLE_RATE })
    const source = ctx.createMediaStreamSource(stream)
    const processor = ctx.createScriptProcessor(2048, PP_CHANNELS, 1)
    const chunks: Float32Array[] = []
    processor.onaudioprocess = (ev) => {
      // Buffer is reused; copy.
      const ch = ev.inputBuffer.getChannelData(0)
      chunks.push(new Float32Array(ch))
    }
    source.connect(processor)
    // Need a sink for the ScriptProcessor to fire — silent gain.
    const sink = ctx.createGain()
    sink.gain.value = 0
    processor.connect(sink)
    sink.connect(ctx.destination)

    recordingRef.current = {
      stream, ctx, processor, source, chunks,
      startedAt: performance.now(),
      actualSr: ctx.sampleRate,
    }
  }, [])

  const stopRecording = useCallback(async (slotId: string): Promise<void> => {
    const r = recordingRef.current
    if (!r) throw new Error("No recording in progress.")
    recordingRef.current = null

    r.processor.disconnect()
    r.source.disconnect()
    r.stream.getTracks().forEach((t) => t.stop())

    const total = r.chunks.reduce((n, c) => n + c.length, 0)
    const combined = new Float32Array(total)
    let off = 0
    for (const c of r.chunks) { combined.set(c, off); off += c.length }

    const durationMs = (total / r.actualSr) * 1000
    if (durationMs < MIN_RECORDING_SECONDS * 1000) {
      await r.ctx.close()
      throw new Error(`Recording too short (${(durationMs/1000).toFixed(2)}s, minimum ${MIN_RECORDING_SECONDS}s).`)
    }
    if (durationMs > MAX_RECORDING_SECONDS * 1000) {
      // Truncate rather than error.
      combined.fill(0, Math.floor(MAX_RECORDING_SECONDS * r.actualSr))
    }

    // Build the canonical WAV at PP_SAMPLE_RATE.
    let pcm = combined
    let finalSr = r.actualSr
    if (r.actualSr !== PP_SAMPLE_RATE) {
      // The AudioContext should have given us PP_SAMPLE_RATE; if not, this
      // path is the explicit resample we allow at the device boundary.
      const { resampleTo } = await import("@/lib/audioFormat")
      pcm = await resampleTo(combined, r.actualSr, PP_SAMPLE_RATE)
      finalSr = PP_SAMPLE_RATE
    }
    const wav = encodeWavAtPpRate(pcm, finalSr)
    await r.ctx.close()

    // Patch the slot with the new raw take. Clears any old baked audio —
    // a re-record invalidates the bake by definition.
    const existing = (await listSlots()).find((s) => s.id === slotId)
    if (!existing) throw new Error(`Slot ${slotId} not found.`)
    // Auto-transcribe the fresh take so slot.transcript is populated for the
    // runtime panel. Non-blocking would be nicer UX but the returned Slot is
    // stored immediately; we just update once transcription lands.
    const transcript = await safeTranscribe(wav)
    const patched: Slot = {
      ...existing,
      raw: wav,
      baked: null,
      sampleRate: PP_SAMPLE_RATE,
      rawDurationMs: (pcm.length / PP_SAMPLE_RATE) * 1000,
      bakedDurationMs: 0,
      driftMs: 0,
      bakeTimestamp: undefined,
      qualityScore: null,
      transcript,
    }
    await putSlot(patched)
    await refresh()
    notifyChange()
  }, [refresh])

  const abortRecording = useCallback(async () => {
    const r = recordingRef.current
    if (!r) return
    recordingRef.current = null
    r.processor.disconnect()
    r.source.disconnect()
    r.stream.getTracks().forEach((t) => t.stop())
    await r.ctx.close()
  }, [])

  // Alternative to recording: ingest a local WAV (or any browser-decodable
  // audio file) and store it as the slot's raw take. Conforms to PP's
  // sample rate / mono via decodeAndConformToPp + encodeWavAtPpRate — the
  // same path as target uploads, so format integrity holds.
  const loadRawFromFile = useCallback(async (slotId: string, file: File) => {
    const conformed = await decodeAndConformToPp(file)
    const wav = encodeWavAtPpRate(conformed.pcm, PP_SAMPLE_RATE)
    const existing = (await listSlots()).find((s) => s.id === slotId)
    if (!existing) throw new Error(`Slot ${slotId} not found.`)
    const transcript = await safeTranscribe(wav)
    const patched: Slot = {
      ...existing,
      raw: wav,
      baked: null,            // re-source invalidates any prior bake
      sampleRate: PP_SAMPLE_RATE,
      rawDurationMs: conformed.durationMs,
      bakedDurationMs: 0,
      driftMs: 0,
      bakeTimestamp: undefined,
      qualityScore: null,
      transcript,
    }
    await putSlot(patched)
    await refresh()
    notifyChange()
  }, [refresh])

  // Download a slot's raw or baked WAV. Mostly convenience for spot-checks
  // outside the app — bulk export via exportZip is what you use to move a
  // whole stimulus set.
  const downloadSlotAudio = useCallback(
    (slot: Slot, which: "raw" | "baked") => {
      const blob = which === "raw" ? slot.raw : slot.baked
      if (!blob) return
      const safe = slot.label.replace(/[^\w\-]+/g, "_").slice(0, 40) || slot.id
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${safe}_${which}.wav`
      a.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    },
    [],
  )

  // ---------- bake ----------
  // VC mode: send raw + target WAVs to /api/voice-conversion, conform the
  // returned WAV to PP_SAMPLE_RATE, store as slot.baked.
  // Pitch/formant: send raw to /api/pitch-formant; server preserves SR and
  // duration. The response is already at PP_SAMPLE_RATE.
  const bakeSlot = useCallback(
    async (slotId: string, opts: {
      mode: ManipulationMode
      targetId?: string
      pitchSemitones?: number
      formantShift?: number
      engine?: "meanvc" | "xvc"
      // P3 loudness normalization: EBU-R128 gain-match the FINAL clip to a
      // common target so conditions don't differ in playback level.
      normalize?: boolean
      targetLufs?: number
    }) => {
      const slot = (await listSlots()).find((s) => s.id === slotId)
      if (!slot) throw new Error(`Slot ${slotId} not found.`)
      if (!slot.raw) throw new Error(`Slot ${slot.label} has no raw recording.`)

      let baked: Blob | null = null
      const updates: Partial<Slot> = {
        manipulation: opts.mode,
        targetId: opts.targetId,
        engine: opts.engine,
        pitchSemitones: opts.pitchSemitones ?? PITCH_FORMANT_DEFAULTS.semitones,
        formantShift: opts.formantShift ?? PITCH_FORMANT_DEFAULTS.formantShift,
      }

      if (opts.mode === "unconverted") {
        // No transform — soundboard plays slot.raw directly.
        baked = null
        updates.bakedDurationMs = slot.rawDurationMs
        updates.driftMs = 0
      } else if (opts.mode === "vc") {
        if (!opts.targetId) throw new Error("VC bake requires a targetId.")
        const target = (await listTargets()).find((t) => t.id === opts.targetId)
        if (!target) throw new Error(`Target ${opts.targetId} not found.`)
        // Built-in targets have no WAV blob here; the live convertVoice
        // endpoint expects a File for both source and target. For the
        // built-in default we synthesise a minimal silent stub File and rely
        // on the server's default target (PP's voice_prompt). If you add
        // server-side target-id resolution later, swap this in.
        if (!target.wav && target.builtin) {
          throw new Error(
            "Built-in target VC bake requires a target WAV. Upload a target " +
              "for now (or extend /api/voice-conversion to accept a target_id).",
          )
        }
        const rawFile = new File([slot.raw], "raw.wav", { type: "audio/wav" })
        const targetFile = new File([target.wav!], "target.wav", { type: "audio/wav" })
        const vcWav = await convertVoice(rawFile, targetFile)
        // Seed-VC output SR is not guaranteed to match PP_SAMPLE_RATE — we
        // conform here explicitly. This is allowed because it's at the bake
        // *output* boundary and is the only resample the WAV undergoes.
        const conformed = await decodeAndConformToPp(vcWav)
        baked = encodeWavAtPpRate(conformed.pcm, PP_SAMPLE_RATE)
        const drift = checkDuration(slot.rawDurationMs, conformed.durationMs)
        updates.bakedDurationMs = conformed.durationMs
        updates.driftMs = drift.driftMs
      } else if (opts.mode === "pitch_formant") {
        const result = await apiPitchFormantBake(
          slot.raw,
          opts.pitchSemitones ?? PITCH_FORMANT_DEFAULTS.semitones,
          opts.formantShift ?? PITCH_FORMANT_DEFAULTS.formantShift,
          PP_SAMPLE_RATE,
        )
        if (result.sampleRate !== PP_SAMPLE_RATE) {
          throw new Error(
            `pitch-formant bake returned SR ${result.sampleRate}, expected ${PP_SAMPLE_RATE}.`,
          )
        }
        baked = result.wav
        updates.bakedDurationMs = result.outMs
        updates.driftMs = result.driftMs
      }

      // Loudness normalization (P3): applied to the FINAL clip so every
      // condition lands at a common integrated loudness. For unconverted mode
      // this produces a `baked` (the level-matched raw) while `raw` stays the
      // untouched take; for VC / pitch-formant it re-levels their output.
      if (opts.normalize) {
        const source = baked ?? slot.raw
        const norm = await loudnessNormalize(
          source,
          opts.targetLufs ?? LOUDNESS_TARGET_LUFS,
          PP_SAMPLE_RATE,
        )
        if (norm.sampleRate !== PP_SAMPLE_RATE) {
          throw new Error(
            `loudness-normalize returned SR ${norm.sampleRate}, expected ${PP_SAMPLE_RATE}.`,
          )
        }
        baked = norm.wav
        // Duration is unchanged by gain; keep the mode's bakedDurationMs when
        // set, else the raw duration (unconverted path).
        updates.bakedDurationMs = updates.bakedDurationMs ?? slot.rawDurationMs
        updates.normalized = true
        updates.targetLufs = opts.targetLufs ?? LOUDNESS_TARGET_LUFS
        updates.measuredLufs = norm.outLufs
      } else {
        updates.normalized = false
        updates.measuredLufs = null
      }

      // Transcribe the FINAL clip that will be sent to PP (baked if there
      // is one, else raw for unconverted mode). Runtime panel injects this
      // text into the conversation transcript when the slot plays.
      const forTranscript = baked ?? slot.raw
      const transcript = forTranscript ? await safeTranscribe(forTranscript) : null

      const patched: Slot = {
        ...slot,
        ...updates,
        baked,
        transcript,
        bakeTimestamp: Date.now(),
      }
      await putSlot(patched)
      await refresh()
      notifyChange()
    },
    [refresh],
  )

  // ---------- preview ----------
  // Decode + play locally via AudioBufferSourceNode → destination. Does NOT
  // touch PP. Used in Configure to sanity-check a bake before committing.
  // onEnded fires when the clip finishes naturally OR when stop() is called,
  // so callers can flip a play/pause icon back to "play" without a timer.
  const previewBlob = useCallback(
    async (blob: Blob, onEnded?: () => void): Promise<{ stop: () => void }> => {
      const ctx = new AudioContext()
      const audio = await ctx.decodeAudioData(await blob.arrayBuffer())
      const src = ctx.createBufferSource()
      src.buffer = audio
      src.connect(ctx.destination)
      let finished = false
      const cleanup = () => {
        if (finished) return
        finished = true
        try { src.stop() } catch { /* already ended */ }
        try { ctx.close() } catch { /* noop */ }
        onEnded?.()
      }
      src.onended = cleanup
      src.start(0)
      return { stop: cleanup }
    },
    [],
  )

  // ---------- session log ----------
  const logPlayback = useCallback(async (
    slot: Slot,
    sessionCtx: SessionContext,
    playStartMs: number,
    playEndMs: number,
    clipDurationMs: number,
    retry: boolean = false,
  ) => {
    const row: SessionRow = {
      sessionId: sessionCtx.sessionId,
      conditionContext: sessionCtx.conditionContext,
      eventType: "slot",
      timestampMs: playStartMs,
      slotId: slot.id,
      slotLabel: slot.label,
      slotCondition: slot.condition,
      playStartMs,
      playEndMs,
      clipDurationMs,
      retry,
      timestamp: Date.now(),
    }
    await appendSessionRow(row)
  }, [])

  // Log a PersonaPlex speech-start/-end event into the same session, on the
  // same performance.now() clock as slot playback (RQ2b timing). Wired from the
  // SoundboardPanel's PP-speech listener.
  const logPpEvent = useCallback(async (
    sessionCtx: SessionContext,
    eventType: "pp_speech_start" | "pp_speech_end",
    timestampMs: number,
  ) => {
    await appendSessionRow({
      sessionId: sessionCtx.sessionId,
      conditionContext: sessionCtx.conditionContext,
      eventType,
      timestampMs,
      timestamp: Date.now(),
    })
  }, [])

  const downloadSessionLog = useCallback(async (sessionId?: string, fmt: "csv" | "json" = "csv") => {
    const rows = await listSessionRows(sessionId)
    let blob: Blob
    let filename: string
    if (fmt === "csv") {
      const num = (v?: number) => (v == null ? "" : v.toFixed(3))
      const str = (v?: string) => JSON.stringify(v ?? "")
      const header = [
        "sessionId", "conditionContext", "eventType", "timestampMs",
        "slotId", "slotLabel", "slotCondition",
        "playStartMs", "playEndMs", "clipDurationMs", "retry", "timestamp",
      ]
      const lines = [header.join(",")]
      for (const r of rows) {
        // Legacy rows predate eventType/timestampMs — treat as slot plays.
        const eventType = r.eventType ?? "slot"
        const timestampMs = r.timestampMs ?? r.playStartMs
        lines.push([
          r.sessionId,
          str(r.conditionContext),
          eventType,
          num(timestampMs),
          r.slotId ?? "",
          str(r.slotLabel),
          str(r.slotCondition),
          num(r.playStartMs),
          num(r.playEndMs),
          num(r.clipDurationMs),
          eventType === "slot" ? (r.retry ? "1" : "0") : "",
          r.timestamp,
        ].join(","))
      }
      blob = new Blob([lines.join("\n")], { type: "text/csv" })
      filename = `soundboard-session${sessionId ? `-${sessionId}` : ""}.csv`
    } else {
      blob = new Blob([JSON.stringify(rows, null, 2)], { type: "application/json" })
      filename = `soundboard-session${sessionId ? `-${sessionId}` : ""}.json`
    }
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    a.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }, [])

  const clearSession = useCallback(async (sessionId?: string) => {
    await clearSessionRows(sessionId)
  }, [])

  // ---------- export / import ----------
  const exportZip = useCallback(async (): Promise<Blob> => {
    const { slots: s, targets: t } = await bulkExport()
    return exportSoundboard(s, t)
  }, [])

  const importZip = useCallback(async (zip: Blob) => {
    const { slots: s, targets: t } = await importSoundboard(zip)
    await bulkReplace(s, t)
    await refresh()
    notifyChange()
  }, [refresh])

  return {
    slots, targets, loading, error,
    createSlot, updateSlot, removeSlot, moveSlot,
    addUploadedTarget, removeTarget,
    startRecording, stopRecording, abortRecording, loadRawFromFile,
    bakeSlot, previewBlob, downloadSlotAudio,
    logPlayback, logPpEvent, downloadSessionLog, clearSession,
    exportZip, importZip,
    refresh,
  }
}
