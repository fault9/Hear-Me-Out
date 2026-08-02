// ============================================================================
//  SOUNDBOARD PLAYBACK — direct-to-PP
// ----------------------------------------------------------------------------
//  When a researcher clicks a slot button during a conversation, this hook
//  feeds the slot's baked WAV to PP via the SAME Opus pipeline live mic uses.
//
//  ARCHITECTURE — why this design:
//    PP's WebSocket expects Ogg-Opus packets, NOT raw PCM. The live-mic path
//    (useRecorder.ts) wraps the mic stream in opus-recorder configured at
//    PP_SAMPLE_RATE / 80ms frames / 1 frame per page. We do exactly the same
//    thing here, but with an AudioBufferSourceNode driving the encoder
//    instead of getUserMedia. Net effect: the Opus stream the soundboard
//    emits is what live mic input WOULD emit if you spoke the baked WAV's
//    content. PP cannot distinguish the two.
//
//  WHAT "BYTE FIDELITY" MEANS HERE:
//    The spec says "the byte baked must equal the byte PP receives." Strictly
//    that would require raw-PCM wire framing, but PP expects Opus. The
//    practical, useful version of the property:
//
//      Baked WAV is the canonical artifact. Same WAV in →
//      same Opus stream out (Opus is deterministic at fixed settings) →
//      same bytes to PP.
//
//    So bakes are reproducible, exports are reproducible, and the only
//    indeterminism is the Opus encoder itself (which is deterministic given
//    the same input + config). The bake's WAV is what you'd inspect / hash /
//    diff for reproducibility, not the Opus stream.
//
//  CRITICAL — no silent resampling:
//    AudioContext is created at PP_SAMPLE_RATE so decodeAudioData → source →
//    opus-encoder never crosses a sample-rate boundary. If the OS refuses
//    PP_SAMPLE_RATE on this platform, we surface the error rather than
//    falling back to an implicit-resample mode that would invalidate the
//    timing-sensitive experiment.
//
//  ROUTING:
//    Output goes through ws.sendAudio (in useWebSocket.ts) which prepends
//    the 0x01 tag byte and forwards untouched. That function makes NO
//    resampling / gain / format-conversion changes — it is the byte-fidelity
//    contract on the wire side. See the comments in useWebSocket.ts:sendAudio.
//    This hook never uses sendRawAudio (the proxy/VC path). When the
//    soundboard is in use the conversation must be opened in non-VC mode so
//    that pre-baked clips are not re-converted by the live MeanVC/X-VC stack.
// ============================================================================

import { useCallback, useEffect, useRef, useState } from "react"
import { OPUS_ENCODER_CONFIG, PP_SAMPLE_RATE } from "@/lib/soundboardConfig"
import type { Slot } from "@/lib/soundboardDb"
import type { useWebSocket } from "@shared/hooks/useWebSocket"
import { transcribeWavBlob } from "@shared/services/api"
import { putSlot } from "@/lib/soundboardDb"
import { captureClip } from "@/lib/soundboardCapture"

// opus-recorder's Recorder is loaded globally via <script> tag in index.html.
// Mirror the type already declared in useRecorder.ts but add the sourceNode
// option we rely on here.
declare class Recorder {
  constructor(opts: Record<string, unknown>)
  start(): Promise<void>
  stop(): Promise<void> | void
  ondataavailable: ((buf: ArrayBuffer) => void) | null
}

export interface PlaybackRecord {
  slotId: string
  startMs: number          // performance.now() at src.start()
  endMs: number            // performance.now() at src.onended / stop
  clipDurationMs: number   // canonical, from baked AudioBuffer
}

interface UseSoundboardPlaybackOpts {
  ws: ReturnType<typeof useWebSocket>
  onPlayStart?: (slot: Slot, startMs: number) => void
  onPlayEnd?: (slot: Slot, rec: PlaybackRecord) => void
}

export function useSoundboardPlayback(opts: UseSoundboardPlaybackOpts) {
  const { ws, onPlayStart, onPlayEnd } = opts
  const [playingSlotId, setPlayingSlotId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const ctxRef = useRef<AudioContext | null>(null)
  const srcRef = useRef<AudioBufferSourceNode | null>(null)
  const recRef = useRef<Recorder | null>(null)
  const activeSlotRef = useRef<Slot | null>(null)
  const startedAtRef = useRef<number>(0)
  const sourceEndedAtRef = useRef<number>(0)
  const clipDurationRef = useRef<number>(0)

  // Local monitor: whether the researcher hears the clip through their own
  // speakers as it plays to PP. This is purely a local tap — muting it does
  // NOT change what PP receives (the Opus stream is unaffected). Ref mirrors
  // the state so a toggle mid-clip updates the live gain node immediately.
  const [monitorMuted, setMonitorMutedState] = useState(false)
  const monitorMutedRef = useRef(false)
  const monitorGainRef = useRef<GainNode | null>(null)
  const setMonitorMuted = useCallback((muted: boolean) => {
    monitorMutedRef.current = muted
    setMonitorMutedState(muted)
    if (monitorGainRef.current) monitorGainRef.current.gain.value = muted ? 0 : 1
  }, [])

  const teardown = useCallback(async () => {
    const rec = recRef.current
    const src = srcRef.current
    const ctx = ctxRef.current
    recRef.current = null
    srcRef.current = null
    ctxRef.current = null
    monitorGainRef.current = null
    try {
      src?.stop()
    } catch { /* already stopped */ }
    try { src?.disconnect() } catch { /* noop */ }
    try { await rec?.stop() } catch { /* noop */ }
    try { await ctx?.close() } catch { /* noop */ }
  }, [])

  const stop = useCallback(async () => {
    const slot = activeSlotRef.current
    await teardown()
    // Un-suppress the live mic so normal conversation can resume.
    ws.setMicMuted(false)
    if (slot) {
      // The encoder receives a short flush grace period after the source ends.
      // Audit timing must use the audio-source boundary, not teardown time.
      const endMs = sourceEndedAtRef.current || performance.now()
      onPlayEnd?.(slot, {
        slotId: slot.id,
        startMs: startedAtRef.current,
        endMs,
        clipDurationMs: clipDurationRef.current,
      })
    }
    sourceEndedAtRef.current = 0
    activeSlotRef.current = null
    setPlayingSlotId(null)
  }, [onPlayEnd, teardown, ws])

  const playSlot = useCallback(
    async (slot: Slot) => {
      setError(null)
      // The clip we send: baked if it exists, otherwise raw (for "unconverted"
      // condition slots that play the researcher's own voice unchanged).
      const blob = slot.baked ?? slot.raw
      if (!blob) {
        setError("Slot has no audio to play.")
        return
      }
      if (!ws.connected) {
        setError("Not connected to PersonaPlex (the conversation closed — "
          + "check the browser console for a recorder error, then restart the "
          + "session).")
        return
      }
      // Socket open but PP hasn't sent its handshake yet: sending Opus now
      // would hit a server not ready to read it. Wait for warmup.
      if (!ws.warmupComplete) {
        setError("PersonaPlex is still warming up — wait for it to be ready "
          + "before playing a clip.")
        return
      }
      // Recorder is loaded from CDN; if the script tag failed (offline),
      // surface a clear error rather than crashing.
      const RecorderCtor = (window as unknown as { Recorder?: typeof Recorder })
        .Recorder
      if (!RecorderCtor) {
        setError("Opus encoder not available (window.Recorder missing).")
        return
      }
      // If another slot is mid-play, cut it off cleanly first.
      if (playingSlotId) await stop()

      // --- AudioContext at PP_SAMPLE_RATE so no silent resampling ---
      let ctx: AudioContext
      try {
        ctx = new AudioContext({ sampleRate: PP_SAMPLE_RATE })
      } catch (e) {
        setError(
          `Browser refused AudioContext at ${PP_SAMPLE_RATE} Hz: ${(e as Error).message}. ` +
            `Soundboard playback aborted to avoid implicit resampling.`,
        )
        return
      }
      // Defensive: some browsers ignore the sampleRate hint and silently use
      // the device default. If so, abort — implicit resampling would invalidate
      // timing.
      if (ctx.sampleRate !== PP_SAMPLE_RATE) {
        await ctx.close()
        setError(
          `AudioContext rate is ${ctx.sampleRate} Hz, expected ${PP_SAMPLE_RATE}. ` +
            `Soundboard playback aborted to avoid implicit resampling.`,
        )
        return
      }

      // Decode the baked WAV into an AudioBuffer at PP_SAMPLE_RATE.
      // decodeAudioData uses the context's SR; since the baked WAV is already
      // at PP_SAMPLE_RATE this is a pass-through, not a resample.
      const arr = await blob.arrayBuffer()
      const audioBuffer = await ctx.decodeAudioData(arr.slice(0))
      const clipDurationMs = audioBuffer.duration * 1000

      const src = ctx.createBufferSource()
      src.buffer = audioBuffer
      // LIVE MONITOR: route the source to the local speakers through a gain
      // node so the researcher hears the same audio PP is receiving, in real
      // time — and can mute/unmute it with the panel toggle without affecting
      // what PP gets. This is a passive tap; the opus-recorder branch (which
      // reads the source through its ScriptProcessor) is unaffected either way.
      const monitorGain = ctx.createGain()
      monitorGain.gain.value = monitorMutedRef.current ? 0 : 1
      src.connect(monitorGain)
      monitorGain.connect(ctx.destination)
      monitorGainRef.current = monitorGain

      // Opus encoder mirrors live-mic config exactly.
      const recorder = new RecorderCtor({
        ...OPUS_ENCODER_CONFIG,
        // opus-recorder accepts an AudioNode as sourceNode; it grabs
        // sourceNode.context so we don't need to pass audioContext separately.
        sourceNode: src,
      })

      recorder.ondataavailable = (packet: ArrayBuffer) => {
        // Direct-to-PP, byte-fidelity send path. See useWebSocket.sendAudio.
        ws.sendAudio(packet)
      }

      ctxRef.current = ctx
      srcRef.current = src
      recRef.current = recorder
      activeSlotRef.current = slot
      sourceEndedAtRef.current = 0
      clipDurationRef.current = clipDurationMs

      // Suppress the live mic BEFORE starting the encoder so no mic packets
      // slip through while the soundboard's Opus stream ramps up.
      ws.setMicMuted(true)
      await recorder.start()
      // Recorder is now connected to src in opus-recorder's internal graph.
      // Start the source AFTER recorder.start() so we don't lose initial frames.
      const startMs = performance.now()
      startedAtRef.current = startMs
      setPlayingSlotId(slot.id)
      onPlayStart?.(slot, startMs)
      src.start(0)

      // Record what we're sending into the conversation-scoped capture buffer.
      // This is the ONLY faithful record of the soundboard audio PP hears (the
      // mic recorder is muted), and it's what useConversation reads at the end
      // to build the final transcript + user WAV + VC-quality inputs.
      const startSec = ws.getConversationElapsed()
      const durationSec = clipDurationMs / 1000
      const captured = captureClip({
        slotId: slot.id,
        slotLabel: slot.label,
        condition: slot.condition,
        manipulation: slot.manipulation,
        targetId: slot.targetId,
        sentBlob: blob,
        rawBlob: slot.raw ?? null,
        transcript: slot.transcript ?? null,
        startSec,
        endSec: startSec + durationSec,
      })

      // Inject slot transcript into the conversation stream so soundboard
      // turns appear alongside live-voice turns. If the slot has no cached
      // transcript (older slot baked before auto-Whisper, or transcription
      // failed at bake time), kick off Whisper NOW and inject when done —
      // it'll land in the transcript slightly late but still attributed. We
      // pin the turn to `startSec` so the live view and the final diarized
      // transcript agree on placement.
      if (slot.transcript) {
        ws.addUserTranscript(slot.transcript, durationSec, startSec)
      } else {
        void (async () => {
          try {
            const result = await transcribeWavBlob(blob)
            const text = (result.text || "").trim()
            if (!text) return
            // Fill the capture entry in place so the final transcript picks up
            // the real text instead of the slot-label placeholder.
            captured.transcript = text
            ws.addUserTranscript(text, durationSec, startSec)
            // Persist so future plays are instant. Refetching from IDB
            // avoids clobbering any concurrent edits.
            try {
              const { getSlot } = await import("@/lib/soundboardDb")
              const fresh = await getSlot(slot.id)
              if (fresh) await putSlot({ ...fresh, transcript: text })
              window.dispatchEvent(new CustomEvent("hmo-soundboard-changed"))
            } catch { /* cache write is best-effort */ }
          } catch (e) {
            console.warn("[soundboard] on-demand transcript failed:", e)
          }
        })()
      }

      // onended fires when the buffer plays through; we give the encoder a
      // tiny grace period to flush any partial frame before stopping.
      src.onended = () => {
        sourceEndedAtRef.current = performance.now()
        // Small flush window so the encoder emits the last frame's page.
        setTimeout(() => { void stop() }, 60)
      }
    },
    [onPlayStart, playingSlotId, stop, ws],
  )

  // Tear down on unmount.
  useEffect(() => () => { void teardown() }, [teardown])

  return { playingSlotId, playSlot, stop, error, monitorMuted, setMonitorMuted }
}
