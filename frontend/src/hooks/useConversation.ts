import { useState, useRef, useCallback, useEffect, useMemo } from "react"
import { webmToWavBlob } from "@/lib/audio"
import {
  transcribeRecording,
  transcribeWavBlob,
  compareMetricsData,
  vcQuality,
  type MetricsResult,
  type TranscriptionResult,
  type TranscriptionSegment,
  type VcQualityMetric,
  type VcQualityResult,
} from "@/services/api"
import { mergeAudioTracks } from "@/services/audioMerge"
import { getVoiceAnalysisMode } from "@/lib/config"
import {
  hasCapture,
  getCapturedClips,
  assembleSentWav,
  assembleRawWav,
  singleVcTarget,
  resetCapture,
} from "@/lib/soundboardCapture"
import { listTargets } from "@/lib/soundboardDb"
import { formatTime } from "@/lib/utils"
import type { useWebSocket } from "@/hooks/useWebSocket"
import type { useRecorder } from "@/hooks/useRecorder"
import type { useMeanVCPipeline } from "@/hooks/useMeanVCPipeline"

type WsState = ReturnType<typeof useWebSocket>
type RecorderState = ReturnType<typeof useRecorder>
type VCState = ReturnType<typeof useMeanVCPipeline>

export interface DiarizedTurn {
  speaker: "user" | "personaplex"
  text: string
  start: number
  end: number
}

// Diarization timing heuristics. These tune how Whisper words are grouped into
// turns and how PersonaPlex transcripts (which lack precise per-word timestamps)
// are placed on the timeline. Equivalent constants live in useWebSocket.ts —
// keep them in sync if you tweak the speech-rate estimate.
const WORD_GAP_BREAK_SECONDS = 1.15        // word-to-word gap that starts a new turn
const MIN_TURN_DURATION_SECONDS = 0.2      // shortest renderable turn (prevents zero-length blips)
const MIN_PP_TURN_DURATION_SECONDS = 0.8   // shortest plausible PP utterance
const MAX_PP_TURN_DURATION_SECONDS = 12    // cap for over-estimated PP turn length
const PP_WORDS_PER_SECOND = 2.7            // typical PP speech rate, used to estimate turn duration

function textFromWords(words: { word: string }[]) {
  return words.map((w) => w.word).join(" ").replace(/\s+([,.!?;:])/g, "$1").trim()
}

function segmentToFallbackTurn(
  segment: TranscriptionSegment,
  speaker: DiarizedTurn["speaker"],
): DiarizedTurn | null {
  const text = segment.text.trim()
  if (!text) return null
  const start = Math.max(0, segment.start)
  const end = Math.max(start + MIN_TURN_DURATION_SECONDS, segment.end)
  return { speaker, text, start, end }
}

function transcriptionToTurns(
  result: TranscriptionResult,
  speaker: DiarizedTurn["speaker"],
): DiarizedTurn[] {
  const turns: DiarizedTurn[] = []
  for (const segment of result.segments || []) {
    const words = (segment.words || []).filter(
      (w) => w.word.trim() && Number.isFinite(w.start) && Number.isFinite(w.end),
    )
    if (words.length === 0) {
      const fallback = segmentToFallbackTurn(segment, speaker)
      if (fallback) turns.push(fallback)
      continue
    }

    let group: typeof words = []
    const flush = () => {
      if (group.length === 0) return
      const start = Math.max(0, group[0].start)
      const end = Math.max(start + MIN_TURN_DURATION_SECONDS, group[group.length - 1].end)
      const text = textFromWords(group)
      if (text) turns.push({ speaker, text, start, end })
      group = []
    }

    for (const word of words) {
      const prev = group[group.length - 1]
      if (prev && word.start - prev.end > WORD_GAP_BREAK_SECONDS) flush()
      group.push(word)
    }
    flush()
  }
  return turns
}

// Soundboard turns for the FINAL transcript, sourced from the capture buffer
// (not the mic recording, which is muted during playback). Prefers the real
// Whisper text; falls back to a labeled placeholder so a play is always
// visible in the transcript even if transcription failed entirely.
function soundboardTurns(): DiarizedTurn[] {
  return getCapturedClips().map((c) => ({
    speaker: "user" as const,
    text: (c.transcript && c.transcript.trim()) || `[soundboard: ${c.slotLabel}]`,
    start: Math.max(0, c.startSec),
    end: Math.max(c.startSec + MIN_TURN_DURATION_SECONDS, c.endSec),
  }))
}

function fallbackPersonaplexTurns(transcripts: WsState["transcripts"]): DiarizedTurn[] {
  let cursor = 0
  // Only PersonaPlex entries belong here — user entries (soundboard plays
  // injected via addUserTranscript) must NOT be relabeled as PP.
  return transcripts.filter((t) => t.speaker === "personaplex").map((t) => {
    const text = t.text.trim()
    const wordCount = Math.max(1, text.split(/\s+/).length)
    const estimatedDuration = Math.min(
      MAX_PP_TURN_DURATION_SECONDS,
      Math.max(MIN_PP_TURN_DURATION_SECONDS, wordCount / PP_WORDS_PER_SECOND),
    )
    const end = Math.max(t.end ?? 0, cursor + MIN_TURN_DURATION_SECONDS)
    const start = Math.max(cursor, t.start ?? Math.max(0, end - estimatedDuration))
    cursor = Math.max(end, start + MIN_TURN_DURATION_SECONDS)
    return { speaker: "personaplex" as const, text, start, end: cursor }
  }).filter((t) => t.text)
}

export function useConversation(ws: WsState, recorder: RecorderState, vcPipeline: VCState) {
  const micClicked = useRef(false)
  const transcribed = useRef(false)
  const conversationRunId = useRef(0)
  const voiceAnalysisMode = useMemo(() => getVoiceAnalysisMode(), [])

  const [textPrompt, setTextPrompt] = useState("You enjoy having a good conversation.")
  const [diarized, setDiarized] = useState<DiarizedTurn[] | null>(null)
  const [userWavUrl, setUserWavUrl] = useState<string | null>(null)
  const [personaplexWavUrl, setPersonaplexWavUrl] = useState<string | null>(null)
  const [mergedWavUrl, setMergedWavUrl] = useState<string | null>(null)
  // Voice-change metrics (VC mode only): original mic vs converted voice.
  const [originalUserWavUrl, setOriginalUserWavUrl] = useState<string | null>(null)
  const [vcMetrics, setVcMetrics] = useState<MetricsResult | null>(null)
  const [vcMetricsLoading, setVcMetricsLoading] = useState(false)
  // VC-quality (WER/SECS/UTMOS/DNSMOS/F0) — separate from vcMetrics above
  // (which is LLM-response-comparison). vcQuality is the audio-side quality.
  const [vcQualityData, setVcQualityData] = useState<VcQualityResult | null>(null)
  const [vcQualityLoading, setVcQualityLoading] = useState(false)
  // Soundboard sessions have no VC-pipeline target, but a clean single-target
  // VC-baked session still has a well-defined reference for VC-quality. When
  // that holds we resolve the target's WAV to a blob URL here so the analysis
  // can run (see singleVcTarget()).
  const [soundboardTargetUrl, setSoundboardTargetUrl] = useState<string | null>(null)
  // True between conversation end and the results being ready (drives the shimmer).
  const [processing, setProcessing] = useState(false)

  const { vcEnabled, vcTargetId, vcTargetUrl, vcStreaming, startMic, beginSending, stopVCStream: vcStop, getOriginalUserWav } = vcPipeline
  const {
    clearTranscripts,
    clearResponseChunks,
    clearError,
    connect,
    disconnect,
    getVcUserWav,
    getPersonaplexWav,
    handshakeReceived,
    setMergedOutput,
    transcripts,
  } = ws
  const {
    isRecording,
    start: startRecorder,
    stop: stopRecorder,
    recordingAvailable,
    recordedChunks,
    mergedContext,
    mergedDestination,
    getMergedChunks,
  } = recorder
  const sendingBegun = useRef(false)

  // The post-recording finalization effect must read the latest transcripts
  // WITHOUT listing `transcripts` as a dependency. startConversation calls
  // clearTranscripts(), which mutates `transcripts` and would re-fire the
  // effect while the recorder still holds the PREVIOUS session's
  // recordingAvailable/recordedChunks (they only reset when the new recorder
  // starts, after handshake). That premature run consumes the
  // transcribed.current guard, so the new session's real stop never finalizes
  // — the 2nd+ conversation ends with no transcript. A ref hands the effect
  // fresh transcripts without being a dependency.
  const transcriptsRef = useRef(transcripts)
  transcriptsRef.current = transcripts

  const startConversation = useCallback(async () => {
    conversationRunId.current += 1
    clearTranscripts()
    clearResponseChunks()
    clearError()
    micClicked.current = true
    transcribed.current = false
    sendingBegun.current = false
    setDiarized(null)
    setUserWavUrl(null)
    setPersonaplexWavUrl(null)
    setMergedWavUrl(null)
    setOriginalUserWavUrl(null)
    setVcMetrics(null)
    setVcMetricsLoading(false)
    setVcQualityData(null)
    setVcQualityLoading(false)
    setSoundboardTargetUrl(null)
    // Drop any soundboard clips captured in a previous conversation so this
    // session's finalization only sees what was actually played this time.
    resetCapture()
    setProcessing(false)
    if (vcEnabled && vcTargetId) {
      // VC mode: acquire mic first to learn the sample rate, then connect to the
      // chat-proxy (which speaks PersonaPlex's protocol on this same socket).
      try {
        const proxy = await startMic()
        connect(textPrompt, proxy)
      } catch {
        micClicked.current = false
      }
    } else {
      connect(textPrompt)
    }
  }, [clearTranscripts, clearResponseChunks, clearError, connect, textPrompt, vcEnabled, vcTargetId, startMic])

  const stopConversation = useCallback(() => {
    const runId = conversationRunId.current
    const wasVC = vcStreaming
    if (vcStreaming) vcStop()
    stopRecorder()
    disconnect()
    micClicked.current = false
    setProcessing(true)

    if (wasVC) {
      ;(async () => {
        const stepDurationsMs: Record<string, number> = {}
        const isCurrentRun = () => runId === conversationRunId.current
        const timed = async <T,>(name: string, fn: () => Promise<T> | T): Promise<T> => {
          const start = performance.now()
          try {
            return await fn()
          } finally {
            stepDurationsMs[name] = Math.round(performance.now() - start)
          }
        }

        try {
          const vcWav = await timed("collect_converted_user_wav", () => getVcUserWav())
          if (!isCurrentRun()) return
          if (!vcWav) { setProcessing(false); return }
          setUserWavUrl(URL.createObjectURL(vcWav))

          const originalWav = await timed("collect_original_user_wav", () => getOriginalUserWav())
          if (!isCurrentRun()) return
          if (originalWav) setOriginalUserWavUrl(URL.createObjectURL(originalWav))

          let pplxWav: Blob | null = null
          try {
            pplxWav = await timed("collect_personaplex_wav", () => getPersonaplexWav())
            if (!isCurrentRun()) return
            if (pplxWav) setPersonaplexWavUrl(URL.createObjectURL(pplxWav))
          } catch (e) {
            console.warn("PersonaPlex WAV assembly failed:", e)
          }

          if (pplxWav) {
            try {
              const merged = await timed("merge_audio", () => mergeAudioTracks(vcWav, pplxWav))
              if (!isCurrentRun()) return
              setMergedWavUrl(URL.createObjectURL(merged))
            } catch {
              if (isCurrentRun()) setMergedWavUrl(URL.createObjectURL(vcWav))
            }
          } else {
            setMergedWavUrl(URL.createObjectURL(vcWav))
          }

          const pplxTurns = fallbackPersonaplexTurns(transcripts)

          let vcTurns: DiarizedTurn[] = []
          try {
            const result = await timed("transcribe_converted_user_wav", () => transcribeWavBlob(vcWav))
            vcTurns = transcriptionToTurns(result, "user")
          } catch (e) {
            console.error("Converted-voice transcription failed:", e)
          }
          if (!isCurrentRun()) return
          setDiarized([...vcTurns, ...pplxTurns].sort((a, b) => a.start - b.start))

          const voiceMetricsAutoRan = voiceAnalysisMode === "after_vc" && !!originalWav
          console.info("[conversation reset]", {
            runId,
            resetType: "fresh_websocket_session",
            voiceAnalysisMode,
            voiceMetricsAutoRan,
            stepDurationsMs,
          })

          if (voiceMetricsAutoRan && originalWav) {
            setVcMetricsLoading(true)
            const started = performance.now()
            compareMetricsData(originalWav, vcWav)
              .then((data) => {
                if (isCurrentRun()) setVcMetrics(data)
              })
              .catch(() => {
                if (isCurrentRun()) setVcMetrics(null)
              })
              .finally(() => {
                if (isCurrentRun()) {
                  console.info("[voice metrics]", {
                    runId,
                    mode: voiceAnalysisMode,
                    durationMs: Math.round(performance.now() - started),
                  })
                  setVcMetricsLoading(false)
                }
              })
          }
        } catch (e) {
          console.error("VC finalization failed:", e)
          if (isCurrentRun()) setProcessing(false)
        }
      })()
    }
  }, [
    vcStreaming,
    vcStop,
    stopRecorder,
    disconnect,
    getVcUserWav,
    getOriginalUserWav,
    getPersonaplexWav,
    transcripts,
    voiceAnalysisMode,
  ])

  // VC mode: once the proxy relays PersonaPlex's handshake, open the gate so mic
  // PCM starts flowing. (Mic was already acquired in startConversation.)
  useEffect(() => {
    if (handshakeReceived && micClicked.current && vcStreaming && !sendingBegun.current) {
      sendingBegun.current = true
      beginSending()
    }
  }, [handshakeReceived, vcStreaming, beginSending])

  // Non-VC mode: start recording after handshake
  useEffect(() => {
    if (handshakeReceived && micClicked.current && !isRecording && !vcStreaming) {
      startRecorder().catch(() => {
        disconnect()
        micClicked.current = false
      })
    }
  }, [handshakeReceived, isRecording, vcStreaming, startRecorder, disconnect])

  // Route PersonaPlex audio into merged capture
  useEffect(() => {
    if (isRecording && mergedContext && mergedDestination) {
      setMergedOutput(mergedContext, mergedDestination)
    }
  }, [isRecording, mergedContext, mergedDestination, setMergedOutput])

  // Post-recording transcription (non-VC path)
  useEffect(() => {
    if (!recordingAvailable || recordedChunks.length === 0 || transcribed.current) return
    transcribed.current = true
    const runId = conversationRunId.current
    const isCurrentRun = () => runId === conversationRunId.current
    ;(async () => {
      try {
        // Mic transcription + WAV are BEST-EFFORT. A soundboard session records
        // a muted (silent) mic, and transcribing/decoding that can throw or come
        // back empty. That must NOT abort building the final transcript from the
        // soundboard capture + PersonaPlex turns — otherwise a soundboard call
        // ends with no transcript and no download bar at all.
        let userSegments: DiarizedTurn[] = []
        try {
          const result = await transcribeRecording(recordedChunks)
          userSegments = transcriptionToTurns(result, "user")
        } catch (e) {
          console.warn("Mic transcription failed (expected for soundboard-only calls):", e)
        }
        try {
          const userWav = await webmToWavBlob(recordedChunks)
          if (isCurrentRun()) setUserWavUrl(URL.createObjectURL(userWav))
        } catch (e) {
          console.warn("Mic WAV assembly failed (expected for soundboard-only calls):", e)
        }
        if (!isCurrentRun()) return

        let pplxWav: Blob | null = null
        try {
          pplxWav = await getPersonaplexWav()
          if (isCurrentRun() && pplxWav) setPersonaplexWavUrl(URL.createObjectURL(pplxWav))
        } catch (e) {
          console.warn("PersonaPlex WAV assembly failed:", e)
        }
        if (!isCurrentRun()) return

        const pplxTurns = fallbackPersonaplexTurns(transcriptsRef.current)

        // Soundboard turns come from the capture buffer, not the (muted) mic.
        // Each carries its own start/end (conversation-relative seconds), so the
        // diarized view + transcript download keep their timestamps.
        const sbTurns = hasCapture() ? soundboardTurns() : []
        const diarizedResult = [...userSegments, ...sbTurns, ...pplxTurns].sort((a, b) => a.start - b.start)
        setDiarized(diarizedResult)

        // If the soundboard drove this conversation, the mic recording is
        // silence — the audio PP actually heard from the "user" is the clips
        // we sent. Override the user WAV with the assembled sent audio, expose
        // the raw takes as the VC-quality source, and resolve the VC target so
        // the "Analyze VC quality" option can appear (clean single-target VC
        // sessions only; see singleVcTarget).
        if (hasCapture()) {
          try {
            const sentWav = await assembleSentWav()
            if (isCurrentRun() && sentWav) {
              setUserWavUrl(URL.createObjectURL(sentWav))
              // "conversation.wav" / inline player should be PP + soundboard,
              // not PP + silent mic. Merge the sent audio with PP's response.
              if (pplxWav) {
                try {
                  const merged = await mergeAudioTracks(sentWav, pplxWav)
                  if (isCurrentRun()) setMergedWavUrl(URL.createObjectURL(merged))
                } catch { /* keep the mic-merged track as a fallback */ }
              }
            }
            const rawWav = await assembleRawWav()
            if (isCurrentRun() && rawWav) setOriginalUserWavUrl(URL.createObjectURL(rawWav))
            const tgtId = singleVcTarget()
            if (tgtId) {
              const tgt = (await listTargets()).find((t) => t.id === tgtId)
              if (isCurrentRun() && tgt?.wav) {
                setSoundboardTargetUrl(URL.createObjectURL(tgt.wav))
              }
            }
          } catch (e) {
            console.warn("[soundboard] finalize assembly failed:", e)
          }
        }

        // The mic-merged track (PP + physical mic) is only meaningful when the
        // researcher spoke live. In a soundboard session the mic is silence and
        // we've already built a PP + soundboard merge above, so don't clobber it.
        const mergedChunks = getMergedChunks()
        if (!hasCapture() && mergedChunks.length > 0) {
          try {
            const merged = await webmToWavBlob(mergedChunks)
            if (isCurrentRun()) setMergedWavUrl(URL.createObjectURL(merged))
          } catch (e) { console.error("Merged audio conversion failed:", e) }
        }
      } catch (err) {
        console.error("Transcription failed:", err)
        if (isCurrentRun()) setProcessing(false)
      }
    })()
    // `transcripts` intentionally omitted — read via transcriptsRef so
    // clearTranscripts() on the next session start doesn't re-fire this effect
    // against the previous session's stale recorder state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recordingAvailable, recordedChunks, getPersonaplexWav, getMergedChunks])

  // Results are ready once diarization lands → drop the shimmer.
  useEffect(() => {
    if (diarized !== null) setProcessing(false)
  }, [diarized])

  const downloadTranscript = useCallback(() => {
    if (!diarized) return
    const lines = diarized.map(
      (t) => `[${formatTime(t.start)}-${formatTime(t.end)}] ${t.speaker === "user" ? "You" : "PersonaPlex"}: ${t.text}`
    )
    const blob = new Blob([lines.join("\n")], { type: "text/plain" })
    const a = document.createElement("a")
    a.href = URL.createObjectURL(blob)
    a.download = "transcript.txt"
    a.click()
  }, [diarized])

  const downloadVcQuality = useCallback(() => {
    if (!vcQualityData) return
    const blob = new Blob([JSON.stringify(vcQualityData, null, 2)],
                          { type: "application/json" })
    const a = document.createElement("a")
    a.href = URL.createObjectURL(blob)
    a.download = `vc_quality_${new Date().toISOString().replace(/[:.]/g, "-")}.json`
    a.click()
  }, [vcQualityData])

  // Manual triggers. The post-conversation flow stashes the original/converted
  // user voice (and target file) as blob URLs; these handlers pull the bytes
  // back on demand and run the heavy backend analysis only when asked.
  const triggerVcMetrics = useCallback(async () => {
    if (voiceAnalysisMode === "off") return
    if (!originalUserWavUrl || !userWavUrl) return
    if (vcMetrics || vcMetricsLoading) return
    const runId = conversationRunId.current
    setVcMetricsLoading(true)
    try {
      const [orig, conv] = await Promise.all([
        fetch(originalUserWavUrl).then(r => r.blob()),
        fetch(userWavUrl).then(r => r.blob()),
      ])
      const data = await compareMetricsData(orig, conv)
      if (runId === conversationRunId.current) setVcMetrics(data)
    } catch {
      if (runId === conversationRunId.current) setVcMetrics(null)
    } finally {
      if (runId === conversationRunId.current) setVcMetricsLoading(false)
    }
  }, [originalUserWavUrl, userWavUrl, vcMetrics, vcMetricsLoading, voiceAnalysisMode])

  // In a soundboard session there's no VC-pipeline target, but a clean
  // single-target VC bake resolves one into soundboardTargetUrl. Use whichever
  // is present.
  const effectiveVcTargetUrl = vcTargetUrl ?? soundboardTargetUrl

  const triggerVcQuality = useCallback(async (skipMetrics?: VcQualityMetric[]) => {
    if (voiceAnalysisMode === "off") return
    if (!originalUserWavUrl || !userWavUrl || !effectiveVcTargetUrl) return
    if (vcQualityData || vcQualityLoading) return
    const runId = conversationRunId.current
    setVcQualityLoading(true)
    try {
      const [orig, conv, tgt] = await Promise.all([
        fetch(originalUserWavUrl).then(r => r.blob()),
        fetch(userWavUrl).then(r => r.blob()),
        fetch(effectiveVcTargetUrl).then(r => r.blob()),
      ])
      const data = await vcQuality(orig, tgt, conv, {
        segmentMode: "fixed",
        // 5s/5s windows give ~5x fewer WavLM forward passes than the 2s/1s
        // CLI default — essential on CPU. Coarser anomaly resolution; still
        // localizes drift to a 5-second region.
        segmentWin: 5,
        segmentHop: 5,
        skipMetrics,
      })
      if (runId === conversationRunId.current) setVcQualityData(data)
    } catch {
      if (runId === conversationRunId.current) setVcQualityData(null)
    } finally {
      if (runId === conversationRunId.current) setVcQualityLoading(false)
    }
  }, [originalUserWavUrl, userWavUrl, effectiveVcTargetUrl, vcQualityData, vcQualityLoading, voiceAnalysisMode])

  return {
    textPrompt,
    setTextPrompt,
    diarized,
    userWavUrl,
    personaplexWavUrl,
    mergedWavUrl,
    originalUserWavUrl,
    vcMetrics,
    vcMetricsLoading,
    vcQuality: vcQualityData,
    vcQualityLoading,
    downloadVcQuality,
    triggerVcMetrics,
    triggerVcQuality,
    canTriggerVcMetrics: voiceAnalysisMode !== "off" && !!(originalUserWavUrl && userWavUrl),
    canTriggerVcQuality: voiceAnalysisMode !== "off" && !!(originalUserWavUrl && userWavUrl && effectiveVcTargetUrl),
    processing,
    startConversation,
    stopConversation,
    downloadTranscript,
  }
}
