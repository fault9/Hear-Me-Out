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

function fallbackPersonaplexTurns(transcripts: WsState["transcripts"]): DiarizedTurn[] {
  let cursor = 0
  return transcripts.map((t) => {
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
  // True between conversation end and the results being ready (drives the shimmer).
  const [processing, setProcessing] = useState(false)

  const { vcTargetId, vcTargetUrl, vcStreaming, startMic, beginSending, stopVCStream: vcStop, getOriginalUserWav } = vcPipeline
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
    setProcessing(false)
    if (vcTargetId) {
      // A target exists → always route through the chat-proxy so VC can be
      // toggled live mid-conversation (the proxy converts or passes through
      // per the vc_control channel). This holds even if VC is currently off:
      // startMic seeds the proxy with the toggle's initial state.
      try {
        const proxy = await startMic()
        connect(textPrompt, proxy)
      } catch {
        micClicked.current = false
      }
    } else {
      // No target uploaded → direct PersonaPlex connection, no VC available.
      connect(textPrompt)
    }
  }, [clearTranscripts, clearResponseChunks, clearError, connect, textPrompt, vcTargetId, startMic])

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

          const originalWav = await timed("collect_original_user_wav", () => getOriginalUserWav())
          if (!isCurrentRun()) return
          if (originalWav) setOriginalUserWavUrl(URL.createObjectURL(originalWav))

          // The proxy only emits converted (0x03) audio while VC is ON. A
          // conversation that stayed in passthrough the whole time has none, so
          // fall back to the raw mic capture for the user track/downloads.
          const userWav = vcWav ?? originalWav
          if (!userWav) { setProcessing(false); return }
          setUserWavUrl(URL.createObjectURL(userWav))

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
              const merged = await timed("merge_audio", () => mergeAudioTracks(userWav, pplxWav))
              if (!isCurrentRun()) return
              setMergedWavUrl(URL.createObjectURL(merged))
            } catch {
              if (isCurrentRun()) setMergedWavUrl(URL.createObjectURL(userWav))
            }
          } else {
            setMergedWavUrl(URL.createObjectURL(userWav))
          }

          const pplxTurns = fallbackPersonaplexTurns(transcripts)

          let vcTurns: DiarizedTurn[] = []
          try {
            const result = await timed("transcribe_converted_user_wav", () => transcribeWavBlob(userWav))
            vcTurns = transcriptionToTurns(result, "user")
          } catch (e) {
            console.error("Converted-voice transcription failed:", e)
          }
          if (!isCurrentRun()) return
          setDiarized([...vcTurns, ...pplxTurns].sort((a, b) => a.start - b.start))

          // Auto voice-change metrics compare original vs converted, so only run
          // them when actual converted audio exists (skip pure-passthrough runs).
          const voiceMetricsAutoRan = voiceAnalysisMode === "after_vc" && !!originalWav && !!vcWav
          console.info("[conversation reset]", {
            runId,
            resetType: "fresh_websocket_session",
            voiceAnalysisMode,
            voiceMetricsAutoRan,
            stepDurationsMs,
          })

          if (voiceMetricsAutoRan && originalWav && vcWav) {
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
        const result = await transcribeRecording(recordedChunks)
        const userSegments = transcriptionToTurns(result, "user")
        const userWav = await webmToWavBlob(recordedChunks)
        if (!isCurrentRun()) return
        setUserWavUrl(URL.createObjectURL(userWav))

        const pplxWav = await getPersonaplexWav()
        if (!isCurrentRun()) return
        if (pplxWav) setPersonaplexWavUrl(URL.createObjectURL(pplxWav))

        const pplxTurns = fallbackPersonaplexTurns(transcripts)
        if (!isCurrentRun()) return

        const diarizedResult = [...userSegments, ...pplxTurns].sort((a, b) => a.start - b.start)
        setDiarized(diarizedResult)

        const mergedChunks = getMergedChunks()
        if (mergedChunks.length > 0) {
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
  }, [recordingAvailable, recordedChunks, getPersonaplexWav, transcripts, getMergedChunks])

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

  const triggerVcQuality = useCallback(async (skipMetrics?: VcQualityMetric[]) => {
    if (voiceAnalysisMode === "off") return
    if (!originalUserWavUrl || !userWavUrl || !vcTargetUrl) return
    if (vcQualityData || vcQualityLoading) return
    const runId = conversationRunId.current
    setVcQualityLoading(true)
    try {
      const [orig, conv, tgt] = await Promise.all([
        fetch(originalUserWavUrl).then(r => r.blob()),
        fetch(userWavUrl).then(r => r.blob()),
        fetch(vcTargetUrl).then(r => r.blob()),
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
  }, [originalUserWavUrl, userWavUrl, vcTargetUrl, vcQualityData, vcQualityLoading, voiceAnalysisMode])

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
    canTriggerVcQuality: voiceAnalysisMode !== "off" && !!(originalUserWavUrl && userWavUrl && vcTargetUrl),
    processing,
    startConversation,
    stopConversation,
    downloadTranscript,
  }
}
