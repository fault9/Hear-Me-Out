import { useCallback, useEffect, useRef, useState } from "react"
import { useWebSocket } from "@shared/hooks/useWebSocket"
import { useMeanVCPipeline } from "@shared/hooks/useMeanVCPipeline"
import { mergeAudioTracks } from "@shared/services/audioMerge"
import { getStudyChatProxyWsUrl } from "@/lib/config"

export interface SessionArtifacts {
  participant: Blob | null       // converted (or pass-through for natural) user voice
  participant_raw: Blob | null   // raw mic
  model: Blob | null             // PersonaPlex audio
  merged: Blob | null
  model_transcript: unknown      // PersonaPlex turns (cheap; server adds the rest)
}

export type CallStatus = "idle" | "connecting" | "active" | "processing" | "error"

// Drives the shared voice engine for one scenario call. The browser passes only
// an opaque session_id; the VC engine resolves the hidden prompt/target/steps.
export function useStudyConversation() {
  const ws = useWebSocket()
  const vc = useMeanVCPipeline((d) => ws.sendRawAudio(d), 8)
  const [status, setStatus] = useState<CallStatus>("idle")
  const streamingRef = useRef(false)
  const connectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearConnectTimer = () => {
    if (connectTimerRef.current) { clearTimeout(connectTimerRef.current); connectTimerRef.current = null }
  }

  // Start sending mic audio once PersonaPlex's handshake arrives via the proxy.
  useEffect(() => {
    if (ws.handshakeReceived && streamingRef.current) {
      clearConnectTimer()
      vc.beginSending()
      setStatus("active")
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ws.handshakeReceived])

  // Surface a connection error (proxy sent an error, or the socket dropped)
  // instead of spinning forever on "connecting".
  useEffect(() => {
    if (ws.error && streamingRef.current) {
      clearConnectTimer()
      setStatus("error")
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ws.error])

  const start = useCallback(async (sessionId: string) => {
    setStatus("connecting")
    ws.clearTranscripts()
    streamingRef.current = true
    try {
      const proxy = await vc.startMic("session") // dummy target; proxy uses session_id
      ws.connect(getStudyChatProxyWsUrl(sessionId, proxy.sourceSr))
      // If no handshake arrives in time, fail visibly (recoverable via Try again).
      clearConnectTimer()
      connectTimerRef.current = setTimeout(() => {
        setStatus((s) => (s === "active" ? s : "error"))
      }, 30000)
    } catch (e) {
      streamingRef.current = false
      setStatus("error")
      throw e
    }
  }, [ws, vc])

  const stopAndAssemble = useCallback(async (): Promise<SessionArtifacts> => {
    streamingRef.current = false
    clearConnectTimer()
    setStatus("processing")
    vc.stopVCStream()
    ws.disconnect()

    const vcWav = ws.getVcUserWav()
    const rawWav = vc.getOriginalUserWav()
    const modelWav = await ws.getPersonaplexWav()

    // Merge is cheap local audio processing; transcription + VC metrics are done
    // on the server in the background so the participant doesn't wait.
    let merged: Blob | null = null
    if (vcWav && modelWav) {
      try { merged = await mergeAudioTracks(vcWav, modelWav) } catch { merged = null }
    }

    setStatus("idle")
    return {
      participant: vcWav, participant_raw: rawWav, model: modelWav, merged,
      model_transcript: ws.transcripts,
    }
  }, [ws, vc])

  // Tear down without assembling artifacts (used by the audio check).
  const teardown = useCallback(() => {
    streamingRef.current = false
    clearConnectTimer()
    vc.stopVCStream()
    ws.disconnect()
    setStatus("idle")
  }, [ws, vc])

  return {
    status,
    connected: ws.connected,
    warmupComplete: ws.warmupComplete,
    handshakeReceived: ws.handshakeReceived,
    error: ws.error,
    amplitude: vc.amplitude,
    modelAudioReceived: ws.responseChunks.length > 0,
    start,
    stopAndAssemble,
    teardown,
  }
}
