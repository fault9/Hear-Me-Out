import { useState, useRef, useCallback } from "react";
import { createWavFile } from "@shared/lib/audio";
import type { ProxyDescriptor } from "@shared/hooks/useWebSocket";

// App-specific configuration injected by the consumer so this hook stays shared:
// - loadTargetUrl: where uploadTarget() POSTs a target WAV (HMO only; the study
//   app preloads targets server-side and uses setPresetTarget instead).
// - voicePrompt: default voice prompt embedded in the proxy descriptor (HMO uses
//   a fixed prompt; the study app builds its own session-scoped URL and ignores it).
export interface MeanVCPipelineOptions {
  loadTargetUrl?: () => string;
  voicePrompt?: string;
}

export interface MeanVCPipelineState {
  vcEnabled: boolean;
  vcTargetId: string | null;
  vcTargetFile: string | null;
  vcTargetUrl: string | null;
  vcStatus: string;
  vcStreaming: boolean;
  amplitude: number;   // live mic RMS (0..1) for a level meter
}

// VC mic capture. Conversion now happens server-side in the MeanVC chat-proxy,
// so this hook only: (1) uploads the target voice, and (2) captures raw mic PCM
// and forwards it untagged to the proxy socket via `sendRawAudio`. The proxy
// converts each chunk and relays it to PersonaPlex over localhost.
export function useMeanVCPipeline(
  sendRawAudio: (data: ArrayBuffer) => void,
  initialSteps: number = 2,
  options: MeanVCPipelineOptions = {},
) {
  const [state, setState] = useState<MeanVCPipelineState>({
    vcEnabled: false,
    vcTargetId: null,
    vcTargetFile: null,
    vcTargetUrl: null,
    vcStatus: "",
    vcStreaming: false,
    amplitude: 0,
  });

  const pcmStreamRef = useRef<MediaStream | null>(null);
  const pcmContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sendingRef = useRef(false);
  // Raw (pre-conversion) mic PCM, kept for the post-conversation voice-change metrics.
  const originalPcmRef = useRef<Float32Array[]>([]);
  const resumeRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sendRawRef = useRef(sendRawAudio);
  sendRawRef.current = sendRawAudio;

  const uploadTarget = useCallback(async (file: File) => {
    const url = URL.createObjectURL(file);
    // Revoke previous URL
    if (state.vcTargetUrl) URL.revokeObjectURL(state.vcTargetUrl);
    setState(s => ({ ...s, vcTargetFile: file.name, vcTargetUrl: url, vcStatus: "Loading target voice..." }));
    const fd = new FormData();
    fd.append("wav", file);
    const loadTargetUrl = options.loadTargetUrl?.();
    if (!loadTargetUrl) {
      setState(s => ({ ...s, vcStatus: "Error: no load-target URL configured" }));
      return;
    }
    try {
      const resp = await fetch(loadTargetUrl, { method: "POST", body: fd });
      const data = await resp.json();
      if (data.target_id) {
        setState(s => ({
          ...s,
          vcTargetId: data.target_id,
          vcStatus: `Target ready: ${file.name} (${data.duration_seconds}s)`,
        }));
      } else {
        setState(s => ({ ...s, vcStatus: "Error: " + (data.error || "unknown") }));
      }
    } catch (e: any) {
      setState(s => ({ ...s, vcStatus: "Error: " + (e?.message || e) }));
    }
  }, []);

  // Phase 1: acquire mic + audio graph. Returns the proxy descriptor (including
  // the actual mic sample rate) so the caller can open the proxy socket with the
  // right source_sr. Mic frames are NOT sent until beginSending() flips the gate
  // (called once the PersonaPlex handshake arrives via the proxy).
  const startMic = useCallback(async (targetIdOverride?: string): Promise<ProxyDescriptor> => {
    // The study app passes only an opaque session_id in the socket URL, so it
    // provides an override here rather than pre-loading a browser-side target.
    const targetId = targetIdOverride ?? state.vcTargetId;
    if (!targetId) {
      setState(s => ({ ...s, vcStatus: "Upload a target voice first" }));
      throw new Error("No target voice loaded");
    }
    setState(s => ({ ...s, vcStatus: "Starting microphone...", vcStreaming: true }));

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    pcmStreamRef.current = stream;

    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
    pcmContextRef.current = audioCtx;
    await audioCtx.resume();

    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(2048, 1, 1);
    processorRef.current = processor;
    sendingRef.current = false;
    originalPcmRef.current = [];

    processor.onaudioprocess = (e) => {
      const ch = e.inputBuffer.getChannelData(0);
      // Live mic level (updates ~8x/s) so a meter works even before sending starts.
      let sum = 0;
      for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i];
      setState(s => ({ ...s, amplitude: Math.sqrt(sum / ch.length) }));
      if (!sendingRef.current) return;
      // Snapshot the raw mic (inputBuffer is reused, so copy) before sending.
      originalPcmRef.current.push(new Float32Array(ch));
      sendRawRef.current(ch.buffer);
    };
    source.connect(processor);
    // Near-silent sink keeps the ScriptProcessor firing.
    const gainNode = audioCtx.createGain();
    gainNode.gain.value = 0.001;
    processor.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    // Keep AudioContext alive during streaming
    resumeRef.current = setInterval(() => {
      if (pcmContextRef.current?.state === "suspended") {
        pcmContextRef.current.resume();
      }
    }, 1000);

    return {
      targetId,
      sourceSr: audioCtx.sampleRate,
      steps: initialSteps,
      voicePrompt: options.voicePrompt ?? "NATF2.pt",
    };
  }, [state.vcTargetId, initialSteps, options.voicePrompt]);

  // Assemble the captured raw mic into a 16 kHz WAV (the "Original" side of the
  // voice-change comparison). Returns null if nothing was captured.
  const getOriginalUserWav = useCallback((): Blob | null => {
    const parts = originalPcmRef.current;
    if (!parts.length) return null;
    const total = parts.reduce((n, p) => n + p.length, 0);
    const combined = new Float32Array(total);
    let off = 0;
    for (const p of parts) { combined.set(p, off); off += p.length; }
    return createWavFile(combined, 16000);
  }, []);

  // Phase 2: open the gate so mic PCM starts flowing to the proxy.
  const beginSending = useCallback(() => {
    sendingRef.current = true;
    setState(s => ({ ...s, vcStatus: "VC pipeline active - connected" }));
  }, []);

  const stopVCStream = useCallback(() => {
    sendingRef.current = false;
    if (resumeRef.current) clearInterval(resumeRef.current);
    if (processorRef.current) {
      processorRef.current.onaudioprocess = null;
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    pcmStreamRef.current?.getTracks().forEach(t => t.stop());
    pcmStreamRef.current = null;
    pcmContextRef.current?.close();
    pcmContextRef.current = null;
    setState(s => ({ ...s, vcStreaming: false, vcStatus: "", amplitude: 0 }));
  }, []);

  const setEnabled = useCallback((enabled: boolean) => {
    setState(s => ({ ...s, vcEnabled: enabled }));
  }, []);

  // Use a target that was preloaded server-side (the study app path): set the
  // target_id directly instead of uploading a WAV from the browser.
  const setPresetTarget = useCallback((targetId: string, label?: string) => {
    setState(s => ({
      ...s,
      vcTargetId: targetId,
      vcTargetFile: label ?? targetId,
      vcStatus: "Target ready",
    }));
  }, []);

  return {
    ...state,
    setEnabled,
    uploadTarget,
    setPresetTarget,
    startMic,
    beginSending,
    stopVCStream,
    getOriginalUserWav,
  };
}
