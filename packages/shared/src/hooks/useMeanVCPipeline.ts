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
  studyTimeline?: boolean;
}

interface CapturedPcmPart {
  startSample: number;
  pcm: Float32Array;
}

interface CaptureTimelineEntry {
  chunk_sequence: number;
  capture_start_sample: number;
  sample_count: number;
  timeline_start_ms: number | null;
}

const STUDY_FRAME_HEADER_BYTES = 20;

function frameStudyPcm(
  pcm: Float32Array,
  sequence: number,
  captureStartSample: number,
  sampleRate: number,
): ArrayBuffer {
  const framed = new ArrayBuffer(STUDY_FRAME_HEADER_BYTES + pcm.byteLength);
  const view = new DataView(framed);
  view.setUint8(0, 0x48); // H
  view.setUint8(1, 0x4d); // M
  view.setUint8(2, 0x4f); // O
  view.setUint8(3, 0x31); // 1
  view.setUint32(4, sequence, true);
  view.setUint32(8, captureStartSample, true);
  view.setUint32(12, pcm.length, true);
  view.setUint32(16, sampleRate, true);
  new Uint8Array(framed, STUDY_FRAME_HEADER_BYTES).set(
    new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength),
  );
  return framed;
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
  const originalPcmRef = useRef<CapturedPcmPart[]>([]);
  const resumeRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sendRawRef = useRef(sendRawAudio);
  sendRawRef.current = sendRawAudio;
  const captureStatsRef = useRef({
    callbacks: 0,
    samples: 0,
    estimatedDroppedSamples: 0,
    timelineGapCount: 0,
    maxTimelineGapSamples: 0,
  });
  const captureSampleRateRef = useRef(16000);
  const firstPlaybackTimeRef = useRef<number | null>(null);
  const captureSequenceRef = useRef(0);
  const timelineEpochPerfMsRef = useRef<number | null>(null);
  const captureTimelineRef = useRef<CaptureTimelineEntry[]>([]);

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
    captureSampleRateRef.current = audioCtx.sampleRate;
    await audioCtx.resume();

    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(2048, 1, 1);
    processorRef.current = processor;
    sendingRef.current = false;
    originalPcmRef.current = [];
    captureStatsRef.current = {
      callbacks: 0,
      samples: 0,
      estimatedDroppedSamples: 0,
      timelineGapCount: 0,
      maxTimelineGapSamples: 0,
    };
    firstPlaybackTimeRef.current = null;
    captureSequenceRef.current = 0;
    timelineEpochPerfMsRef.current = null;
    captureTimelineRef.current = [];

    processor.onaudioprocess = (e) => {
      const ch = e.inputBuffer.getChannelData(0);
      // Live mic level (updates ~8x/s) so a meter works even before sending starts.
      let sum = 0;
      for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i];
      setState(s => ({ ...s, amplitude: Math.sqrt(sum / ch.length) }));
      if (!sendingRef.current) return;
      if (firstPlaybackTimeRef.current === null) firstPlaybackTimeRef.current = e.playbackTime;
      const hasPreviousPart = originalPcmRef.current.length > 0;
      const previousEnd = originalPcmRef.current.length
        ? originalPcmRef.current[originalPcmRef.current.length - 1].startSample
          + originalPcmRef.current[originalPcmRef.current.length - 1].pcm.length
        : 0;
      const clockStart = Math.max(
        previousEnd,
        Math.round((e.playbackTime - firstPlaybackTimeRef.current) * audioCtx.sampleRate),
      );
      const timelineGapSamples = hasPreviousPart ? clockStart - previousEnd : 0;
      if (timelineGapSamples > 0) {
        captureStatsRef.current.estimatedDroppedSamples += timelineGapSamples;
        captureStatsRef.current.timelineGapCount += 1;
        captureStatsRef.current.maxTimelineGapSamples = Math.max(
          captureStatsRef.current.maxTimelineGapSamples,
          timelineGapSamples,
        );
      }
      captureStatsRef.current.callbacks += 1;
      captureStatsRef.current.samples += ch.length;
      // Snapshot the raw mic (inputBuffer is reused, so copy) before sending.
      const pcm = new Float32Array(ch);
      originalPcmRef.current.push({ startSample: clockStart, pcm });
      captureSequenceRef.current += 1;
      const performanceStartMs = performance.now()
        + (e.playbackTime - audioCtx.currentTime) * 1000;
      captureTimelineRef.current.push({
        chunk_sequence: captureSequenceRef.current,
        capture_start_sample: clockStart,
        sample_count: pcm.length,
        timeline_start_ms: timelineEpochPerfMsRef.current === null
          ? null
          : performanceStartMs - timelineEpochPerfMsRef.current,
      });
      sendRawRef.current(options.studyTimeline
        ? frameStudyPcm(
            pcm,
            captureSequenceRef.current,
            clockStart,
            audioCtx.sampleRate,
          )
        : pcm.buffer);
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
  }, [state.vcTargetId, initialSteps, options.voicePrompt, options.studyTimeline]);

  // Assemble the captured raw mic into a 16 kHz WAV (the "Original" side of the
  // voice-change comparison). Returns null if nothing was captured.
  const getOriginalUserWav = useCallback((): Blob | null => {
    const parts = originalPcmRef.current;
    if (!parts.length) return null;
    const total = parts.reduce((n, part) => Math.max(n, part.startSample + part.pcm.length), 0);
    const combined = new Float32Array(total);
    for (const part of parts) combined.set(part.pcm, part.startSample);
    return createWavFile(combined, captureSampleRateRef.current);
  }, []);

  const getCaptureStats = useCallback(() => ({
    ...captureStatsRef.current,
    sampleRateHz: captureSampleRateRef.current,
    detector: "script_processor_playback_time",
  }), []);

  // Phase 2: open the gate so mic PCM starts flowing to the proxy.
  const getClientCaptureTimeline = useCallback(() => ({
    schema: "hmo.client-capture-timeline.v2",
    epoch: "personaplex_handshake_performance_now",
    sample_rate_hz: captureSampleRateRef.current,
    callbacks: captureStatsRef.current.callbacks,
    captured_samples: captureStatsRef.current.samples,
    estimated_dropped_samples: captureStatsRef.current.estimatedDroppedSamples,
    timeline_gap_count: captureStatsRef.current.timelineGapCount,
    max_timeline_gap_samples: captureStatsRef.current.maxTimelineGapSamples,
    detector: "sample_timeline_boundary_gap",
    chunks: captureTimelineRef.current,
  }), []);

  const beginSending = useCallback((epochPerformanceMs?: number) => {
    timelineEpochPerfMsRef.current = epochPerformanceMs ?? performance.now();
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
    getCaptureStats,
    getClientCaptureTimeline,
  };
}
