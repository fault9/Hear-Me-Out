import { useState, useRef, useCallback, useEffect } from "react";
import { getPersonaplexWsURL, getChatProxyWsUrl } from "@/lib/config";
import { createWavFile } from "@/lib/audio";

// PersonaPlex transcript timing heuristics. PP emits text without per-word
// timestamps, so we estimate turn placement from sentence end + audio packet
// arrival. Equivalent constants live in useConversation.ts — keep in sync.
const MIN_TURN_GAP_SECONDS = 0.2           // minimum gap between consecutive PP turns
const MIN_PP_TURN_DURATION_SECONDS = 0.8   // shortest plausible PP utterance
const MAX_PP_TURN_DURATION_SECONDS = 12    // cap for over-estimated PP turn length
const PP_WORDS_PER_SECOND = 2.7            // typical PP speech rate, used to estimate turn duration

// When VC is enabled, connect() targets the MeanVC chat-proxy instead of
// PersonaPlex directly. The proxy converts mic audio server-side and forwards
// it to PersonaPlex over localhost.
export interface ProxyDescriptor {
  targetId: string;
  sourceSr: number;
  steps: number;
  voicePrompt?: string;
  // Initial VC state at connect. The proxy socket is opened whenever a target
  // exists (even when VC is off), then toggled live via sendControl().
  vcEnabled: boolean;
}

export interface Transcript {
  text: string;
  timestamp: number;
  start?: number;
  end?: number;
  speaker: "user" | "personaplex";
}

declare global {
  interface Window {
    webkitAudioContext?: new (options?: AudioContextOptions) => AudioContext;
    "ogg-opus-decoder": {
      OggOpusDecoder: new () => OggOpusDecoder;
    };
  }
}

interface OggOpusDecoder {
  readonly ready: Promise<void>;
  decode(packet: Uint8Array): Promise<{
    channelData: Float32Array[];
    samplesDecoded: number;
    sampleRate: number;
  }>;
  free(): void;
}

type SinkableAudioContext = AudioContext & {
  setSinkId?: (sinkId: string) => Promise<void>;
};

function createAudioContext(options?: AudioContextOptions): AudioContext {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  return new AudioContextCtor(options);
}

async function setAudioSink(ctx: AudioContext | null, deviceId: string, label: string) {
  const sinkable = ctx as SinkableAudioContext | null;
  if (typeof sinkable?.setSinkId !== "function") return;
  try {
    await sinkable.setSinkId(deviceId || "");
  } catch (e) {
    console.warn(`setSinkId (${label}) failed`, e);
  }
}

export function useWebSocket() {
  const socketRef = useRef<WebSocket | null>(null);
  const intentionalClose = useRef(false);
  const decoderRef = useRef<OggOpusDecoder | null>(null);
  const mergedCtxRef = useRef<AudioContext | null>(null);
  const mergedDestRef = useRef<AudioNode | null>(null);
  const mergedEndRef = useRef(0);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const scheduledEnd = useRef(0);
  // Optional live monitor of the converted voice (0x03 frames), playable
  // through a user-selected output device, separate from PersonaPlex output.
  const feedbackCtxRef = useRef<AudioContext | null>(null);
  const feedbackEnd = useRef(0);
  const feedbackEnabledRef = useRef(false);
  const desiredPplxSinkRef = useRef<string>("");
  const runIdRef = useRef(0);
  const lastTranscriptEndRef = useRef(0);

  const setMergedOutput = useCallback((ctx: AudioContext | null, dest: AudioNode | null) => {
    mergedCtxRef.current = ctx;
    mergedDestRef.current = dest;
    mergedEndRef.current = 0;
  }, []);
  const personaplexOpus = useRef<{ packet: Uint8Array; time: number }[]>([]);
  const vcUserPcm = useRef<Float32Array[]>([]);
  const conversationStart = useRef(0);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [partialTranscript, setPartialTranscript] = useState("");
  const [responseChunks, setResponseChunks] = useState<ArrayBuffer[]>([]);
  const [warmupComplete, setWarmupComplete] = useState(false);
  const [handshakeReceived, setHandshakeReceived] = useState(false);

  useEffect(() => {
    const init = async () => {
      const OggDecoder = window["ogg-opus-decoder"]?.OggOpusDecoder;
      if (OggDecoder) {
        const decoder = new OggDecoder();
        await decoder.ready;
        decoderRef.current = decoder;
        console.log("Opus decoder ready");
      }
      audioCtxRef.current = createAudioContext({ sampleRate: 48000 });
      // Apply any output device chosen before the context existed.
      if (desiredPplxSinkRef.current) {
        await setAudioSink(audioCtxRef.current, desiredPplxSinkRef.current, "personaplex");
      }
    };
    init();
    return () => {
      decoderRef.current?.free();
      audioCtxRef.current?.close();
      feedbackCtxRef.current?.close();
    };
  }, []);

  const playAudio = useCallback((payload: ArrayBuffer, runId: number) => {
    const decoder = decoderRef.current;
    const ctx = audioCtxRef.current;
    if (!decoder || !ctx) return;

    const raw = new Uint8Array(payload);
    personaplexOpus.current.push({ packet: raw, time: Date.now() });

    decoder.decode(raw).then(({ channelData, samplesDecoded }) => {
      if (runId !== runIdRef.current) return;
      if (samplesDecoded === 0) return;

      // Play through speakers
      const buffer = ctx.createBuffer(1, samplesDecoded, ctx.sampleRate);
      buffer.copyToChannel(channelData[0], 0);
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      const now = ctx.currentTime;
      const start = Math.max(scheduledEnd.current, now);
      src.start(start);
      scheduledEnd.current = start + buffer.duration;

      // Also route to merged capture stream
      const mctx = mergedCtxRef.current;
      const mdest = mergedDestRef.current;
      if (mctx && mdest && mctx.state !== "closed") {
        const mbuf = mctx.createBuffer(1, samplesDecoded, mctx.sampleRate);
        mbuf.copyToChannel(channelData[0], 0);
        const msrc = mctx.createBufferSource();
        msrc.buffer = mbuf;
        msrc.connect(mdest);
        const mnow = mctx.currentTime;
        const mstart = Math.max(mergedEndRef.current, mnow);
        msrc.start(mstart);
        mergedEndRef.current = mstart + mbuf.duration;
      }
    }).catch(() => {});
  }, []);

  const latestPersonaplexAudioEnd = useCallback((): number => {
    const packets = personaplexOpus.current;
    if (packets.length === 0 || conversationStart.current === 0) return 0;
    const lastPacket = packets[packets.length - 1];
    return Math.max(0, (lastPacket.time - conversationStart.current) / 1000);
  }, []);

  // Schedule a chunk of converted-voice PCM (raw float32 @16kHz) into the
  // feedback context for monitoring.
  const playFeedback = useCallback((pcm: Float32Array) => {
    const ctx = feedbackCtxRef.current;
    if (!ctx || ctx.state === "closed" || pcm.length === 0) return;
    const buf = ctx.createBuffer(1, pcm.length, 16000);
    buf.copyToChannel(pcm, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const now = ctx.currentTime;
    const start = Math.max(feedbackEnd.current, now);
    src.start(start);
    feedbackEnd.current = start + buf.duration;
  }, []);

  // Route PersonaPlex playback to a chosen output device ("" = system default).
  const setPersonaplexSink = useCallback(async (deviceId: string) => {
    desiredPplxSinkRef.current = deviceId;
    await setAudioSink(audioCtxRef.current, deviceId, "personaplex");
  }, []);

  // Enable/disable the converted-voice monitor and pick its output device.
  const configureFeedback = useCallback(async (enabled: boolean, deviceId: string) => {
    feedbackEnabledRef.current = enabled;
    if (!enabled) {
      await feedbackCtxRef.current?.suspend().catch(() => {});
      return;
    }
    let ctx = feedbackCtxRef.current;
    if (!ctx || ctx.state === "closed") {
      ctx = createAudioContext();
      feedbackCtxRef.current = ctx;
      feedbackEnd.current = 0;
    }
    await ctx.resume().catch(() => {});
    await setAudioSink(ctx, deviceId, "feedback");
  }, []);

  const connect = useCallback((textPrompt?: string, proxy?: ProxyDescriptor) => {
    const runId = runIdRef.current + 1;
    runIdRef.current = runId;
    const url = proxy
      ? getChatProxyWsUrl(proxy.targetId, proxy.sourceSr, proxy.steps, textPrompt ?? "", proxy.voicePrompt, proxy.vcEnabled)
      : getPersonaplexWsURL(textPrompt);
    console.log("Connecting to:", url);
    setError(null);
    personaplexOpus.current = [];
    vcUserPcm.current = [];
    conversationStart.current = 0;
    lastTranscriptEndRef.current = 0;
    intentionalClose.current = false;

    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;

    socket.onopen = () => {
      if (runId !== runIdRef.current) return;
      console.log("WebSocket connected, waiting for handshake...");
      setConnected(true);
    };

    socket.onerror = () => {
      if (runId !== runIdRef.current) return;
      setError("Connection failed. Check if the server is running.");
    };

    socket.onclose = (event) => {
      if (runId !== runIdRef.current) return;
      setConnected(false);
      if (!intentionalClose.current) {
        if (event.code === 1006) {
          setError("Server disconnected unexpectedly. The model may be overloaded.");
        } else if (event.code !== 1000 && event.code !== 1005) {
          setError(`Connection closed (code ${event.code}). ${event.reason || ""}`.trim());
        }
      }
      intentionalClose.current = false;
    };

    socket.onmessage = async (event) => {
      if (runId !== runIdRef.current) return;
      try {
        const arrayBuffer = await (event.data instanceof Blob
          ? event.data.arrayBuffer()
          : event.data);
        const view = new Uint8Array(arrayBuffer);
        const tag = view[0];
        const payload = arrayBuffer.slice(1);

        if (tag === 0) {
          if (runId !== runIdRef.current) return;
          console.log("Handshake received, server ready");
          conversationStart.current = Date.now();
          scheduledEnd.current = 0;
          mergedEndRef.current = 0;
          feedbackEnd.current = 0;
          setWarmupComplete(true);
          setHandshakeReceived(true);
        } else if (tag === 1) {
          playAudio(payload, runId);
        } else if (tag === 2) {
          const decoder = new TextDecoder();
          const text = decoder.decode(payload);
          setPartialTranscript((prev) => {
            if (runId !== runIdRef.current) return prev;
            const updated = prev + text;
            if (updated.endsWith(".") || updated.endsWith("!") || updated.endsWith("?")) {
              const nowSec =
                conversationStart.current > 0
                  ? (Date.now() - conversationStart.current) / 1000
                  : 0;
              const audioEnd = latestPersonaplexAudioEnd();
              const end = Math.max(audioEnd, nowSec, lastTranscriptEndRef.current + MIN_TURN_GAP_SECONDS);
              const wordCount = Math.max(1, updated.trim().split(/\s+/).length);
              const estimatedDuration = Math.min(
                MAX_PP_TURN_DURATION_SECONDS,
                Math.max(MIN_PP_TURN_DURATION_SECONDS, wordCount / PP_WORDS_PER_SECOND),
              );
              const start = Math.max(lastTranscriptEndRef.current, end - estimatedDuration);
              lastTranscriptEndRef.current = Math.max(end, start + MIN_TURN_GAP_SECONDS);
              setTranscripts((t) => [
                ...t,
                {
                  text: updated,
                  timestamp: Date.now(),
                  start,
                  end: lastTranscriptEndRef.current,
                  speaker: "personaplex",
                },
              ]);
              return "";
            }
            return updated;
          });
        } else if (tag === 3) {
          // Converted user voice from the proxy (raw float32 PCM @16kHz):
          // kept for the user/merged WAV downloads, and optionally monitored live.
          const pcm = new Float32Array(payload);
          vcUserPcm.current.push(pcm);
          if (feedbackEnabledRef.current) playFeedback(pcm);
        }
      } catch {
        // Ignore unrecognized messages
      }
    };
  }, [latestPersonaplexAudioEnd, playAudio, playFeedback]);

  const sendAudio = useCallback((data: ArrayBuffer) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      const tagged = new Uint8Array(data.byteLength + 1);
      tagged[0] = 1;
      tagged.set(new Uint8Array(data), 1);
      socketRef.current.send(tagged.buffer);
    }
  }, []);

  // Raw, untagged binary send — used in proxy/VC mode where the chat-proxy
  // expects raw float32 mic PCM.
  const sendRawAudio = useCallback((data: ArrayBuffer) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(data);
    }
  }, []);

  // JSON text-frame control channel to the chat-proxy (e.g. live VC on/off:
  // { type: "vc_control", enabled: boolean }). No-op if the socket is closed
  // or this is a direct PersonaPlex connection (which ignores text frames).
  const sendControl = useCallback((message: Record<string, unknown>) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message));
    }
  }, []);

  const disconnect = useCallback(() => {
    runIdRef.current += 1;
    intentionalClose.current = true;
    socketRef.current?.close();
    socketRef.current = null;
    setConnected(false);
    setWarmupComplete(false);
    setHandshakeReceived(false);
    scheduledEnd.current = 0;
    feedbackEnd.current = 0;
  }, []);

  useEffect(() => {
    return () => disconnect();
  }, [disconnect]);

  const clearTranscripts = useCallback(() => {
    setTranscripts([]);
    setPartialTranscript("");
    lastTranscriptEndRef.current = 0;
  }, []);

  const clearResponseChunks = useCallback(() => {
    setResponseChunks([]);
    personaplexOpus.current = [];
    vcUserPcm.current = [];
    scheduledEnd.current = 0;
    mergedEndRef.current = 0;
    feedbackEnd.current = 0;
  }, []);

  const getPersonaplexWav = useCallback(async (): Promise<Blob | null> => {
    const packets = personaplexOpus.current;
    console.log("getPersonaplexWav:", packets.length, "packets, decoder:", !!decoderRef.current);
    if (packets.length === 0) return null;
    const DecoderCtor = window["ogg-opus-decoder"]?.OggOpusDecoder;
    const decoder = DecoderCtor ? new DecoderCtor() : decoderRef.current;
    if (!decoder) return null;
    const shouldFreeDecoder = decoder !== decoderRef.current;
    if (shouldFreeDecoder) await decoder.ready;

    const decoded: { pcm: Float32Array; offset: number }[] = [];
    let sampleRate = 48000;
    try {
      for (const { packet, time } of packets) {
        try {
          const { channelData, samplesDecoded, sampleRate: decodedSampleRate } = await decoder.decode(packet);
          if (samplesDecoded > 0) {
            const offsetSeconds =
              conversationStart.current > 0
                ? Math.max(0, (time - conversationStart.current) / 1000)
                : 0;
            sampleRate = decodedSampleRate || sampleRate;
            decoded.push({
              pcm: new Float32Array(channelData[0]),
              offset: Math.round(offsetSeconds * sampleRate),
            });
          }
        } catch {
          continue;
        }
      }
    } finally {
      if (shouldFreeDecoder) decoder.free();
    }

    if (decoded.length === 0) return null;
    const total = decoded.reduce((s, c) => Math.max(s, c.offset + c.pcm.length), 0);
    const combined = new Float32Array(total);
    for (const { pcm, offset } of decoded) {
      for (let i = 0; i < pcm.length; i++) {
        const idx = offset + i;
        combined[idx] = Math.max(-1, Math.min(1, combined[idx] + pcm[i]));
      }
    }
    return createWavFile(combined, sampleRate);
  }, []);

  // Assemble the converted user voice (0x03 frames) collected in proxy mode.
  const getVcUserWav = useCallback((): Blob | null => {
    const chunks = vcUserPcm.current;
    if (chunks.length === 0) return null;
    const total = chunks.reduce((s, c) => s + c.length, 0);
    const combined = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) { combined.set(c, offset); offset += c.length; }
    console.log("[proxy] VC user WAV:", total, "samples");
    vcUserPcm.current = [];
    return createWavFile(combined, 16000);
  }, []);

  const getPersonaplexStartTime = useCallback((): number => {
    if (personaplexOpus.current.length === 0) return 0;
    return (personaplexOpus.current[0].time - conversationStart.current) / 1000;
  }, []);

  const getConversationDuration = useCallback((): number => {
    const packets = personaplexOpus.current;
    return packets.length * 0.02; // ~20ms per Opus frame
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    connected,
    error,
    transcripts,
    partialTranscript,
    responseChunks,
    warmupComplete,
    handshakeReceived,
    connect,
    disconnect,
    sendAudio,
    sendRawAudio,
    sendControl,
    getVcUserWav,
    setPersonaplexSink,
    configureFeedback,
    clearTranscripts,
    clearResponseChunks,
    clearError,
    getPersonaplexWav,
    getPersonaplexStartTime,
    getConversationDuration,
    setMergedOutput,
  };
}
