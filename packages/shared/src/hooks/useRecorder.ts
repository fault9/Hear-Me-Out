import { useState, useRef, useCallback } from "react";

declare class Recorder {
  constructor(opts: Record<string, unknown>);
  start(): Promise<void>;
  stop(): void;
  ondataavailable: ((buf: ArrayBuffer) => void) | null;
}

export interface RecorderState {
  recorder: Recorder | null;
  isRecording: boolean;
  amplitude: number;
  recordedChunks: Blob[];
  recordingAvailable: boolean;
}

export function useRecorder(onAudioData: (buf: ArrayBuffer) => void) {
  const [state, setState] = useState<RecorderState>({
    recorder: null,
    isRecording: false,
    amplitude: 0,
    recordedChunks: [],
    recordingAvailable: false,
  });
  const recordingStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mergedCtxRef = useRef<AudioContext | null>(null);
  const mergedDestRef = useRef<MediaStreamAudioDestinationNode | null>(null);
  const mergedRecorderRef = useRef<MediaRecorder | null>(null);
  const mergedChunksRef = useRef<Blob[]>([]);

  const getMergedChunks = useCallback(() => mergedChunksRef.current, []);

  const start = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordingStreamRef.current = stream;

    // Opus encoder for PersonaPlex
    const recorder = new Recorder({
      encoderPath: "https://cdn.jsdelivr.net/npm/opus-recorder@latest/dist/encoderWorker.min.js",
      streamPages: true,
      encoderApplication: 2049,
      encoderFrameSize: 80,
      encoderSampleRate: 24000,
      maxFramesPerPage: 1,
      numberOfChannels: 1,
    });
    recorder.ondataavailable = async (arrayBuffer: ArrayBuffer) => {
      onAudioData(arrayBuffer);
    };
    await recorder.start();
    setState((s) => ({ ...s, recorder, isRecording: true, recordedChunks: [], recordingAvailable: false }));

    // NOTE: a per-frame amplitude analyzer used to live here, calling setState
    // ~60×/sec. `amplitude` is consumed nowhere, so that was re-rendering the
    // whole conversation tree on every animation frame — starving the main
    // thread that also decodes/schedules PersonaPlex's Opus audio in the WS
    // onmessage handler, which caused audible playback stutter. Removed. (It
    // also leaked an AudioContext that was never closed.) Re-add a THROTTLED
    // meter if a mic level indicator is ever needed.

    // Mic-only WebM recorder
    const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorderRef.current = mr;
    mr.ondataavailable = (ev) => {
      if (ev.data.size > 0) {
        setState((s) => ({ ...s, recordedChunks: [...s.recordedChunks, ev.data] }));
      }
    };
    mr.onstop = () => setState((s) => ({ ...s, recordingAvailable: true }));
    mr.start();

    // Merged audio context: routes mic + PersonaPlex into one stream
    mergedChunksRef.current = [];
    const mctx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 48000 });
    mergedCtxRef.current = mctx;
    const micSource = mctx.createMediaStreamSource(stream);
    const mergedDest = mctx.createMediaStreamDestination();
    mergedDestRef.current = mergedDest;
    micSource.connect(mergedDest);

    // Recorder for merged stream
    const mmr = new MediaRecorder(mergedDest.stream, { mimeType: "audio/webm" });
    mergedRecorderRef.current = mmr;
    mmr.ondataavailable = (ev) => {
      if (ev.data.size > 0) mergedChunksRef.current = [...mergedChunksRef.current, ev.data];
    };
    mmr.start();
  }, [onAudioData]);

  const stop = useCallback(() => {
    state.recorder?.stop();
    setState((s) => ({ ...s, isRecording: false }));
    mediaRecorderRef.current?.stop();
    recordingStreamRef.current?.getTracks().forEach((t) => t.stop());
    mergedRecorderRef.current?.stop();
    setTimeout(() => {
      mergedCtxRef.current?.close();
      mergedCtxRef.current = null;
      mergedDestRef.current = null;
    }, 500);
  }, [state.recorder]);

  return {
    ...state,
    start,
    stop,
    mergedDestination: mergedDestRef.current,
    mergedContext: mergedCtxRef.current,
    getMergedChunks,
  };
}