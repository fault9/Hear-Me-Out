// ============================================================================
//  SOUNDBOARD AUDIT INPUT TIMING
// ----------------------------------------------------------------------------
//  Deterministic, sample-referenced speech/pause annotations for frozen audit
//  stimuli. Matched natural/converted slots are annotated from their shared raw
//  take, so the route comparison uses identical participant timing even if VC
//  leaves low-level energy in a source pause.
// ============================================================================

import { decodeAndConformToPp } from "@/lib/audioFormat"

export const AUDIT_INPUT_FRAME_MS = 20
export const AUDIT_INPUT_RMS_THRESHOLD = 0.012
export const AUDIT_MIN_SPEECH_MS = 80
export const AUDIT_MIN_PAUSE_MS = 250

export interface AuditSampleInterval {
  start_sample: number
  end_sample: number
}

export interface AuditInputTiming {
  sample_rate_hz: number
  total_samples: number
  frame_samples: number
  rms_threshold: number
  min_speech_samples: number
  min_pause_samples: number
  speech_start_sample: number | null
  speech_end_sample: number | null
  speech_intervals: AuditSampleInterval[]
  pause_intervals: AuditSampleInterval[]
}

function mergeIntervals(
  intervals: AuditSampleInterval[],
  maxGapSamples: number,
): AuditSampleInterval[] {
  const merged: AuditSampleInterval[] = []
  for (const interval of intervals) {
    const previous = merged[merged.length - 1]
    if (previous && interval.start_sample - previous.end_sample < maxGapSamples) {
      previous.end_sample = Math.max(previous.end_sample, interval.end_sample)
    } else {
      merged.push({ ...interval })
    }
  }
  return merged
}

export function detectAuditInputTiming(
  pcm: Float32Array,
  sampleRate: number,
): AuditInputTiming {
  const frameSamples = Math.max(1, Math.round(sampleRate * AUDIT_INPUT_FRAME_MS / 1000))
  const minSpeechSamples = Math.round(sampleRate * AUDIT_MIN_SPEECH_MS / 1000)
  const minPauseSamples = Math.round(sampleRate * AUDIT_MIN_PAUSE_MS / 1000)
  const activeFrames: AuditSampleInterval[] = []

  for (let start = 0; start < pcm.length; start += frameSamples) {
    const end = Math.min(pcm.length, start + frameSamples)
    let sumSquares = 0
    for (let sample = start; sample < end; sample++) {
      sumSquares += pcm[sample] * pcm[sample]
    }
    const rms = Math.sqrt(sumSquares / Math.max(1, end - start))
    if (rms >= AUDIT_INPUT_RMS_THRESHOLD) {
      activeFrames.push({ start_sample: start, end_sample: end })
    }
  }

  const activeRuns = mergeIntervals(activeFrames, 1)
  const speechIntervals = mergeIntervals(activeRuns, minPauseSamples)
    .filter((interval) => interval.end_sample - interval.start_sample >= minSpeechSamples)
  const pauseIntervals = speechIntervals.slice(1).map((interval, index) => ({
    start_sample: speechIntervals[index].end_sample,
    end_sample: interval.start_sample,
  }))

  return {
    sample_rate_hz: sampleRate,
    total_samples: pcm.length,
    frame_samples: frameSamples,
    rms_threshold: AUDIT_INPUT_RMS_THRESHOLD,
    min_speech_samples: minSpeechSamples,
    min_pause_samples: minPauseSamples,
    speech_start_sample: speechIntervals[0]?.start_sample ?? null,
    speech_end_sample: speechIntervals[speechIntervals.length - 1]?.end_sample ?? null,
    speech_intervals: speechIntervals,
    pause_intervals: pauseIntervals,
  }
}

export async function analyzeAuditInput(blob: Blob): Promise<AuditInputTiming> {
  const decoded = await decodeAndConformToPp(blob)
  return detectAuditInputTiming(decoded.pcm, decoded.sampleRate)
}
