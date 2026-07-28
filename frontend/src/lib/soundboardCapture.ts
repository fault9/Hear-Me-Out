// ============================================================================
//  SOUNDBOARD CAPTURE — conversation-scoped record of what was sent to PP
// ----------------------------------------------------------------------------
//  The runtime SoundboardPanel and the conversation-finalization logic
//  (useConversation) live in different component subtrees and don't share
//  React state — and ConversationView unmounts on tab switch. Rather than
//  prop-drill through that boundary, useSoundboardPlayback writes each sent
//  clip into this module-level buffer, and useConversation reads it back when
//  the conversation ends.
//
//  Why this is the right source of truth:
//    - The mic recorder captures the physical microphone, which is MUTED while
//      the soundboard plays. So the mic recording contains none of the
//      soundboard audio. The only faithful record of what PP heard from the
//      "user" is the clips we deliberately sent — captured here.
//    - Each entry keeps BOTH the exact bytes sent (baked when a bake exists,
//      else raw) AND the pre-bake raw take, so VC-quality can use the sent
//      audio as "converted" and the raw as "source".
//
//  Lifecycle: resetCapture() at conversation start (useConversation), one
//  captureClip() per soundboard play (useSoundboardPlayback), and the
//  assemble*/singleVcTarget readers at conversation end.
// ============================================================================

import type { ManipulationMode } from "@/lib/soundboardDb"
import { decodeToPcm, resampleTo } from "@/lib/audioFormat"
import { createWavFile } from "@shared/lib/audio"
import { PP_SAMPLE_RATE } from "@/lib/soundboardConfig"

export interface CapturedClip {
  slotId: string
  slotLabel: string
  condition: string
  manipulation: ManipulationMode
  targetId?: string
  // Exact bytes sent to PP for this play (baked if a bake exists, else raw).
  sentBlob: Blob
  // Pre-bake raw take, for VC-quality's "source" reference. Null only if the
  // slot somehow had no raw (shouldn't happen for a bakeable slot).
  rawBlob: Blob | null
  // Whisper text for the clip; may be filled in late by the on-demand
  // transcribe fallback (the entry is mutated in place once it resolves).
  transcript: string | null
  // Conversation-relative seconds (from ws.getConversationElapsed) so the
  // final transcript can interleave soundboard turns with PersonaPlex turns.
  startSec: number
  endSec: number
}

let clips: CapturedClip[] = []

export function resetCapture(): void {
  clips = []
}

// Push a clip and return the SAME object reference so the caller can mutate
// `.transcript` later (on-demand Whisper resolves after the clip is captured).
export function captureClip(c: CapturedClip): CapturedClip {
  clips.push(c)
  return c
}

export function getCapturedClips(): CapturedClip[] {
  return clips.slice()
}

export function hasCapture(): boolean {
  return clips.length > 0
}

// A session-level VC-quality comparison (converted vs source vs target) is only
// well-defined when EVERY sent clip is a VC bake toward the SAME single target.
// Mixed modes (some unconverted / pitch-formant) or multiple targets would make
// a single converted/source/target triple meaningless, so we return null and
// the UI leaves the analysis disabled. Returns the shared targetId, or null.
export function singleVcTarget(): string | null {
  if (clips.length === 0) return null
  if (!clips.every((c) => c.manipulation === "vc" && c.targetId)) return null
  const ids = new Set(clips.map((c) => c.targetId))
  return ids.size === 1 ? clips[0].targetId! : null
}

// Decode each blob (already at PP_SAMPLE_RATE for soundboard bakes, but we
// resample defensively) and concatenate back-to-back in play order. Gaps of
// conversational silence are intentionally dropped — for WER/SECS/F0 the
// spoken content is what matters, and long silences would dilute the scores.
async function concatToWav(blobs: Blob[]): Promise<Blob | null> {
  const pcms: Float32Array[] = []
  for (const b of blobs) {
    const decoded = await decodeToPcm(b)
    const pcm =
      decoded.sampleRate === PP_SAMPLE_RATE
        ? decoded.pcm
        : await resampleTo(decoded.pcm, decoded.sampleRate, PP_SAMPLE_RATE)
    pcms.push(pcm)
  }
  const total = pcms.reduce((n, p) => n + p.length, 0)
  if (total === 0) return null
  const combined = new Float32Array(total)
  let off = 0
  for (const p of pcms) {
    combined.set(p, off)
    off += p.length
  }
  return createWavFile(combined, PP_SAMPLE_RATE)
}

// GAPLESS concat of the sent clips — VC-quality's "converted" input. Silence
// gaps are intentionally dropped here so WER/SECS/UTMOS score the speech, not
// the dead air between plays.
export function assembleSentWav(): Promise<Blob | null> {
  return concatToWav(clips.map((c) => c.sentBlob))
}

// GAPLESS concat of the pre-bake raw takes — VC-quality's "source". Null if any
// clip lacks a raw take, since a partial source would misalign with converted.
export function assembleRawWav(): Promise<Blob | null> {
  const raws = clips.map((c) => c.rawBlob)
  if (raws.some((b) => !b)) return Promise.resolve(null)
  return concatToWav(raws as Blob[])
}

// TIMELINE placement: each clip is written into a silence buffer at its
// conversation-relative start time (startSec), so the recording reflects WHEN
// each clip was actually played. PersonaPlex audio is positioned on the same
// t=0 (handshake) timeline, so merging the two yields a properly synced
// conversation recording — this is what the "You"/"All" downloads use.
async function timelineToWav(pick: (c: CapturedClip) => Blob | null): Promise<Blob | null> {
  const placed: { pcm: Float32Array; startSample: number }[] = []
  let maxEnd = 0
  for (const c of clips) {
    const b = pick(c)
    if (!b) continue
    const decoded = await decodeToPcm(b)
    const pcm =
      decoded.sampleRate === PP_SAMPLE_RATE
        ? decoded.pcm
        : await resampleTo(decoded.pcm, decoded.sampleRate, PP_SAMPLE_RATE)
    const startSample = Math.max(0, Math.round(c.startSec * PP_SAMPLE_RATE))
    placed.push({ pcm, startSample })
    maxEnd = Math.max(maxEnd, startSample + pcm.length)
  }
  if (maxEnd === 0) return null
  const buf = new Float32Array(maxEnd)
  for (const { pcm, startSample } of placed) {
    // Overlay (clips shouldn't overlap in practice; clamp defensively).
    for (let i = 0; i < pcm.length; i++) {
      const idx = startSample + i
      buf[idx] = Math.max(-1, Math.min(1, buf[idx] + pcm[i]))
    }
  }
  return createWavFile(buf, PP_SAMPLE_RATE)
}

// Downloadable "You" WAV: sent clips on the conversation timeline.
export function assembleSentTimelineWav(): Promise<Blob | null> {
  return timelineToWav((c) => c.sentBlob)
}

// Downloadable "You (raw)" WAV: pre-bake raw takes on the conversation timeline.
export function assembleRawTimelineWav(): Promise<Blob | null> {
  if (clips.some((c) => !c.rawBlob)) return Promise.resolve(null)
  return timelineToWav((c) => c.rawBlob)
}
