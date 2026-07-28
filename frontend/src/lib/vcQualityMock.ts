// Synthetic VcQualityResult for previewing the overlay without a backend.
// Loaded by ConversationView when ?demo=vc-quality is in the URL.
// Values mimic a real X-VC run on a ~8-second VCTK utterance where one
// mid-clip window has a speaker-identity drop (the kind of regression
// the per-segment anomaly view is designed to surface).
import type { VcQualityResult } from "@shared/services/api"
import type { DiarizedTurn } from "@/hooks/useConversation"

// Synthetic diarized conversation that aligns with the segment grid below.
// User turns sit inside the segment windows so clicking a bar produces a
// "Conversation in this window" snippet in the detail card.
export const VC_QUALITY_DEMO_DIARIZED: DiarizedTurn[] = [
  { speaker: "user",        text: "Can you tell me a short story about a fox?", start: 0.2, end: 3.4 },
  { speaker: "personaplex", text: "Sure! Once upon a time, a quick brown fox lived in the woods.", start: 3.6, end: 5.6 },
  { speaker: "user",        text: "What did he do next?", start: 5.8, end: 7.0 },
  { speaker: "personaplex", text: "He jumped over a lazy dog and ran home for dinner.", start: 7.1, end: 8.2 },
]

export const VC_QUALITY_DEMO: VcQualityResult = {
  converted_path: "demo://session/turn3_converted.wav",
  target_path:    "vctk_16k/scottish_p234/p234_010_mic1.wav",
  source_path:    "demo://session/turn3_source.wav",

  // Headline (whole-clip)
  wer: 0.12,
  sim: 0.78,
  utmos: 3.61,

  vc_transcript: "she had your dark suit in greasy wash water all year",
  ref_transcript: "she had your dark suit in greasy wash water all year",
  ref_kind: "ground_truth",

  segment_mode: "fixed_2s_hop1s",
  segments: [
    { start: 0.0, end: 2.0, sim: 0.81, utmos: 3.70, wer: 0.00 },
    { start: 1.0, end: 3.0, sim: 0.77, utmos: 3.62, wer: 0.00 },
    { start: 2.0, end: 4.0, sim: 0.42, utmos: 2.10, wer: 0.50 },
    { start: 3.0, end: 5.0, sim: 0.55, utmos: 2.85, wer: 0.50 },
    { start: 4.0, end: 6.0, sim: 0.79, utmos: 3.66, wer: 0.00 },
    { start: 5.0, end: 7.0, sim: 0.80, utmos: 3.68, wer: 0.00 },
    { start: 6.0, end: 8.0, sim: 0.79, utmos: 3.65, wer: 0.00 },
    { start: 6.2, end: 8.2, sim: 0.78, utmos: 3.63, wer: 0.00 },
  ],
  anomalies: [
    { start: 2.0, end: 4.0, metric: "sim",   score: 0.42, z: -2.71 },
    { start: 2.0, end: 4.0, metric: "utmos", score: 2.10, z: -2.42 },
    { start: 2.0, end: 4.0, metric: "wer",   score: 0.50, z:  2.05 },
    { start: 3.0, end: 5.0, metric: "utmos", score: 2.85, z: -1.05 },
  ],
}
