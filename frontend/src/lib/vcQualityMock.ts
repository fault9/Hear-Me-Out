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
  wer: 0.12, cer: 0.05,
  secs: 0.78,
  f0_pcc: 0.71, f0_rmse: 18.4,
  sig: 3.42, bak: 4.05, ovrl: 3.21, p808_mos: 3.55,
  utmos: 3.61,

  vc_transcript: "she had your dark suit in greasy wash water all year",
  ref_transcript: "she had your dark suit in greasy wash water all year",
  ref_kind: "ground_truth",

  segment_mode: "fixed_2s_hop1s",
  segments: [
    { start: 0.0, end: 2.0, secs: 0.81, utmos: 3.70, sig: 3.50, ovrl: 3.30, wer: 0.00, f0_pcc: 0.72, f0_rmse: 17.1 },
    { start: 1.0, end: 3.0, secs: 0.77, utmos: 3.62, sig: 3.45, ovrl: 3.22, wer: 0.00, f0_pcc: 0.70, f0_rmse: 18.2 },
    { start: 2.0, end: 4.0, secs: 0.42, utmos: 2.10, sig: 2.05, ovrl: 2.00, wer: 0.50, f0_pcc: 0.41, f0_rmse: 31.8 },
    { start: 3.0, end: 5.0, secs: 0.55, utmos: 2.85, sig: 2.65, ovrl: 2.50, wer: 0.50, f0_pcc: 0.58, f0_rmse: 25.4 },
    { start: 4.0, end: 6.0, secs: 0.79, utmos: 3.66, sig: 3.48, ovrl: 3.25, wer: 0.00, f0_pcc: 0.71, f0_rmse: 17.5 },
    { start: 5.0, end: 7.0, secs: 0.80, utmos: 3.68, sig: 3.51, ovrl: 3.28, wer: 0.00, f0_pcc: 0.73, f0_rmse: 16.9 },
    { start: 6.0, end: 8.0, secs: 0.79, utmos: 3.65, sig: 3.49, ovrl: 3.26, wer: 0.00, f0_pcc: 0.72, f0_rmse: 17.3 },
    { start: 6.2, end: 8.2, secs: 0.78, utmos: 3.63, sig: 3.47, ovrl: 3.24, wer: 0.00, f0_pcc: 0.71, f0_rmse: 17.6 },
  ],
  anomalies: [
    { start: 2.0, end: 4.0, metric: "secs",  score: 0.42, z: -2.71 },
    { start: 2.0, end: 4.0, metric: "utmos", score: 2.10, z: -2.42 },
    { start: 2.0, end: 4.0, metric: "ovrl",  score: 2.00, z: -2.55 },
    { start: 2.0, end: 4.0, metric: "wer",   score: 0.50, z:  2.05 },
    { start: 3.0, end: 5.0, metric: "utmos", score: 2.85, z: -1.05 },
  ],
}
