export const API_BASE = "";

const env = import.meta.env as Record<string, string | undefined>;

export type VoiceAnalysisMode = "off" | "manual" | "after_vc";

export function normalizeVoiceAnalysisMode(
  value: string | null | undefined,
  fallback: VoiceAnalysisMode = "manual",
): VoiceAnalysisMode {
  if (value === "off" || value === "manual" || value === "after_vc") return value;
  if (value === "auto") return "after_vc";
  return fallback;
}

export function getVoiceAnalysisMode(): VoiceAnalysisMode {
  const queryValue =
    typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("voice_analysis")
      : null;
  return normalizeVoiceAnalysisMode(queryValue ?? env.VITE_VOICE_ANALYSIS_MODE);
}

export function getMeanvcWsUrl(targetId: string, sourceSr: number, steps: number = 8): string {
  const host = env.VITE_MEANVC_HOST || "130.237.3.103";
  return `wss://${host}:5002/api/meanvc/stream?target_id=${targetId}&steps=${steps}&source_sr=${sourceSr}`;
}

export function getMeanvcLoadTargetUrl(): string {
  const host = env.VITE_MEANVC_HOST || "130.237.3.103";
  return `https://${host}:5002/api/meanvc/load-target`;
}

// Both MeanVC and X-VC mount the same routes on :5002, so the only way to know
// which engine is actually running is to ask. Returned shape: { engine: 'meanvc' | 'xvc' }.
export function getMeanvcInfoUrl(): string {
  const host = env.VITE_MEANVC_HOST || "130.237.3.103";
  return `https://${host}:5002/api/meanvc/info`;
}

// Server-side VC bridge: the browser talks to MeanVC, which converts the mic
// audio and forwards it to PersonaPlex over localhost (no browser round trip).
// The returned socket speaks PersonaPlex's framing (0x00/0x01/0x02) plus 0x03
// for the converted user voice used by downloads.
export function getChatProxyWsUrl(
  targetId: string,
  sourceSr: number,
  steps: number,
  textPrompt: string,
  voicePrompt: string = "NATF2.pt",
  vcEnabled: boolean = true,
): string {
  const host = env.VITE_MEANVC_HOST || "130.237.3.103";
  const params = new URLSearchParams({
    target_id: targetId,
    steps: String(steps),
    source_sr: String(sourceSr),
    text_prompt: textPrompt,
    voice_prompt: voicePrompt,
    // Initial conversion state. The socket is always the proxy when a target
    // exists; the proxy converts or passes through per this flag, flipped live
    // via {"type":"vc_control",...} control frames.
    vc_enabled: String(vcEnabled),
  });
  return `wss://${host}:5002/api/meanvc/chat-proxy?${params.toString()}`;
}

const DEFAULT_PROMPT = "You enjoy having a good conversation.";

export function getPersonaplexWsURL(textPrompt?: string): string {
  // Allow override via URL query param (?personaplex_ws=ws://...)
  const override = new URLSearchParams(window.location.search).get("personaplex_ws");
  if (override) return override;
  const wsHost = env.VITE_PERSONAPLEX_HOST || window.location.hostname;
  const voicePrompt = "NATF2.pt";
  const tp = textPrompt || DEFAULT_PROMPT;
  return `wss://${wsHost}:8000/api/chat?voice_prompt=${encodeURIComponent(voicePrompt)}&text_prompt=${encodeURIComponent(tp)}`;
}
