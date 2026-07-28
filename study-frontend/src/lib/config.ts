// The study client only ever holds an opaque session_id. The active VC engine
// resolves the hidden prompt/target/steps server-side, so the WS URL carries no
// system prompt or target — that is the whole point of the study privacy model.
import { traceparent } from "@/lib/trace";

export function getStudyChatProxyWsUrl(sessionId: string, sourceSr: number): string {
  const host = (import.meta as any).env?.VITE_MEANVC_HOST || window.location.hostname;
  const params = new URLSearchParams({
    session_id: sessionId,
    source_sr: String(sourceSr),
    // W3C trace context: the browser can't set WS headers, so pass it as a query
    // param; the VC proxy reads it to continue the session's distributed trace.
    traceparent: traceparent(),
  });
  return `wss://${host}:5002/api/meanvc/chat-proxy?${params.toString()}`;
}
