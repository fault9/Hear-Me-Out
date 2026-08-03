// The study client only ever holds an opaque session_id. The active VC engine
// resolves the hidden prompt/target/steps server-side, so the WS URL carries no
// system prompt or target — that is the whole point of the study privacy model.
import { traceparent } from "@/lib/trace";

export function getStudyChatProxyWsUrl(sessionId: string, sourceSr: number): string {
  const params = new URLSearchParams({
    session_id: sessionId,
    source_sr: String(sourceSr),
    // W3C trace context: the browser can't set WS headers, so pass it as a query
    // param; the VC proxy reads it to continue the session's distributed trace.
    traceparent: traceparent(),
  });
  // Same-origin by default: the app-api relays this socket to the VC proxy, so
  // the audio connection shares the page's port and certificate — first-party
  // to ad blockers, and no second certificate acceptance in Firefox. Setting
  // VITE_MEANVC_HOST (dev) connects straight to the engine on :5002 instead.
  const devHost = (import.meta as any).env?.VITE_MEANVC_HOST;
  if (devHost) return `wss://${devHost}:5002/api/meanvc/chat-proxy?${params.toString()}`;
  return `wss://${window.location.host}/api/meanvc/chat-proxy?${params.toString()}`;
}
