// Minimal W3C trace-context for the browser — no OpenTelemetry JS SDK / bundle.
// The frontend doesn't emit spans itself; it just generates a `traceparent` so the
// backend (VC proxy :5002 + app-api :5001 /condition + PersonaPlex bridge) roots a
// single distributed trace at the participant's scenario session. Search Jaeger by
// the trace id, or by the `study.session_id` attribute the backend stamps.

function hex(bytes: number): string {
  const a = new Uint8Array(bytes);
  crypto.getRandomValues(a);
  return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
}

// One trace id per scenario session; refreshed by startTrace() at session start.
let currentTraceId = hex(16);

/** Begin a fresh trace (call when a scenario session / audio check starts). */
export function startTrace(): string {
  currentTraceId = hex(16);
  return currentTraceId;
}

/** A W3C traceparent (new span id) under the current trace, for one request/WS. */
export function traceparent(): string {
  return `00-${currentTraceId}-${hex(8)}-01`;
}

/** Header form for fetch(). */
export function traceHeaders(): Record<string, string> {
  return { traceparent: traceparent() };
}
