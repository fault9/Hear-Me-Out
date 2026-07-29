// ============================================================================
//  COUNTED-SESSION COUNTER
// ----------------------------------------------------------------------------
//  A persistent, monotonically increasing session number for the soundboard
//  "counted session" bootstrap (P4 — session independence). Each counted
//  session gets the next integer, surviving page reloads via localStorage, so
//  session ids in the timing log are stable and ordered rather than random.
//
//  Uniqueness across machines/operators is provided by the short random suffix
//  in the session id (see SoundboardPanel); the counter gives human-readable,
//  ordered numbering within one operator's browser.
// ============================================================================

const KEY = "hmo_soundboard_session_counter"

function read(): number {
  if (typeof localStorage === "undefined") return 0
  const n = Number(localStorage.getItem(KEY) || "0")
  return Number.isFinite(n) && n >= 0 ? n : 0
}

// Next session number, incrementing and persisting the counter.
export function nextSessionNumber(): number {
  const next = read() + 1
  try { localStorage.setItem(KEY, String(next)) } catch { /* private mode: best-effort */ }
  return next
}

// Current counter without incrementing (for display before a session starts).
export function peekSessionNumber(): number {
  return read()
}

// Reset to 0 — e.g. starting a fresh experiment run.
export function resetSessionCounter(): void {
  try { localStorage.setItem(KEY, "0") } catch { /* best-effort */ }
}
