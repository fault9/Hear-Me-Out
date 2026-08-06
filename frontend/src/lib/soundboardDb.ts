// ============================================================================
//  SOUNDBOARD STORAGE — IndexedDB
// ----------------------------------------------------------------------------
//  Three object stores:
//    slots    — per-slot data (raw + baked WAV blobs, all metadata)
//    targets  — target voices (built-in default + user uploads)
//    sessions — timing log rows, one per slot playback during a conversation
//
//  No external dep — raw IDB wrapped in tiny promise helpers. Blobs are
//  stored natively (the browser handles binary fine; no base64 round-trips).
// ============================================================================

import {
  SOUNDBOARD_DB_NAME,
  SOUNDBOARD_DB_VERSION,
  DEFAULT_TARGET,
} from "@/lib/soundboardConfig"

export type ManipulationMode = "unconverted" | "vc" | "pitch_formant"
export type SourceSpeaker = "masculine_presenting" | "feminine_presenting"

// Per-slot record. raw is the as-recorded WAV at PP_SAMPLE_RATE; baked is the
// post-VC or post-pitch-formant WAV (also at PP_SAMPLE_RATE). For unconverted
// slots, baked === raw conceptually and we just point to the raw blob.
export interface Slot {
  id: string
  label: string
  // Researcher-set experimental condition tag. Used for filtering the runtime
  // soundboard and for analysing session logs afterward.
  condition: string
  manipulation: ManipulationMode
  // Prespecified audit source identity (e.g. masculine_presenting). This is
  // experimental metadata, not inferred from the recording or its filename.
  sourceSpeaker?: SourceSpeaker

  // VC-mode fields
  targetId?: string
  engine?: "meanvc" | "xvc" | "seedvc"

  // Pitch/formant-mode fields
  pitchSemitones?: number
  formantShift?: number

  // Audio
  raw: Blob | null              // null only on transient (just-created) slots
  baked: Blob | null            // null until bake completes
  sampleRate: number            // SHOULD equal PP_SAMPLE_RATE; recorded for audit
  rawDurationMs: number
  bakedDurationMs: number

  // Format-integrity audit
  driftMs: number               // |baked - raw| in ms; >tolerance = flagged

  // Optional: VC-quality eval result (lazy / nice-to-have)
  qualityScore?: number | null

  // Loudness normalization audit (P3). When `normalized`, the baked clip was
  // EBU-R128 gain-matched to `targetLufs`; `measuredLufs` is the achieved
  // integrated loudness (may differ slightly after the anti-clip peak guard).
  normalized?: boolean
  targetLufs?: number
  measuredLufs?: number | null

  // Auto-Whisper transcript of the baked (or raw, for unconverted slots) audio.
  // Injected into the conversation transcript when the slot plays, so
  // soundboard-driven turns appear alongside live-voice turns in the diarized
  // view. Populated during bake or on first play if missing.
  transcript?: string | null

  bakeTimestamp?: number        // unix ms
  // Manual sort key for the Configure tab's up/down reordering. Defaults to
  // createdAt (insertion order) until the researcher reorders. Kept on the
  // createdAt scale so slots with and without an explicit order sort together.
  order?: number
  createdAt: number
  updatedAt: number
}

export interface Target {
  id: string
  label: string
  builtin: boolean
  // Built-in targets (e.g. NATF2) are referenced by name on the server side
  // and don't carry a blob here. Uploaded targets store the WAV directly.
  wav: Blob | null
  sampleRate: number | null
  createdAt: number
}

// One row in the per-session timing log. Two kinds, distinguished by
// `eventType`:
//   - "slot": a soundboard clip was sent to PP (the slot* + play* fields).
//   - "pp_speech_start" / "pp_speech_end": PersonaPlex began/finished a speech
//     run (only timestampMs is meaningful; slot fields are absent).
// All rows carry `timestampMs` on the SAME performance.now() clock, so response
// latency, overlap, and barge-in are computable across both kinds. Legacy rows
// written before this field existed have eventType/timestampMs undefined; the
// exporter treats them as "slot" with timestampMs = playStartMs.
export type SessionEventType = "slot" | "pp_speech_start" | "pp_speech_end"

export interface SessionRow {
  // Auto-increment id (set by IDB). Not stable across exports.
  id?: number
  sessionId: string             // groups rows of one conversation
  conditionContext: string      // session-level tag (e.g. "experiment_2_pilot")
  eventType?: SessionEventType  // undefined on legacy rows → treated as "slot"
  // performance.now() at the event — the common clock for all row kinds.
  timestampMs?: number
  // Slot-playback fields (present when eventType === "slot").
  slotId?: string
  slotLabel?: string
  slotCondition?: string
  // Times are performance.now() snapshots: monotonic, sub-ms resolution.
  playStartMs?: number
  playEndMs?: number
  clipDurationMs?: number
  // True when this playback was a rule-triggered retry (operator replayed the
  // turn because PP stayed silent / barged in). Lets analysis separate planned
  // turns from protocol-driven repeats. Only meaningful for eventType "slot".
  retry?: boolean
  // Absolute wall-clock (unix ms) — used for cross-kind chronological sort.
  timestamp: number
}

// ----------------------------------------------------------------------------
//  IDB open + tiny promise wrappers
// ----------------------------------------------------------------------------

let _dbPromise: Promise<IDBDatabase> | null = null

function open(): Promise<IDBDatabase> {
  if (_dbPromise) return _dbPromise
  _dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(SOUNDBOARD_DB_NAME, SOUNDBOARD_DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains("slots")) {
        const slots = db.createObjectStore("slots", { keyPath: "id" })
        slots.createIndex("byCondition", "condition", { unique: false })
        slots.createIndex("byCreatedAt", "createdAt", { unique: false })
      }
      if (!db.objectStoreNames.contains("targets")) {
        db.createObjectStore("targets", { keyPath: "id" })
      }
      if (!db.objectStoreNames.contains("sessions")) {
        const sessions = db.createObjectStore("sessions", {
          keyPath: "id",
          autoIncrement: true,
        })
        sessions.createIndex("bySessionId", "sessionId", { unique: false })
        sessions.createIndex("byTimestamp", "timestamp", { unique: false })
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
  return _dbPromise
}

function tx<T>(
  store: string,
  mode: IDBTransactionMode,
  fn: (s: IDBObjectStore) => IDBRequest<T> | Promise<T>,
): Promise<T> {
  return open().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const t = db.transaction(store, mode)
        const s = t.objectStore(store)
        const r = fn(s)
        if (r instanceof Promise) {
          r.then(resolve, reject)
          t.oncomplete = () => {}
        } else {
          r.onsuccess = () => resolve(r.result)
          r.onerror = () => reject(r.error)
        }
        t.onerror = () => reject(t.error)
        t.onabort = () => reject(t.error)
      }),
  )
}

// ----------------------------------------------------------------------------
//  Slot CRUD
// ----------------------------------------------------------------------------

export function listSlots(): Promise<Slot[]> {
  return tx("slots", "readonly", (s) => s.getAll() as IDBRequest<Slot[]>).then(
    (rows) => rows.sort((a, b) => (a.order ?? a.createdAt) - (b.order ?? b.createdAt)),
  )
}

export function getSlot(id: string): Promise<Slot | undefined> {
  return tx("slots", "readonly", (s) => s.get(id) as IDBRequest<Slot | undefined>)
}

export function putSlot(slot: Slot): Promise<void> {
  slot.updatedAt = Date.now()
  return tx("slots", "readwrite", (s) => s.put(slot) as IDBRequest<IDBValidKey>).then(() => {})
}

export function deleteSlot(id: string): Promise<void> {
  return tx("slots", "readwrite", (s) => s.delete(id) as IDBRequest<undefined>).then(() => {})
}

// ----------------------------------------------------------------------------
//  Target CRUD
// ----------------------------------------------------------------------------

export function listTargets(): Promise<Target[]> {
  return tx("targets", "readonly", (s) => s.getAll() as IDBRequest<Target[]>).then(
    (rows) => rows.sort((a, b) => a.createdAt - b.createdAt),
  )
}

export function putTarget(target: Target): Promise<void> {
  return tx("targets", "readwrite", (s) => s.put(target) as IDBRequest<IDBValidKey>).then(() => {})
}

export function deleteTarget(id: string): Promise<void> {
  return tx("targets", "readwrite", (s) => s.delete(id) as IDBRequest<undefined>).then(() => {})
}

// Seed the built-in default target if it isn't there yet. Idempotent.
export async function ensureBuiltinTarget(): Promise<void> {
  const existing = await tx("targets", "readonly", (s) =>
    s.get(DEFAULT_TARGET.id) as IDBRequest<Target | undefined>,
  )
  if (existing) return
  await putTarget({
    id: DEFAULT_TARGET.id,
    label: DEFAULT_TARGET.label,
    builtin: true,
    wav: null,
    sampleRate: null,
    createdAt: Date.now(),
  })
}

// ----------------------------------------------------------------------------
//  Session log
// ----------------------------------------------------------------------------

export function appendSessionRow(row: SessionRow): Promise<void> {
  return tx("sessions", "readwrite", (s) => s.add(row) as IDBRequest<IDBValidKey>).then(() => {})
}

export function listSessionRows(sessionId?: string): Promise<SessionRow[]> {
  return tx("sessions", "readonly", (s) => {
    if (sessionId) {
      const idx = s.index("bySessionId")
      return idx.getAll(IDBKeyRange.only(sessionId)) as IDBRequest<SessionRow[]>
    }
    return s.getAll() as IDBRequest<SessionRow[]>
  }).then((rows) => rows.sort((a, b) => a.timestamp - b.timestamp))
}

export function clearSessionRows(sessionId?: string): Promise<void> {
  return open().then(
    (db) =>
      new Promise<void>((resolve, reject) => {
        const t = db.transaction("sessions", "readwrite")
        const s = t.objectStore("sessions")
        if (!sessionId) {
          const r = s.clear()
          r.onsuccess = () => resolve()
          r.onerror = () => reject(r.error)
          return
        }
        const idx = s.index("bySessionId")
        const cursorReq = idx.openCursor(IDBKeyRange.only(sessionId))
        cursorReq.onsuccess = () => {
          const c = cursorReq.result
          if (!c) return resolve()
          c.delete()
          c.continue()
        }
        cursorReq.onerror = () => reject(cursorReq.error)
      }),
  )
}

// ----------------------------------------------------------------------------
//  Bulk read for export
// ----------------------------------------------------------------------------

export async function bulkExport(): Promise<{ slots: Slot[]; targets: Target[] }> {
  const [slots, targets] = await Promise.all([listSlots(), listTargets()])
  return { slots, targets }
}

// Replace EVERYTHING with imported state. Use only from explicit import flow.
export async function bulkReplace(
  slots: Slot[],
  targets: Target[],
): Promise<void> {
  await open()
  return new Promise<void>((resolve, reject) => {
    open().then((db) => {
      const t = db.transaction(["slots", "targets"], "readwrite")
      const slotStore = t.objectStore("slots")
      const targetStore = t.objectStore("targets")
      slotStore.clear()
      targetStore.clear()
      for (const s of slots) slotStore.put(s)
      for (const tgt of targets) targetStore.put(tgt)
      t.oncomplete = () => resolve()
      t.onerror = () => reject(t.error)
      t.onabort = () => reject(t.error)
    }, reject)
  })
}

// Util: generate a stable-enough id without pulling in uuid.
export function newId(prefix: string = "id"): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
}
