// ============================================================================
//  SOUNDBOARD ZIP — export / import the stimulus set
// ----------------------------------------------------------------------------
//  Minimal store-mode (no compression) ZIP writer and reader. WAVs don't
//  compress meaningfully and DEFLATE would add a heavy dep, so we keep the
//  bytes raw and skip the compressor entirely.
//
//  Layout of an exported soundboard:
//      manifest.json   — { version, slots: [...], targets: [...] } (no blobs)
//      slots/<id>/raw.wav
//      slots/<id>/baked.wav        (if baked)
//      targets/<id>.wav            (only for non-builtin targets)
//
//  A collaborator who imports this should reconstruct an IDENTICAL set of
//  slots — same raw audio, same baked audio, same metadata. PP will receive
//  bit-identical bytes when those slots play.
// ============================================================================

import type { Slot, Target } from "@/lib/soundboardDb"

const ZIP_VERSION_NEEDED = 20
const SIGNATURE_LFH = 0x04034b50    // local file header
const SIGNATURE_CDH = 0x02014b50    // central directory header
const SIGNATURE_EOCD = 0x06054b50   // end of central directory

const MANIFEST_FILENAME = "manifest.json"
const MANIFEST_VERSION = 1

interface ManifestSlot {
  id: string
  label: string
  condition: string
  manipulation: string
  sourceSpeaker?: string
  targetId?: string
  engine?: string
  pitchSemitones?: number
  formantShift?: number
  sampleRate: number
  rawDurationMs: number
  bakedDurationMs: number
  driftMs: number
  qualityScore?: number | null
  normalized?: boolean
  targetLufs?: number
  measuredLufs?: number | null
  bakeTimestamp?: number
  createdAt: number
  updatedAt: number
  hasRaw: boolean
  hasBaked: boolean
}

interface ManifestTarget {
  id: string
  label: string
  builtin: boolean
  sampleRate: number | null
  createdAt: number
  hasWav: boolean
}

interface Manifest {
  version: number
  exportedAt: number
  slots: ManifestSlot[]
  targets: ManifestTarget[]
}

// ----------------------------------------------------------------------------
//  CRC-32 (IEEE 802.3, polynomial 0xEDB88320). One-shot, table-driven.
// ----------------------------------------------------------------------------

let _crcTable: Uint32Array | null = null
function crcTable(): Uint32Array {
  if (_crcTable) return _crcTable
  const t = new Uint32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    t[n] = c
  }
  _crcTable = t
  return t
}

function crc32(bytes: Uint8Array): number {
  const t = crcTable()
  let c = 0xffffffff
  for (let i = 0; i < bytes.length; i++) c = t[(c ^ bytes[i]) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

// ----------------------------------------------------------------------------
//  Writer
// ----------------------------------------------------------------------------

interface WriteEntry {
  name: string
  data: Uint8Array
}

function writeLfh(
  view: DataView,
  off: number,
  nameBytes: Uint8Array,
  data: Uint8Array,
  crc: number,
): number {
  view.setUint32(off + 0, SIGNATURE_LFH, true)
  view.setUint16(off + 4, ZIP_VERSION_NEEDED, true)
  view.setUint16(off + 6, 0x0800, true)         // bit 11: UTF-8 filename
  view.setUint16(off + 8, 0, true)               // method: 0 = store
  view.setUint16(off + 10, 0, true)              // mod time
  view.setUint16(off + 12, 0, true)              // mod date
  view.setUint32(off + 14, crc, true)
  view.setUint32(off + 18, data.length, true)    // compressed size
  view.setUint32(off + 22, data.length, true)    // uncompressed size
  view.setUint16(off + 26, nameBytes.length, true)
  view.setUint16(off + 28, 0, true)              // extra field length
  return 30 + nameBytes.length + data.length
}

function writeCdh(
  view: DataView,
  off: number,
  nameBytes: Uint8Array,
  data: Uint8Array,
  crc: number,
  localOffset: number,
): number {
  view.setUint32(off + 0, SIGNATURE_CDH, true)
  view.setUint16(off + 4, ZIP_VERSION_NEEDED, true)
  view.setUint16(off + 6, ZIP_VERSION_NEEDED, true)
  view.setUint16(off + 8, 0x0800, true)
  view.setUint16(off + 10, 0, true)
  view.setUint16(off + 12, 0, true)
  view.setUint16(off + 14, 0, true)
  view.setUint32(off + 16, crc, true)
  view.setUint32(off + 20, data.length, true)
  view.setUint32(off + 24, data.length, true)
  view.setUint16(off + 28, nameBytes.length, true)
  view.setUint16(off + 30, 0, true)              // extra field length
  view.setUint16(off + 32, 0, true)              // comment length
  view.setUint16(off + 34, 0, true)              // disk number start
  view.setUint16(off + 36, 0, true)              // internal attrs
  view.setUint32(off + 38, 0, true)              // external attrs
  view.setUint32(off + 42, localOffset, true)
  return 46 + nameBytes.length
}

export function makeZip(entries: WriteEntry[]): Blob {
  const enc = new TextEncoder()
  const pre = entries.map((e) => {
    const nameBytes = enc.encode(e.name)
    const crc = crc32(e.data)
    return { ...e, nameBytes, crc }
  })

  // Phase 1: total size pass
  let dataSize = 0
  for (const e of pre) dataSize += 30 + e.nameBytes.length + e.data.length
  let cdSize = 0
  for (const e of pre) cdSize += 46 + e.nameBytes.length
  const total = dataSize + cdSize + 22

  const buf = new ArrayBuffer(total)
  const view = new DataView(buf)
  const u8 = new Uint8Array(buf)

  // Phase 2: write local file headers + data
  let off = 0
  const localOffsets: number[] = []
  for (const e of pre) {
    localOffsets.push(off)
    writeLfh(view, off, e.nameBytes, e.data, e.crc)
    off += 30
    u8.set(e.nameBytes, off)
    off += e.nameBytes.length
    u8.set(e.data, off)
    off += e.data.length
  }

  // Phase 3: central directory
  const cdStart = off
  for (let i = 0; i < pre.length; i++) {
    const e = pre[i]
    writeCdh(view, off, e.nameBytes, e.data, e.crc, localOffsets[i])
    off += 46
    u8.set(e.nameBytes, off)
    off += e.nameBytes.length
  }

  // Phase 4: end of central directory
  view.setUint32(off + 0, SIGNATURE_EOCD, true)
  view.setUint16(off + 4, 0, true)
  view.setUint16(off + 6, 0, true)
  view.setUint16(off + 8, pre.length, true)
  view.setUint16(off + 10, pre.length, true)
  view.setUint32(off + 12, cdSize, true)
  view.setUint32(off + 16, cdStart, true)
  view.setUint16(off + 20, 0, true)

  return new Blob([buf], { type: "application/zip" })
}

// ----------------------------------------------------------------------------
//  Reader
// ----------------------------------------------------------------------------

interface ReadEntry {
  name: string
  data: Uint8Array
}

export async function readZip(blob: Blob): Promise<ReadEntry[]> {
  const buf = await blob.arrayBuffer()
  const view = new DataView(buf)
  const u8 = new Uint8Array(buf)

  // Locate EOCD by scanning backward from end (max comment = 65535).
  let eocdOff = -1
  const maxScan = Math.min(buf.byteLength, 22 + 0xffff)
  for (let i = buf.byteLength - 22; i >= buf.byteLength - maxScan; i--) {
    if (view.getUint32(i, true) === SIGNATURE_EOCD) { eocdOff = i; break }
  }
  if (eocdOff < 0) throw new Error("readZip: EOCD not found (not a ZIP?)")

  const numEntries = view.getUint16(eocdOff + 10, true)
  const cdOff = view.getUint32(eocdOff + 16, true)

  const dec = new TextDecoder("utf-8")
  const entries: ReadEntry[] = []
  let p = cdOff
  for (let i = 0; i < numEntries; i++) {
    if (view.getUint32(p, true) !== SIGNATURE_CDH) {
      throw new Error(`readZip: bad central directory signature at ${p}`)
    }
    const method = view.getUint16(p + 10, true)
    if (method !== 0) {
      throw new Error(`readZip: entry uses compression method ${method}; only store (0) supported`)
    }
    const compSize = view.getUint32(p + 20, true)
    const nameLen = view.getUint16(p + 28, true)
    const extraLen = view.getUint16(p + 30, true)
    const commentLen = view.getUint16(p + 32, true)
    const localOff = view.getUint32(p + 42, true)
    const name = dec.decode(u8.subarray(p + 46, p + 46 + nameLen))
    p += 46 + nameLen + extraLen + commentLen

    // Walk to the data via the local file header.
    if (view.getUint32(localOff, true) !== SIGNATURE_LFH) {
      throw new Error(`readZip: bad local header at ${localOff}`)
    }
    const lhNameLen = view.getUint16(localOff + 26, true)
    const lhExtraLen = view.getUint16(localOff + 28, true)
    const dataOff = localOff + 30 + lhNameLen + lhExtraLen
    const data = u8.subarray(dataOff, dataOff + compSize)
    entries.push({ name, data: new Uint8Array(data) })
  }
  return entries
}

// ----------------------------------------------------------------------------
//  Manifest + entry packaging
// ----------------------------------------------------------------------------

async function blobToBytes(b: Blob | null): Promise<Uint8Array | null> {
  if (!b) return null
  return new Uint8Array(await b.arrayBuffer())
}

export async function exportSoundboard(
  slots: Slot[],
  targets: Target[],
): Promise<Blob> {
  const manifest: Manifest = {
    version: MANIFEST_VERSION,
    exportedAt: Date.now(),
    slots: slots.map((s) => ({
      id: s.id,
      label: s.label,
      condition: s.condition,
      manipulation: s.manipulation,
      sourceSpeaker: s.sourceSpeaker,
      targetId: s.targetId,
      engine: s.engine,
      pitchSemitones: s.pitchSemitones,
      formantShift: s.formantShift,
      sampleRate: s.sampleRate,
      rawDurationMs: s.rawDurationMs,
      bakedDurationMs: s.bakedDurationMs,
      driftMs: s.driftMs,
      qualityScore: s.qualityScore ?? null,
      normalized: s.normalized,
      targetLufs: s.targetLufs,
      measuredLufs: s.measuredLufs ?? null,
      bakeTimestamp: s.bakeTimestamp,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
      hasRaw: !!s.raw,
      hasBaked: !!s.baked,
    })),
    targets: targets.map((t) => ({
      id: t.id,
      label: t.label,
      builtin: t.builtin,
      sampleRate: t.sampleRate,
      createdAt: t.createdAt,
      hasWav: !!t.wav,
    })),
  }

  const entries: WriteEntry[] = [
    {
      name: MANIFEST_FILENAME,
      data: new TextEncoder().encode(JSON.stringify(manifest, null, 2)),
    },
  ]
  for (const s of slots) {
    const raw = await blobToBytes(s.raw)
    if (raw) entries.push({ name: `slots/${s.id}/raw.wav`, data: raw })
    const baked = await blobToBytes(s.baked)
    if (baked) entries.push({ name: `slots/${s.id}/baked.wav`, data: baked })
  }
  for (const t of targets) {
    const wav = await blobToBytes(t.wav)
    if (wav) entries.push({ name: `targets/${t.id}.wav`, data: wav })
  }
  return makeZip(entries)
}

export async function importSoundboard(
  zip: Blob,
): Promise<{ slots: Slot[]; targets: Target[] }> {
  const entries = await readZip(zip)
  const byName = new Map<string, Uint8Array>()
  for (const e of entries) byName.set(e.name, e.data)

  const manifestBytes = byName.get(MANIFEST_FILENAME)
  if (!manifestBytes) throw new Error("importSoundboard: manifest.json missing")
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as Manifest
  if (manifest.version !== MANIFEST_VERSION) {
    throw new Error(
      `importSoundboard: manifest version ${manifest.version} unsupported (expected ${MANIFEST_VERSION})`,
    )
  }

  const wavBlob = (bytes: Uint8Array | undefined): Blob | null =>
    bytes ? new Blob([bytes], { type: "audio/wav" }) : null

  const slots: Slot[] = manifest.slots.map((m) => ({
    id: m.id,
    label: m.label,
    condition: m.condition,
    manipulation: m.manipulation as Slot["manipulation"],
    sourceSpeaker: m.sourceSpeaker as Slot["sourceSpeaker"],
    targetId: m.targetId,
    engine: m.engine as Slot["engine"],
    pitchSemitones: m.pitchSemitones,
    formantShift: m.formantShift,
    raw: wavBlob(byName.get(`slots/${m.id}/raw.wav`)),
    baked: wavBlob(byName.get(`slots/${m.id}/baked.wav`)),
    sampleRate: m.sampleRate,
    rawDurationMs: m.rawDurationMs,
    bakedDurationMs: m.bakedDurationMs,
    driftMs: m.driftMs,
    qualityScore: m.qualityScore ?? null,
    normalized: m.normalized,
    targetLufs: m.targetLufs,
    measuredLufs: m.measuredLufs ?? null,
    bakeTimestamp: m.bakeTimestamp,
    createdAt: m.createdAt,
    updatedAt: m.updatedAt,
  }))

  const targets: Target[] = manifest.targets.map((m) => ({
    id: m.id,
    label: m.label,
    builtin: m.builtin,
    wav: wavBlob(byName.get(`targets/${m.id}.wav`)),
    sampleRate: m.sampleRate,
    createdAt: m.createdAt,
  }))

  return { slots, targets }
}
