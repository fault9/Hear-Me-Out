// ============================================================================
//  CONFIGURE SOUNDBOARD — slot authoring (slow setup)
// ----------------------------------------------------------------------------
//  Slots, targets, recording, bake (VC / pitch-formant / unconverted),
//  preview, export/import. The runtime panel in ConversationView never
//  shows this UI — it just renders the buttons that play these slots.
// ============================================================================

import { useEffect, useRef, useState } from "react"
import { Button } from "@shared/ui/button"
import { Input } from "@shared/ui/input"
import { Label } from "@shared/ui/label"
import { Card, CardContent } from "@shared/ui/card"
import { Spinner } from "@shared/ui/spinner"
import { Badge } from "@shared/ui/badge"
import {
  Mic, Square, Trash2, Play, Download, Upload, Wand2, Sparkles,
  CheckCircle2, AlertTriangle, Pause, FileAudio, ChevronUp, ChevronDown, CopyPlus,
} from "lucide-react"
import { useSoundboard } from "@/hooks/useSoundboard"
import {
  PP_SAMPLE_RATE,
  PP_BIT_DEPTH,
  DURATION_DRIFT_TOLERANCE_MS,
  PITCH_FORMANT_PRESETS,
  PITCH_FORMANT_DEFAULTS,
  LOUDNESS_TARGET_LUFS,
  LOUDNESS_NORMALIZE_DEFAULT,
} from "@/lib/soundboardConfig"
import type { Slot, Target, ManipulationMode } from "@/lib/soundboardDb"

// Suggested condition tags — researchers add their own freely.
const CONDITION_PRESETS = [
  "unconverted",
  "same_voice_vc",
  "male_shift",
  "female_shift",
  "control",
]

function fmtMs(ms: number): string {
  if (!ms) return "—"
  return `${(ms / 1000).toFixed(2)}s`
}

export function ConfigureSoundboard() {
  const sb = useSoundboard()
  const [newSlotLabel, setNewSlotLabel] = useState("")
  const [newSlotCondition, setNewSlotCondition] = useState(CONDITION_PRESETS[0])
  const [busy, setBusy] = useState(false)
  const [feedback, setFeedback] = useState<string | null>(null)
  const importInputRef = useRef<HTMLInputElement | null>(null)
  // Global loudness-normalization defaults applied to every bake (P3), so all
  // conditions land at the same playback level.
  const [normalizeLoudness, setNormalizeLoudness] = useState(LOUDNESS_NORMALIZE_DEFAULT)
  const [targetLufs, setTargetLufs] = useState(LOUDNESS_TARGET_LUFS)

  const showError = (e: unknown) => {
    // Longer timeout so errors during a fast-throwing bake are readable.
    setFeedback((e as Error)?.message || String(e))
    setTimeout(() => setFeedback(null), 15000)
  }

  const handleCreate = async () => {
    if (!newSlotLabel.trim()) return
    setBusy(true)
    try {
      await sb.createSlot(newSlotLabel.trim(), newSlotCondition)
      setNewSlotLabel("")
    } catch (e) { showError(e) } finally { setBusy(false) }
  }

  const handleExport = async () => {
    try {
      const blob = await sb.exportZip()
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `soundboard-${Date.now()}.zip`
      a.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (e) { showError(e) }
  }

  const handleImport = async (file: File) => {
    if (!confirm("Importing will REPLACE all current slots and targets. Continue?")) return
    setBusy(true)
    try { await sb.importZip(file) } catch (e) { showError(e) } finally { setBusy(false) }
  }

  if (sb.loading) {
    return <div className="p-8 flex items-center gap-2"><Spinner /> Loading soundboard…</div>
  }

  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between gap-4 flex-wrap">
        <div className="space-y-1">
          <h2 className="text-lg font-semibold">Configure Soundboard</h2>
          <p className="text-xs text-muted-foreground max-w-prose">
            Pre-record short scripted turns, bake them once into each experimental
            condition, then trigger them during a live conversation. Built for
            reproducible stimulus delivery — same WAV in, same bytes to PP.
            Format: <code>{PP_SAMPLE_RATE} Hz mono {PP_BIT_DEPTH}-bit</code>.
            Duration drift over {DURATION_DRIFT_TOLERANCE_MS} ms is flagged.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="size-3.5" /> Export
          </Button>
          <Button variant="outline" size="sm" onClick={() => importInputRef.current?.click()}>
            <Upload className="size-3.5" /> Import
          </Button>
          <input
            ref={importInputRef}
            type="file"
            accept=".zip"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void handleImport(f)
              e.currentTarget.value = ""
            }}
          />
        </div>
      </header>

      {feedback && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {feedback}
        </div>
      )}

      <TargetLibrary sb={sb} />

      {/* Bake defaults — loudness normalization (P3). Applied to every bake so
          per-condition level differences don't confound the study. */}
      <Card>
        <CardContent className="p-4 flex flex-wrap items-center gap-x-6 gap-y-2">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Bake defaults
          </div>
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              className="size-3.5 accent-primary"
              checked={normalizeLoudness}
              onChange={(e) => setNormalizeLoudness(e.target.checked)}
            />
            Normalize loudness (EBU R128)
          </label>
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            target
            <input
              type="number"
              step={0.5}
              value={targetLufs}
              disabled={!normalizeLoudness}
              onChange={(e) => setTargetLufs(Number(e.target.value) || LOUDNESS_TARGET_LUFS)}
              className="w-16 h-7 rounded border bg-background px-1.5 text-xs disabled:opacity-50"
            />
            LUFS
          </label>
          <p className="text-[10px] text-muted-foreground max-w-prose">
            Gain-matches each baked clip to a common integrated loudness (with a
            peak guard). Applies to VC, pitch/formant, and unconverted bakes —
            re-bake existing slots to apply.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4 space-y-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            New slot
          </div>
          <div className="flex flex-wrap items-end gap-2">
            <div className="flex-1 min-w-[200px] space-y-1">
              <Label className="text-[10px]">Label</Label>
              <Input
                value={newSlotLabel}
                onChange={(e) => setNewSlotLabel(e.target.value)}
                placeholder='e.g. "Hello, how are you?"'
              />
            </div>
            <div className="w-[160px] space-y-1">
              <Label className="text-[10px]">Condition</Label>
              <select
                className="h-8 w-full rounded-md border bg-background px-2 text-xs"
                value={newSlotCondition}
                onChange={(e) => setNewSlotCondition(e.target.value)}
              >
                {CONDITION_PRESETS.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <Button onClick={handleCreate} disabled={busy || !newSlotLabel.trim()}>
              {busy ? <Spinner className="size-3.5" /> : <Sparkles className="size-3.5" />}
              Add slot
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-3">
        {sb.slots.length === 0 && (
          <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
            No slots yet. Add one above to begin.
          </div>
        )}
        {sb.slots.map((slot, i) => (
          <SlotCard
            key={slot.id}
            slot={slot}
            sb={sb}
            onError={showError}
            isFirst={i === 0}
            isLast={i === sb.slots.length - 1}
            normalizeLoudness={normalizeLoudness}
            targetLufs={targetLufs}
          />
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
//  Target library
// ---------------------------------------------------------------------------

function TargetLibrary({ sb }: { sb: ReturnType<typeof useSoundboard> }) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [uploadLabel, setUploadLabel] = useState("")
  const [busy, setBusy] = useState(false)
  // Track WHICH target is currently playing so the button can toggle icons.
  const [playingId, setPlayingId] = useState<string | null>(null)
  const stopRef = useRef<(() => void) | null>(null)

  const handleFile = async (file: File) => {
    if (!uploadLabel.trim()) {
      alert("Set a label for the target first.")
      return
    }
    setBusy(true)
    try {
      await sb.addUploadedTarget(uploadLabel.trim(), file)
      setUploadLabel("")
    } catch (e) {
      alert((e as Error).message)
    } finally { setBusy(false) }
  }

  const togglePreview = async (t: Target) => {
    if (!t.wav) return
    // Second click on the same target = pause.
    if (playingId === t.id) {
      stopRef.current?.()
      return
    }
    // Different target playing — stop it first.
    stopRef.current?.()
    setPlayingId(t.id)
    const handle = await sb.previewBlob(t.wav, () => {
      setPlayingId((cur) => (cur === t.id ? null : cur))
      stopRef.current = null
    })
    stopRef.current = handle.stop
  }

  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Target voices
        </div>
        <div className="flex flex-wrap gap-2">
          {sb.targets.map((t) => (
            <div key={t.id} className="rounded-md border px-2 py-1.5 flex items-center gap-2 text-xs">
              <span className="font-medium">{t.label}</span>
              {t.builtin && <Badge variant="secondary" className="text-[9px] h-4">builtin</Badge>}
              {t.wav && (
                <Button variant="ghost" size="xs" onClick={() => togglePreview(t)}>
                  {playingId === t.id
                    ? <Pause className="size-3" />
                    : <Play className="size-3" />}
                </Button>
              )}
              {!t.builtin && (
                <Button variant="ghost" size="xs" onClick={() => sb.removeTarget(t.id)}>
                  <Trash2 className="size-3 text-destructive" />
                </Button>
              )}
            </div>
          ))}
        </div>
        <div className="flex flex-wrap items-end gap-2 pt-1">
          <div className="flex-1 min-w-[160px] space-y-1">
            <Label className="text-[10px]">Add target — label</Label>
            <Input
              value={uploadLabel}
              onChange={(e) => setUploadLabel(e.target.value)}
              placeholder="e.g. NATM3 (male)"
            />
          </div>
          <Button
            variant="outline"
            disabled={busy || !uploadLabel.trim()}
            onClick={() => inputRef.current?.click()}
          >
            {busy ? <Spinner className="size-3.5" /> : <Upload className="size-3.5" />}
            Upload WAV
          </Button>
          <input
            ref={inputRef}
            type="file"
            accept="audio/wav,audio/*"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void handleFile(f)
              e.currentTarget.value = ""
            }}
          />
        </div>
        <p className="text-[10px] text-muted-foreground">
          Targets are the voice identity X-VC converts your recording into
          — e.g. upload a WAV of a female speaker to make your voice sound
          female to PP. The soundboard never touches PP's own response voice.
          Upload at least one WAV to enable VC-mode bakes.
        </p>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
//  Per-slot card — record, bake, preview, delete
// ---------------------------------------------------------------------------

function SlotCard({
  slot,
  sb,
  onError,
  isFirst,
  isLast,
  normalizeLoudness,
  targetLufs,
}: {
  slot: Slot
  sb: ReturnType<typeof useSoundboard>
  onError: (e: unknown) => void
  isFirst: boolean
  isLast: boolean
  normalizeLoudness: boolean
  targetLufs: number
}) {
  const [recording, setRecording] = useState(false)
  const [recElapsedMs, setRecElapsedMs] = useState(0)
  const recTimerRef = useRef<number | null>(null)
  const uploadRef = useRef<HTMLInputElement | null>(null)

  const [mode, setMode] = useState<ManipulationMode>(slot.manipulation)
  // VC bakes need a target WAV (built-in targets are name-only stubs and
  // can't be used for VC yet). Default to the first uploaded target so the
  // <select> value matches an actual <option>.
  const vcCandidates = sb.targets.filter((t) => !!t.wav)
  const [targetId, setTargetId] = useState<string>(slot.targetId || vcCandidates[0]?.id || "")
  // VC engine: X-VC is the production engine (duration-preserving; the only
  // one valid for the frozen soundboard audit). Seed-VC is exploratory.
  const [vcEngine, setVcEngine] = useState<"xvc" | "seedvc">(
    slot.engine === "seedvc" ? "seedvc" : "xvc")
  const [semitones, setSemitones] = useState<number>(slot.pitchSemitones ?? PITCH_FORMANT_DEFAULTS.semitones)
  const [formantShift, setFormantShift] = useState<number>(slot.formantShift ?? PITCH_FORMANT_DEFAULTS.formantShift)
  const [baking, setBaking] = useState(false)
  // "playing" tracks which clip (raw|baked) is currently sounding, so the
  // button icon can toggle Play↔Pause. previewRef.current.stop() is called
  // both to pause manually and by src.onended when the clip finishes.
  const previewRef = useRef<{ stop: () => void } | null>(null)
  const [playing, setPlaying] = useState<"raw" | "baked" | null>(null)
  const [label, setLabel] = useState(slot.label)
  const [condition, setCondition] = useState(slot.condition)

  // Keep local state in sync if a refresh swapped slot data underneath us.
  useEffect(() => {
    setLabel(slot.label)
    setCondition(slot.condition)
    setMode(slot.manipulation)
    setVcEngine(slot.engine === "seedvc" ? "seedvc" : "xvc")
    setSemitones(slot.pitchSemitones ?? PITCH_FORMANT_DEFAULTS.semitones)
    setFormantShift(slot.formantShift ?? PITCH_FORMANT_DEFAULTS.formantShift)
    const cands = sb.targets.filter((t) => !!t.wav)
    setTargetId(slot.targetId || cands[0]?.id || "")
  }, [slot.id, slot.updatedAt]) // intentionally narrow

  const startRec = async () => {
    try {
      await sb.startRecording()
      setRecording(true)
      const startedAt = performance.now()
      recTimerRef.current = window.setInterval(() => {
        setRecElapsedMs(performance.now() - startedAt)
      }, 50)
    } catch (e) { onError(e) }
  }

  const stopRec = async () => {
    if (recTimerRef.current) { clearInterval(recTimerRef.current); recTimerRef.current = null }
    setRecording(false)
    setRecElapsedMs(0)
    try {
      await sb.stopRecording(slot.id)
    } catch (e) { onError(e) }
  }

  const handleBake = async () => {
    setBaking(true)
    try {
      await sb.bakeSlot(slot.id, {
        mode,
        targetId: mode === "vc" ? targetId : undefined,
        engine: mode === "vc" ? vcEngine : undefined,
        pitchSemitones: mode === "pitch_formant" ? semitones : undefined,
        formantShift: mode === "pitch_formant" ? formantShift : undefined,
        normalize: normalizeLoudness,
        targetLufs,
      })
    } catch (e) { onError(e) } finally { setBaking(false) }
  }

  const preview = async (which: "raw" | "baked") => {
    // Second click on the same clip = pause.
    if (playing === which) {
      previewRef.current?.stop()
      return
    }
    // Switch to a different clip — stop the current one first.
    previewRef.current?.stop()
    const blob = which === "raw" ? slot.raw : (slot.baked ?? slot.raw)
    if (!blob) return
    setPlaying(which)
    const handle = await sb.previewBlob(blob, () => {
      setPlaying((cur) => (cur === which ? null : cur))
      previewRef.current = null
    })
    previewRef.current = handle
  }

  const saveMeta = async () => {
    if (label === slot.label && condition === slot.condition) return
    try { await sb.updateSlot({ ...slot, label, condition }) } catch (e) { onError(e) }
  }

  const drift = slot.driftMs ?? 0
  const driftOk = Math.abs(drift) <= DURATION_DRIFT_TOLERANCE_MS

  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        {/* meta row */}
        <div className="flex flex-wrap items-end gap-2">
          <div className="flex-1 min-w-[200px] space-y-1">
            <Label className="text-[10px]">Label</Label>
            <Input value={label} onChange={(e) => setLabel(e.target.value)} onBlur={saveMeta} />
          </div>
          <div className="w-[160px] space-y-1">
            <Label className="text-[10px]">Condition</Label>
            <Input
              list={`cond-${slot.id}`}
              value={condition}
              onChange={(e) => setCondition(e.target.value)}
              onBlur={saveMeta}
            />
            <datalist id={`cond-${slot.id}`}>
              {CONDITION_PRESETS.map((c) => <option key={c} value={c} />)}
            </datalist>
          </div>
          <div className="flex flex-col">
            <Button
              variant="ghost"
              size="xs"
              className="h-4 px-1"
              disabled={isFirst}
              onClick={() => sb.moveSlot(slot.id, "up")}
              title="Move slot up"
            >
              <ChevronUp className="size-3" />
            </Button>
            <Button
              variant="ghost"
              size="xs"
              className="h-4 px-1"
              disabled={isLast}
              onClick={() => sb.moveSlot(slot.id, "down")}
              title="Move slot down"
            >
              <ChevronDown className="size-3" />
            </Button>
          </div>
          <Button
            variant="ghost"
            size="sm"
            disabled={!slot.raw}
            onClick={() => void sb.duplicateSlot(slot.id).catch(onError)}
            title={slot.raw
              ? "Duplicate slot (same source audio, no bake) — use for matched natural/converted pairs"
              : "Record or upload audio first"}
          >
            <CopyPlus className="size-3.5" />
          </Button>
          <Button variant="ghost" size="sm" onClick={() => sb.removeSlot(slot.id)}>
            <Trash2 className="size-3.5 text-destructive" />
          </Button>
        </div>

        {/* record / upload row */}
        <div className="flex items-center gap-2 flex-wrap">
          {!recording ? (
            <Button variant="outline" size="sm" onClick={startRec}>
              <Mic className="size-3.5" /> {slot.raw ? "Re-record" : "Record"}
            </Button>
          ) : (
            <Button variant="default" size="sm" onClick={stopRec} className="bg-red-600 hover:bg-red-500">
              <Square className="size-3.5" /> Stop · {(recElapsedMs/1000).toFixed(1)}s
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            disabled={recording}
            onClick={() => uploadRef.current?.click()}
            title="Use a pre-recorded WAV instead of recording. Auto-conformed to PP's sample rate."
          >
            <FileAudio className="size-3.5" /> Upload
          </Button>
          <input
            ref={uploadRef}
            type="file"
            accept="audio/wav,audio/*"
            className="hidden"
            onChange={async (e) => {
              const f = e.target.files?.[0]
              e.currentTarget.value = ""
              if (!f) return
              try { await sb.loadRawFromFile(slot.id, f) } catch (e) { onError(e) }
            }}
          />
          <span className="text-[11px] text-muted-foreground">
            raw: <span className="font-mono">{fmtMs(slot.rawDurationMs)}</span>
          </span>
          {slot.raw && (
            <>
              <Button variant="ghost" size="xs" onClick={() => preview("raw")}>
                {playing === "raw" ? <Pause className="size-3" /> : <Play className="size-3" />}
                raw
              </Button>
              <Button
                variant="ghost"
                size="xs"
                onClick={() => sb.downloadSlotAudio(slot, "raw")}
                title="Download raw WAV"
              >
                <Download className="size-3" />
              </Button>
            </>
          )}
        </div>

        {/* bake row */}
        <div className="rounded-md border bg-muted/30 p-3 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <ModeTab v="unconverted" mode={mode} setMode={setMode}>Unconverted</ModeTab>
            <ModeTab v="vc" mode={mode} setMode={setMode}>VC</ModeTab>
            <ModeTab v="pitch_formant" mode={mode} setMode={setMode}>Pitch + Formant</ModeTab>
          </div>

          {mode === "vc" && (
            <div className="space-y-1">
              <div className="flex flex-wrap items-end gap-2">
                <div className="space-y-1">
                  <Label className="text-[10px]">
                    Target voice — what to convert YOUR voice INTO
                  </Label>
                  {vcCandidates.length === 0 ? (
                    <p className="text-[11px] text-amber-500 max-w-xs">
                      No uploaded target voices yet. Add one in the "Target voices"
                      section above (drag or upload a WAV of the speaker whose
                      voice identity you want to imitate).
                    </p>
                  ) : (
                    <select
                      className="h-8 rounded-md border bg-background px-2 text-xs"
                      value={targetId}
                      onChange={(e) => setTargetId(e.target.value)}
                    >
                      {vcCandidates.map((t) => (
                        <option key={t.id} value={t.id}>{t.label}</option>
                      ))}
                    </select>
                  )}
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">Engine</Label>
                  <select
                    className="h-8 rounded-md border bg-background px-2 text-xs"
                    value={vcEngine}
                    onChange={(e) => setVcEngine(e.target.value as "xvc" | "seedvc")}
                  >
                    <option value="xvc">X-VC (production)</option>
                    <option value="seedvc">Seed-VC (exploratory)</option>
                  </select>
                </div>
              </div>
              <p className="text-[10px] text-muted-foreground">
                {vcEngine === "xvc"
                  ? "VC-mode bake converts your recording with X-VC before the " +
                    "conversation. The stored 24 kHz clip goes directly to " +
                    "PersonaPlex, with live VC kept off."
                  : "Seed-VC is exploratory only: it VAD-trims and can re-time " +
                    "the clip, so a duration-drift flag is expected, and " +
                    "Seed-VC slots are not valid for the frozen audit (the " +
                    "audit characterizes the production X-VC manipulation)."}
              </p>
            </div>
          )}

          {mode === "pitch_formant" && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-3">
                <NumberSlider
                  label="Pitch (semitones)" min={-12} max={12} step={0.5}
                  value={semitones} onChange={setSemitones}
                />
                <NumberSlider
                  label="Formant shift" min={0.5} max={2.0} step={0.05}
                  value={formantShift} onChange={setFormantShift}
                />
              </div>
              <div className="flex items-center gap-2 text-[10px]">
                <span className="text-muted-foreground">Presets:</span>
                {Object.entries(PITCH_FORMANT_PRESETS).map(([name, p]) => (
                  <button
                    key={name}
                    className="rounded border px-1.5 py-0.5 hover:bg-muted"
                    onClick={() => { setSemitones(p.semitones); setFormantShift(p.formantShift) }}
                  >
                    {name}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between flex-wrap gap-2">
            <Button
              variant="default"
              size="sm"
              onClick={handleBake}
              disabled={
                baking ||
                !slot.raw ||
                (mode === "vc" && (!targetId || vcCandidates.length === 0))
              }
              title={
                !slot.raw ? "Record or upload raw audio first" :
                (mode === "vc" && vcCandidates.length === 0) ? "Upload a target voice first" :
                "Run the bake"
              }
            >
              {baking ? <Spinner className="size-3.5" /> : <Wand2 className="size-3.5" />}
              Bake
            </Button>
            <div className="flex items-center gap-2 text-[11px]">
              <span className="text-muted-foreground">
                baked: <span className="font-mono">{fmtMs(slot.bakedDurationMs)}</span>
              </span>
              {slot.bakeTimestamp && (
                driftOk ? (
                  <Badge variant="secondary" className="text-[9px] h-4">
                    <CheckCircle2 className="size-3" /> drift {drift.toFixed(1)} ms
                  </Badge>
                ) : (
                  <Badge variant="destructive" className="text-[9px] h-4">
                    <AlertTriangle className="size-3" /> drift {drift.toFixed(1)} ms
                  </Badge>
                )
              )}
              {slot.bakeTimestamp && slot.normalized && (
                <Badge variant="secondary" className="text-[9px] h-4" title="Loudness-normalized (EBU R128)">
                  {slot.measuredLufs != null ? `${slot.measuredLufs.toFixed(1)}` : `${slot.targetLufs}`} LUFS
                </Badge>
              )}
              {(slot.baked || (mode === "unconverted" && slot.raw)) && (
                <>
                  <Button variant="ghost" size="xs" onClick={() => preview("baked")}>
                    {playing === "baked" ? <Pause className="size-3" /> : <Play className="size-3" />}
                    baked
                  </Button>
                  {slot.baked && (
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={() => sb.downloadSlotAudio(slot, "baked")}
                      title="Download baked WAV"
                    >
                      <Download className="size-3" />
                    </Button>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function ModeTab({
  v, mode, setMode, children,
}: {
  v: ManipulationMode
  mode: ManipulationMode
  setMode: (m: ManipulationMode) => void
  children: React.ReactNode
}) {
  const active = v === mode
  return (
    <button
      className={`rounded-md border px-2 py-1 text-xs ${active ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
      onClick={() => setMode(v)}
    >
      {children}
    </button>
  )
}

function NumberSlider({
  label, min, max, step, value, onChange,
}: {
  label: string
  min: number
  max: number
  step: number
  value: number
  onChange: (v: number) => void
}) {
  return (
    <div className="flex flex-col gap-1 min-w-[180px]">
      <Label className="text-[10px]">{label}: <span className="font-mono">{value.toFixed(2)}</span></Label>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-primary"
      />
    </div>
  )
}
