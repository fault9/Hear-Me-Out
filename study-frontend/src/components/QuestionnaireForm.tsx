import { useRef, useState } from "react"
import { Button } from "@shared/ui/button"
import { cn } from "@shared/lib/utils"

export interface QItem {
  id: string
  type: "text" | "textarea" | "number" | "radio" | "select" | "checkbox" | "switch" | "scale" | "audio_playback"
  label: string
  required?: boolean
  options?: string[]
  options_source?: string
  allow_other?: boolean
  other_label?: string
  extra_options?: string[]
  show_if?: { field: string; in: any[] }
  min?: number
  max?: number
  min_label?: string
  max_label?: string
  placeholder?: string
  scenario_order?: number   // audio_playback: which scenario's recording
  track?: string            // audio_playback: merged | participant
  max_seconds?: number      // audio_playback: only the first N seconds are playable
  max_plays?: number        // audio_playback: allow at most M plays
}

type Answers = Record<string, any>
const OTHER = "__other__"

function resolveOptions(item: QItem, scenarioOptions?: string[]): string[] {
  const base = item.options_source === "scenarios" ? (scenarioOptions || []) : (item.options || [])
  return item.options_source === "scenarios" ? [...base, ...(item.options || [])] : base
}

function visible(item: QItem, answers: Answers): boolean {
  const s = item.show_if
  if (!s || !s.field) return true
  const v = answers[s.field]
  if (Array.isArray(v)) return v.some(x => s.in.includes(x))
  return s.in.includes(v)
}

function isAnswered(item: QItem, answers: Answers): boolean {
  const v = answers[item.id]
  if (item.type === "switch") return v === true
  if (item.type === "checkbox") return Array.isArray(v) && v.length > 0
  return v !== undefined && v !== null && v !== ""
}

function fieldError(item: QItem, answers: Answers): string | null {
  if (item.type === "audio_playback" || !visible(item, answers)) return null
  const v = answers[item.id]
  if (item.required && !isAnswered(item, answers)) return "This question is required."
  // "Other" selected but no text
  const otherText = answers[item.id + "__other"]
  const otherSelected = item.type === "checkbox" ? Array.isArray(v) && v.includes(OTHER) : v === OTHER
  if (item.allow_other && otherSelected && !(otherText || "").trim()) return "Please specify."
  if (item.type === "number" && v !== undefined && v !== null && v !== "") {
    const n = Number(v)
    if (Number.isNaN(n)) return "Enter a number."
    if (item.min !== undefined && n < item.min) return `Must be ≥ ${item.min}.`
    if (item.max !== undefined && n > item.max) return `Must be ≤ ${item.max}.`
  }
  return null
}

export function QuestionnaireForm({
  title, items, onSubmit, submitLabel = "Continue", busy = false, scenarioOptions, playbackUrl,
}: {
  title: string
  items: QItem[]
  onSubmit: (answers: Answers) => void
  submitLabel?: string
  busy?: boolean
  scenarioOptions?: string[]
  playbackUrl?: (item: QItem) => string
}) {
  const [answers, setAnswers] = useState<Answers>({})
  const [showErrors, setShowErrors] = useState(false)
  const set = (id: string, v: unknown) => setAnswers(a => ({ ...a, [id]: v }))

  const submit = () => {
    if (items.some(i => fieldError(i, answers))) { setShowErrors(true); return }
    // Flatten: replace the "other" sentinel with the typed text.
    const out: Answers = {}
    for (const item of items) {
      if (item.type === "audio_playback" || !visible(item, answers)) continue
      const v = answers[item.id]
      const otherText = answers[item.id + "__other"] || ""
      if (item.type === "checkbox" && Array.isArray(v)) out[item.id] = v.map(x => x === OTHER ? otherText : x)
      else if (v === OTHER) out[item.id] = otherText
      else out[item.id] = v
    }
    onSubmit(out)
  }

  return (
    <div className="mx-auto w-full max-w-2xl">
      <h2 className="mb-5 text-xl font-semibold tracking-tight">{title}</h2>
      <div className="flex flex-col gap-6">
        {items.filter(it => visible(it, answers)).map(item => {
          const err = showErrors ? fieldError(item, answers) : null
          return (
            <div key={item.id} className={cn("rounded-lg border p-4", err && "border-destructive")}>
              {item.type !== "audio_playback" && (
                <label className="mb-3 block text-sm font-medium">
                  {item.label}{item.required && <span className="text-destructive"> *</span>}
                </label>
              )}
              <QuestionInput item={item} answers={answers} set={set}
                scenarioOptions={scenarioOptions} playbackUrl={playbackUrl} />
              {err && <p className="mt-2 text-xs text-destructive">{err}</p>}
            </div>
          )
        })}
        {items.length === 0 && <p className="text-sm text-muted-foreground">No questions.</p>}
      </div>
      <div className="mt-6 flex justify-end">
        <Button onClick={submit} disabled={busy}>{busy ? "Saving…" : submitLabel}</Button>
      </div>
    </div>
  )
}

function OtherBox({ item, answers, set }: { item: QItem; answers: Answers; set: (id: string, v: unknown) => void }) {
  return (
    <input type="text" className="mt-2 w-full rounded-md border bg-background px-3 py-2 text-sm"
      placeholder={item.other_label || "Please specify…"} value={answers[item.id + "__other"] || ""}
      onChange={e => set(item.id + "__other", e.target.value)} />
  )
}

function QuestionInput({ item, answers, set, scenarioOptions, playbackUrl }: {
  item: QItem; answers: Answers; set: (id: string, v: unknown) => void
  scenarioOptions?: string[]; playbackUrl?: (item: QItem) => string
}) {
  const value = answers[item.id]

  if (item.type === "audio_playback") {
    const src = playbackUrl?.(item)
    const limited = item.max_seconds != null || item.max_plays != null
    return (
      <div className="flex flex-col gap-2">
        {item.label && <p className="text-sm">{item.label}</p>}
        {!src
          ? <p className="text-sm text-muted-foreground">Recording not available.</p>
          : limited
            ? <LimitedAudio src={src} maxSeconds={item.max_seconds} maxPlays={item.max_plays} />
            : <audio controls src={src} className="w-full">Your browser cannot play this audio.</audio>}
      </div>
    )
  }

  if (item.type === "scale") {
    const min = item.min ?? 1, max = item.max ?? 7
    const nums = Array.from({ length: Math.max(0, max - min + 1) }, (_, i) => min + i)
    return (
      <div>
        <div className="flex flex-wrap gap-2">
          {nums.map(n => (
            <button key={n} type="button" onClick={() => set(item.id, n)}
              className={cn("h-10 w-10 rounded-md border text-sm font-medium transition-colors",
                value === n ? "border-primary bg-primary text-primary-foreground" : "hover:bg-accent")}>{n}</button>
          ))}
        </div>
        {(item.min_label || item.max_label) && (
          <div className="mt-1 flex justify-between text-xs text-muted-foreground">
            <span>{item.min_label}</span><span>{item.max_label}</span>
          </div>
        )}
        {(item.extra_options || []).length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {item.extra_options!.map(opt => (
              <button key={opt} type="button" onClick={() => set(item.id, opt)}
                className={cn("rounded-md border px-3 py-1.5 text-sm transition-colors",
                  value === opt ? "border-primary bg-primary/10" : "hover:bg-accent")}>{opt}</button>
            ))}
          </div>
        )}
      </div>
    )
  }

  if (item.type === "radio") {
    const opts = resolveOptions(item, scenarioOptions)
    const all = item.allow_other ? [...opts, OTHER] : opts
    return (
      <div className="flex flex-col gap-2">
        {all.map(opt => (
          <button key={opt} type="button" onClick={() => set(item.id, opt)}
            className={cn("rounded-md border px-3 py-2 text-left text-sm transition-colors",
              value === opt ? "border-primary bg-primary/10" : "hover:bg-accent")}>
            {opt === OTHER ? (item.other_label || "Other") : opt}
          </button>
        ))}
        {item.allow_other && value === OTHER && <OtherBox item={item} answers={answers} set={set} />}
      </div>
    )
  }

  if (item.type === "checkbox") {
    const opts = resolveOptions(item, scenarioOptions)
    const all = item.allow_other ? [...opts, OTHER] : opts
    const arr: string[] = Array.isArray(value) ? value : []
    const toggle = (opt: string) => set(item.id, arr.includes(opt) ? arr.filter(o => o !== opt) : [...arr, opt])
    return (
      <div className="flex flex-col gap-2">
        {all.map(opt => (
          <button key={opt} type="button" onClick={() => toggle(opt)}
            className={cn("flex items-center gap-2 rounded-md border px-3 py-2 text-left text-sm transition-colors",
              arr.includes(opt) ? "border-primary bg-primary/10" : "hover:bg-accent")}>
            <span className={cn("flex h-4 w-4 items-center justify-center rounded border",
              arr.includes(opt) && "border-primary bg-primary text-primary-foreground")}>{arr.includes(opt) ? "✓" : ""}</span>
            {opt === OTHER ? (item.other_label || "Other") : opt}
          </button>
        ))}
        {item.allow_other && arr.includes(OTHER) && <OtherBox item={item} answers={answers} set={set} />}
      </div>
    )
  }

  if (item.type === "select") {
    const opts = resolveOptions(item, scenarioOptions)
    return (
      <select className="w-full rounded-md border bg-background px-3 py-2 text-sm"
        value={(value as string) ?? ""} onChange={e => set(item.id, e.target.value)}>
        <option value="" disabled>Select…</option>
        {opts.map(opt => <option key={opt} value={opt}>{opt}</option>)}
      </select>
    )
  }

  if (item.type === "switch") {
    return (
      <button type="button" onClick={() => set(item.id, value !== true)}
        className={cn("flex items-center gap-2 rounded-md border px-3 py-2 text-sm transition-colors",
          value === true ? "border-primary bg-primary/10" : "hover:bg-accent")}>
        <span className={cn("flex h-4 w-4 items-center justify-center rounded border",
          value === true && "border-primary bg-primary text-primary-foreground")}>{value === true ? "✓" : ""}</span>
        Yes
      </button>
    )
  }

  if (item.type === "number") {
    return (
      <input type="number" className="w-40 rounded-md border bg-background px-3 py-2 text-sm"
        value={(value as string) ?? ""} min={item.min} max={item.max}
        placeholder={item.placeholder} onChange={e => set(item.id, e.target.value)} />
    )
  }

  if (item.type === "textarea") {
    return (
      <textarea className="min-h-[90px] w-full rounded-md border bg-background px-3 py-2 text-sm"
        value={(value as string) ?? ""} placeholder={item.placeholder ?? "Your answer…"}
        onChange={e => set(item.id, e.target.value)} />
    )
  }

  return (
    <input type="text" className="w-full rounded-md border bg-background px-3 py-2 text-sm"
      value={(value as string) ?? ""} placeholder={item.placeholder ?? "Your answer…"}
      onChange={e => set(item.id, e.target.value)} />
  )
}

// Playback capped to the first `maxSeconds` and at most `maxPlays` plays. Uses a custom
// Play control (no native seek/replay) so the limits can't be bypassed.
function LimitedAudio({ src, maxSeconds, maxPlays }: { src: string; maxSeconds?: number; maxPlays?: number }) {
  const ref = useRef<HTMLAudioElement>(null)
  const [plays, setPlays] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [t, setT] = useState(0)
  const limitReached = maxPlays != null && plays >= maxPlays

  const play = () => {
    const a = ref.current
    if (!a || playing || limitReached) return
    a.currentTime = 0
    a.play().then(() => { setPlaying(true); setPlays(p => p + 1) }).catch(() => {})
  }
  const onTime = () => {
    const a = ref.current; if (!a) return
    if (maxSeconds != null && a.currentTime >= maxSeconds) { a.pause(); a.currentTime = 0; setPlaying(false); setT(0); return }
    setT(a.currentTime)
  }
  const stop = () => { setPlaying(false); setT(0) }
  const pct = maxSeconds ? Math.min(100, (t / maxSeconds) * 100) : (playing ? 50 : 0)

  return (
    <div className="flex flex-col gap-1.5">
      <audio ref={ref} src={src} onTimeUpdate={onTime} onEnded={stop} onPause={() => setPlaying(false)} preload="metadata" className="hidden" />
      <div className="flex items-center gap-3">
        <Button type="button" size="sm" disabled={playing || limitReached} onClick={play}>
          {playing ? "Playing…" : "▶ Play"}
        </Button>
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
          <div className="h-full rounded-full bg-primary transition-[width] duration-100" style={{ width: `${pct}%` }} />
        </div>
      </div>
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{maxSeconds != null ? `First ${maxSeconds}s only` : ""}</span>
        <span>{maxPlays != null && (limitReached ? "Playback limit reached" : `${maxPlays - plays}/${maxPlays} plays left`)}</span>
      </div>
    </div>
  )
}
