import { useEffect, useRef, useState } from "react"
import { Button } from "@shared/ui/button"
import { cn } from "@shared/lib/utils"
import { RichText } from "@/components/RichText"

export interface QItem {
  id: string
  type: "text" | "textarea" | "number" | "radio" | "select" | "checkbox" | "switch" | "scale" | "audio_playback" | "notice"
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
  condition?: string        // audio_playback: assigned condition id
  max_duration_s?: number   // audio_playback: requested derived-clip duration
  max_plays?: number        // audio_playback: total permitted playback starts
  insert_after?: string     // scenario item: place after this shared item id
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
  if (item.type === "notice" || !visible(item, answers)) return null
  const v = answers[item.id]
  if (item.type === "audio_playback") {
    const playCount = typeof v?.play_count === "number" ? v.play_count : 0
    return item.required && playCount < 1 ? "Play the recording at least once before continuing." : null
  }
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
  const hasErrors = items.some(item => fieldError(item, answers))

  const submit = () => {
    if (hasErrors) { setShowErrors(true); return }
    setShowErrors(false)
    // Flatten: replace the "other" sentinel with the typed text.
    const out: Answers = {}
    for (const item of items) {
      if (item.type === "notice" || !visible(item, answers)) continue
      const v = answers[item.id]
      if (item.type === "audio_playback") {
        if (v !== undefined) out[item.id] = v
        continue
      }
      const otherText = answers[item.id + "__other"] || ""
      if (item.type === "checkbox" && Array.isArray(v)) out[item.id] = v.map(x => x === OTHER ? otherText : x)
      else if (v === OTHER) out[item.id] = otherText
      else out[item.id] = v
    }
    onSubmit(out)
  }

  return (
    <div className="mx-auto w-full max-w-2xl">
      <h2 className="mb-5 text-2xl font-semibold tracking-tight">{title}</h2>
      {showErrors && hasErrors && (
        <div role="alert" className="mb-5 rounded-md border border-destructive bg-destructive/10 px-4 py-3 text-base text-destructive">
          You have not filled in all required answers. Please review the highlighted questions.
        </div>
      )}
      <div className="flex flex-col gap-6">
        {items.filter(it => visible(it, answers)).map(item => {
          const err = showErrors ? fieldError(item, answers) : null
          return (
            <div key={item.id} className={cn(item.type === "notice" ? "py-1" : "rounded-lg border p-4", err && "border-destructive")}>
              {item.type !== "audio_playback" && item.type !== "notice" && (
                <label className="mb-3 block text-base font-medium leading-6">
                  {item.label}{item.required && <span className="text-destructive"> *</span>}
                </label>
              )}
              <QuestionInput item={item} answers={answers} set={set}
                scenarioOptions={scenarioOptions} playbackUrl={playbackUrl} />
              {err && <p className="mt-2 text-sm text-destructive">{err}</p>}
            </div>
          )
        })}
        {items.length === 0 && <p className="text-base text-muted-foreground">No questions.</p>}
      </div>
      <div className="mt-6 flex justify-end">
        <Button className="text-base" onClick={submit} disabled={busy}>{busy ? "Saving…" : submitLabel}</Button>
      </div>
    </div>
  )
}

function OtherBox({ item, answers, set }: { item: QItem; answers: Answers; set: (id: string, v: unknown) => void }) {
  return (
    <input type="text" className="mt-2 w-full rounded-md border bg-background px-3 py-2 text-base"
      placeholder={item.other_label || "Please specify…"} value={answers[item.id + "__other"] || ""}
      onChange={e => set(item.id + "__other", e.target.value)} />
  )
}

function QuestionInput({ item, answers, set, scenarioOptions, playbackUrl }: {
  item: QItem; answers: Answers; set: (id: string, v: unknown) => void
  scenarioOptions?: string[]; playbackUrl?: (item: QItem) => string
}) {
  const value = answers[item.id]

  if (item.type === "notice") {
    return <RichText>{item.label}</RichText>
  }

  if (item.type === "audio_playback") {
    const src = playbackUrl?.(item)
    return (
      <div className="flex flex-col gap-2">
        {item.label && <p className="text-base leading-6">{item.label}</p>}
        {src
          ? <LimitedAudioPlayback
              src={src}
              maxPlays={item.max_plays}
              onStats={stats => set(item.id, stats)}
            />
          : <p className="text-base text-muted-foreground">Recording not available.</p>}
      </div>
    )
  }

  if (item.type === "scale") {
    const min = item.min ?? 1, max = item.max ?? 7
    const nums = Array.from({ length: Math.max(0, max - min + 1) }, (_, i) => min + i)
    return (
      <div>
        <div className="grid grid-cols-7 gap-2 sm:gap-3">
          {nums.map(n => (
            <button key={n} type="button" onClick={() => set(item.id, n)}
              className={cn("h-11 w-full max-w-12 min-w-0 justify-self-center rounded-md border text-base font-medium transition-colors sm:h-12",
                value === n ? "border-primary bg-primary text-primary-foreground" : "hover:bg-accent")}>{n}</button>
          ))}
        </div>
        {(item.min_label || item.max_label) && (
          <div className="mt-3 grid grid-cols-2 gap-6 text-sm leading-5 text-muted-foreground">
            <span>{item.min_label}</span><span className="text-right">{item.max_label}</span>
          </div>
        )}
        {(item.extra_options || []).length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {item.extra_options!.map(opt => (
              <button key={opt} type="button" onClick={() => set(item.id, opt)}
                className={cn("rounded-md border px-3 py-2 text-base transition-colors",
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
            className={cn("rounded-md border px-3 py-2.5 text-left text-base leading-6 transition-colors",
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
            className={cn("flex items-center gap-2 rounded-md border px-3 py-2.5 text-left text-base leading-6 transition-colors",
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
      <select className="w-full rounded-md border bg-background px-3 py-2.5 text-base"
        value={(value as string) ?? ""} onChange={e => set(item.id, e.target.value)}>
        <option value="" disabled>Select…</option>
        {opts.map(opt => <option key={opt} value={opt}>{opt}</option>)}
      </select>
    )
  }

  if (item.type === "switch") {
    return (
      <button type="button" onClick={() => set(item.id, value !== true)}
        className={cn("flex items-center gap-2 rounded-md border px-3 py-2.5 text-base transition-colors",
          value === true ? "border-primary bg-primary/10" : "hover:bg-accent")}>
        <span className={cn("flex h-4 w-4 items-center justify-center rounded border",
          value === true && "border-primary bg-primary text-primary-foreground")}>{value === true ? "✓" : ""}</span>
        Yes
      </button>
    )
  }

  if (item.type === "number") {
    return (
      <input type="number" className="w-40 rounded-md border bg-background px-3 py-2.5 text-base"
        value={(value as string) ?? ""} min={item.min} max={item.max}
        placeholder={item.placeholder} onChange={e => set(item.id, e.target.value)} />
    )
  }

  if (item.type === "textarea") {
    return (
      <textarea className="min-h-[100px] w-full rounded-md border bg-background px-3 py-2.5 text-base"
        value={(value as string) ?? ""} placeholder={item.placeholder ?? "Your answer…"}
        onChange={e => set(item.id, e.target.value)} />
    )
  }

  return (
    <input type="text" className="w-full rounded-md border bg-background px-3 py-2.5 text-base"
      value={(value as string) ?? ""} placeholder={item.placeholder ?? "Your answer…"}
      onChange={e => set(item.id, e.target.value)} />
  )
}

function LimitedAudioPlayback({ src, maxPlays, onStats }: {
  src: string
  maxPlays?: number
  onStats: (stats: { play_count: number; completed_count: number; max_plays: number | null }) => void
}) {
  const storageKey = "hmo:playback-count:" + src
  const audioRef = useRef<HTMLAudioElement>(null)
  const activeAttempt = useRef(false)
  const [playCount, setPlayCount] = useState(() => {
    const stored = Number.parseInt(sessionStorage.getItem(storageKey) || "0", 10)
    return Number.isFinite(stored) && stored > 0 ? stored : 0
  })
  const [completedCount, setCompletedCount] = useState(0)
  const limit = maxPlays && maxPlays > 0 ? Math.floor(maxPlays) : null
  const [locked, setLocked] = useState(() => limit !== null && playCount >= limit)
  const onStatsRef = useRef(onStats)
  onStatsRef.current = onStats

  useEffect(() => {
    if (playCount > 0) {
      onStatsRef.current({ play_count: playCount, completed_count: 0, max_plays: limit })
    }
  }, [])

  const handlePlay = () => {
    if (activeAttempt.current) return
    if (limit !== null && playCount >= limit) {
      audioRef.current?.pause()
      setLocked(true)
      return
    }
    activeAttempt.current = true
    const nextCount = playCount + 1
    setPlayCount(nextCount)
    sessionStorage.setItem(storageKey, String(nextCount))
    onStats({ play_count: nextCount, completed_count: completedCount, max_plays: limit })
  }

  const handleEnded = () => {
    activeAttempt.current = false
    const nextCompleted = completedCount + 1
    setCompletedCount(nextCompleted)
    onStats({ play_count: playCount, completed_count: nextCompleted, max_plays: limit })
    if (limit !== null && playCount >= limit) setLocked(true)
  }

  if (locked) {
    return <p className="text-base text-muted-foreground">Playback limit reached ({limit} plays).</p>
  }

  const remaining = limit === null ? null : Math.max(0, limit - playCount)
  return (
    <div className="flex flex-col gap-1.5">
      <audio
        ref={audioRef}
        controls
        src={src}
        className="w-full"
        onPlay={handlePlay}
        onEnded={handleEnded}
      >
        Your browser cannot play this audio.
      </audio>
      {remaining !== null && (
        <p className="text-sm text-muted-foreground" aria-live="polite">
          {remaining === 1 ? "1 play remaining" : remaining + " plays remaining"}
        </p>
      )}
    </div>
  )
}
