import { useCallback, useEffect, useRef, useState } from "react"
import { Button } from "@shared/ui/button"
import { Input } from "@shared/ui/input"
import { Badge } from "@shared/ui/badge"
import { adminApi } from "@/api"

// Padding around the event so the reviewer hears the approach and the
// aftermath, not a clipped fragment.
const WINDOW_PAD_MS = 2000

type Verdict = Record<string, string | null>
type Event = Record<string, any>
type Utterance = {
  id: string; speaker: string; text: string
  start_ms: number | null; end_ms: number | null
}

const QUESTIONS: { key: string; label: string; hint?: string }[] = [
  { key: "verified_overlap",
    label: "Real simultaneous speech (not noise or leakage)?", hint: "1 / 2" },
  { key: "verified_participant_barge_in", label: "Participant barge-in?" },
  { key: "verified_assistant_premature_onset", label: "Assistant premature onset?" },
  { key: "successful_assistant_yielding", label: "Assistant yielded successfully?" },
  { key: "disruptive_assistant_interruption", label: "Disruptive assistant interruption?" },
]

function eventWindow(event: Event): [number, number] {
  const values = ["overlap_start_ms", "overlap_end_ms", "participant_onset_ms",
                  "participant_offset_ms", "assistant_onset_ms", "assistant_offset_ms"]
    .map((key) => parseFloat(event[key])).filter((v) => !Number.isNaN(v))
  if (!values.length) return [0, 5000]
  return [Math.min(...values) - WINDOW_PAD_MS, Math.max(...values) + WINDOW_PAD_MS]
}

export function ReviewPanel({ token, studyId }: { token: string; studyId: number }) {
  const [reviewer, setReviewer] = useState(localStorage.getItem("review_initials") || "")
  const [started, setStarted] = useState(false)
  const [events, setEvents] = useState<Event[]>([])
  const [index, setIndex] = useState(0)
  const [answers, setAnswers] = useState<Verdict>({})
  const [note, setNote] = useState("")
  const [context, setContext] = useState<Utterance[]>([])
  const [correctionMs, setCorrectionMs] = useState(0)
  const [exportName, setExportName] = useState("")
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const audioRef = useRef<Map<string, HTMLAudioElement>>(new Map())
  const stopTimers = useRef<number[]>([])

  const event = events[index]
  const done = events.filter((e) => e.verdict).length

  const load = useCallback(async () => {
    setBusy(true); setErr(null)
    try {
      const data = await adminApi.reviewQueue(token, studyId)
      setEvents(data.events || [])
      setExportName(data.export || "")
      setCorrectionMs(data.participant_capture_latency_correction_ms || 0)
      const next = (data.events || []).findIndex((e: Event) => !e.verdict)
      setIndex(next === -1 ? 0 : next)
      setStarted(true)
    } catch (e: any) { setErr(e?.message || String(e)) }
    finally { setBusy(false) }
  }, [token, studyId])

  // Audio is cached per session+track: 378 events span far fewer sessions, so
  // most items reuse an already-fetched file.
  const audioFor = useCallback(async (sessionId: string, track: string) => {
    const key = `${sessionId}:${track}`
    const cached = audioRef.current.get(key)
    if (cached) return cached
    const url = await adminApi.reviewAudio(token, studyId, sessionId, track)
    const element = new Audio(url)
    audioRef.current.set(key, element)
    return element
  }, [token, studyId])

  const stopAll = useCallback(() => {
    stopTimers.current.forEach((id) => window.clearTimeout(id))
    stopTimers.current = []
    audioRef.current.forEach((element) => element.pause())
  }, [])

  const play = useCallback(async (only?: string) => {
    if (!event) return
    stopAll()
    const [from, to] = eventWindow(event)
    const tracks = only ? [only] : ["participant_raw", "assistant"]
    for (const track of tracks) {
      try {
        const element = await audioFor(event.session_id, track)
        // Event times carry the frozen capture correction; the raw microphone
        // file is in capture time, so add it back when seeking that track.
        const offset = track === "participant_raw" ? correctionMs : 0
        element.currentTime = Math.max(0, (from + offset) / 1000)
        await element.play()
        stopTimers.current.push(window.setTimeout(() => element.pause(), to - from))
      } catch (e: any) { setErr(e?.message || String(e)) }
    }
  }, [event, audioFor, correctionMs, stopAll])

  // Load the transcript window, and warm the next item's audio so the reviewer
  // never waits on a fetch mid-pass.
  useEffect(() => {
    if (!event) return
    setAnswers({ ...(event.verdict || {}) })
    setNote(event.verdict?.verification_note || "")
    const [from, to] = eventWindow(event)
    adminApi.reviewContext(token, studyId, event.session_id, from, to)
      .then((r) => setContext(r.utterances || []))
      .catch(() => setContext([]))
    const upcoming = events[index + 1]
    if (upcoming) {
      audioFor(upcoming.session_id, "participant_raw").catch(() => {})
      audioFor(upcoming.session_id, "assistant").catch(() => {})
    }
    return stopAll
  }, [event, index, events, token, studyId, audioFor, stopAll])

  const save = useCallback(async () => {
    if (!event) return
    setBusy(true); setErr(null)
    try {
      const body = {
        ...answers, event_key: event.event_key, session_id: event.session_id,
        episode_id: event.episode_id, verifier_initials: reviewer,
        verification_note: note,
      }
      await adminApi.reviewVerdict(token, studyId, body)
      setEvents((rows) => rows.map(
        (row, i) => (i === index ? { ...row, verdict: body } : row)))
      setIndex((i) => Math.min(i + 1, events.length - 1))
    } catch (e: any) { setErr(e?.message || String(e)) }
    finally { setBusy(false) }
  }, [answers, event, events.length, index, note, reviewer, studyId, token])

  useEffect(() => {
    if (!started) return
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return
      if (e.code === "Space") { e.preventDefault(); play() }
      if (e.key === "1") setAnswers((a) => ({ ...a, verified_overlap: "1" }))
      if (e.key === "2") setAnswers((a) => ({ ...a, verified_overlap: "0" }))
      if (e.key === "Enter") save()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [started, play, save])

  if (!started) {
    return (
      <div className="max-w-md">
        <p className="mb-3 text-sm text-muted-foreground">
          Manual verification of nominated overlap, barge-in and stop-latency events.
          The voice condition is withheld — judge only what the two tracks show.
        </p>
        <label className="text-xs text-muted-foreground">Your initials</label>
        <Input value={reviewer} className="mb-3 mt-1 w-40"
          onChange={(e) => { setReviewer(e.target.value); localStorage.setItem("review_initials", e.target.value) }} />
        <Button disabled={!reviewer.trim() || busy} onClick={load}>
          {busy ? "Loading…" : "Start verification"}
        </Button>
        {err && <p className="mt-3 text-sm text-destructive">{err}</p>}
      </div>
    )
  }

  if (!event) return <p className="text-sm text-muted-foreground">Queue is empty.</p>

  const [from, to] = eventWindow(event)
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <Badge variant="secondary">{done} / {events.length} verified</Badge>
        <span className="text-muted-foreground">item {index + 1} · {exportName}</span>
        <span className="ml-auto text-xs text-muted-foreground">
          space = play · 1/2 = overlap yes/no · enter = save
        </span>
      </div>

      <div className="rounded-lg border p-3">
        <div className="grid gap-2 text-xs sm:grid-cols-4">
          {[["session", event.session_id], ["episode", event.episode_id],
            ["initiator", event.initiator],
            ["overlap (ms)", event.overlap_duration_ms || "—"],
            ["≥200 ms nominated", event.overlap_200ms_candidate],
            ["barge-in nominated", event.participant_barge_in_candidate],
            ["premature onset nominated", event.assistant_premature_onset_candidate],
            ["window (ms)", `${Math.round(from)} – ${Math.round(to)}`],
          ].map(([label, value]) => (
            <div key={String(label)}>
              <div className="font-medium">{String(label)}</div>
              <div className="font-mono text-muted-foreground">{String(value ?? "—")}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <Button size="sm" onClick={() => play()}>▶ Play both</Button>
          <Button size="sm" variant="secondary" onClick={() => play("participant_raw")}>Participant only</Button>
          <Button size="sm" variant="secondary" onClick={() => play("assistant")}>Assistant only</Button>
          <Button size="sm" variant="ghost" onClick={stopAll}>Stop</Button>
        </div>
      </div>

      <div className="max-h-56 overflow-auto rounded-lg border p-3 text-sm">
        {context.length === 0 && <p className="text-muted-foreground">No transcript in window.</p>}
        {context.map((u) => {
          const inEvent = (u.end_ms ?? 0) >= (from + WINDOW_PAD_MS)
            && (u.start_ms ?? 0) <= (to - WINDOW_PAD_MS)
          return (
            <p key={u.id} className={inEvent ? "rounded bg-primary/10 px-1" : ""}>
              <span className="font-mono text-xs text-muted-foreground">
                {u.speaker === "participant" ? "P" : "A"} {Math.round(u.start_ms ?? 0)}
              </span>{" "}
              {u.text || <em className="text-muted-foreground">(no text)</em>}
            </p>
          )
        })}
      </div>

      <div className="rounded-lg border p-3">
        {QUESTIONS.map((q) => (
          <div key={q.key} className="mb-3">
            <div className="mb-1 text-sm font-medium">
              {q.label}{q.hint && <span className="ml-2 text-xs text-muted-foreground">{q.hint}</span>}
            </div>
            <div className="flex gap-2">
              {[["1", "Yes"], ["0", "No"]].map(([value, label]) => (
                <Button key={value} size="sm"
                  variant={answers[q.key] === value ? "default" : "secondary"}
                  onClick={() => setAnswers((a) => ({ ...a, [q.key]: value }))}>
                  {label}
                </Button>
              ))}
            </div>
          </div>
        ))}
        <label className="text-xs text-muted-foreground">Note (optional)</label>
        <Input value={note} onChange={(e) => setNote(e.target.value)} className="mt-1" />
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Button size="sm" variant="ghost" disabled={index === 0}
            onClick={() => setIndex((i) => i - 1)}>← Previous</Button>
          <Button size="sm" disabled={busy} onClick={save}>Save &amp; next</Button>
          <Button size="sm" variant="secondary"
            onClick={() => setIndex((i) => Math.min(i + 1, events.length - 1))}>Skip</Button>
          {err && <span className="text-sm text-destructive">{err}</span>}
        </div>
      </div>
    </div>
  )
}
