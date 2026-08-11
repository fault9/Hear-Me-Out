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
// Overlaps and response gaps are the two halves of turn-taking: who spoke over
// whom, and how long the silence between turns ran. Same audio, same
// transport, different question — so they are one panel in two modes.
type Mode = "overlap" | "gap"

type Question = { key: string; label: string; hint?: string
                  dependsOn?: string; dependsValue?: string }

const QUESTIONS: Question[] = [
  { key: "verified_overlap",
    label: "Real simultaneous speech (not noise or leakage)?" },
  { key: "verified_participant_barge_in", label: "Participant barge-in?",
    hint: "a genuine attempt to take the floor — a backchannel or continuer is No",
    dependsOn: "verified_overlap" },
  { key: "verified_assistant_premature_onset", label: "Assistant premature onset?",
    hint: "the assistant started while the participant held the floor",
    dependsOn: "verified_overlap" },
  { key: "successful_assistant_yielding", label: "Assistant yielded successfully?",
    hint: "did the assistant cede the floor — descriptive, not approval",
    dependsOn: "verified_participant_barge_in" },
  // The one judgment that follows a "no": an onset that was not an attempt on
  // the floor is either a continuer or simply speech that happened to land
  // there, and the barge-in question cannot separate them.
  { key: "participant_backchannel_onset", label: "Participant backchannel?",
    hint: "a continuer (mm-hm, yeah, right) rather than speech that merely coincided",
    dependsOn: "verified_participant_barge_in", dependsValue: "0" },
  { key: "disruptive_assistant_interruption", label: "Disruptive assistant interruption?",
    hint: "did it cut the participant off or derail them",
    dependsOn: "verified_assistant_premature_onset" },
  { key: "assistant_backchannel_onset", label: "Backchannel onset?",
    hint: "the early onset was a continuer (mm-hm, okay), not an attempt to take the turn",
    dependsOn: "verified_assistant_premature_onset" },
]

const GAP_QUESTIONS: Question[] = [
  { key: "verified_positive_gap",
    label: "Real response gap (silence between two real turns)?",
    hint: "No if either boundary is noise, or if speech continues inside the gap" },
]

// A question only applies while every ancestor is "yes": yielding is only a
// judgment about a verified barge-in, disruption only about a verified
// premature onset. Anything below a "no" stays blank — never recorded as 0.
function applicable(key: string, answers: Verdict, questions: Question[]): boolean {
  const question = questions.find((q) => q.key === key)
  if (!question?.dependsOn) return true
  return answers[question.dependsOn] === (question.dependsValue ?? "1")
    && applicable(question.dependsOn, answers, questions)
}

// Blank has meant three things at once - not asked, not applicable, and
// deliberately neither - and the ambiguity turned a mid-pass change to the
// question set into an apparent change in the assistant's behaviour. An
// applicable question now has to be answered, so "neither" is recorded as
// two explicit noes rather than as absence.
function unanswered(verdict: Verdict | null | undefined,
                    questions: Question[]): string[] {
  const answers = withInapplicableCleared({ ...(verdict || {}) }, questions)
  return questions
    .filter((q) => applicable(q.key, answers, questions) && answers[q.key] == null)
    .map((q) => q.key)
}

function withInapplicableCleared(answers: Verdict, questions: Question[]): Verdict {
  const next = { ...answers }
  for (const q of questions) if (!applicable(q.key, next, questions)) next[q.key] = null
  return next
}

function eventWindow(event: Event): [number, number] {
  // Centered on the event itself: enough lead-in to hear who already held the
  // floor, not the whole pair of turns.
  const start = parseFloat(event.gap_start_ms ?? event.overlap_start_ms)
  const end = parseFloat(event.gap_end_ms ?? event.overlap_end_ms)
  if (Number.isFinite(start) && Number.isFinite(end) && end >= start)
    return [Math.max(0, start - WINDOW_PAD_MS), end + WINDOW_PAD_MS]
  const values = ["participant_onset_ms", "participant_offset_ms",
                  "assistant_onset_ms", "assistant_offset_ms"]
    .map((key) => parseFloat(event[key])).filter((v) => !Number.isNaN(v))
  if (!values.length) return [0, 5000]
  return [Math.max(0, Math.min(...values) - WINDOW_PAD_MS),
          Math.max(...values) + WINDOW_PAD_MS]
}

export function ReviewPanel({ token, studyId }: { token: string; studyId: number }) {
  const [reviewer, setReviewer] = useState(localStorage.getItem("review_initials") || "")
  const [mode, setMode] = useState<Mode>("overlap")
  const [started, setStarted] = useState(false)
  const [events, setEvents] = useState<Event[]>([])
  const [index, setIndex] = useState(0)
  const [answers, setAnswers] = useState<Verdict>({})
  const [note, setNote] = useState("")
  // A finished item is read-only until deliberately unlocked. The store is
  // append-only so nothing is destroyed, but the export reads the latest
  // record, and an accidental Enter on a completed episode would make a
  // stray keystroke the effective judgment.
  const [unlocked, setUnlocked] = useState<string | null>(null)
  // Verified gap boundaries, in timeline ms. Seeded from the detector so
  // confirming a gap unchanged records what it measured.
  const [bounds, setBounds] = useState<{ start: any; end: any }>({ start: 0, end: 0 })
  const [correctionMs, setCorrectionMs] = useState(0)
  const [exportName, setExportName] = useState("")
  const [latestExport, setLatestExport] = useState("")
  const [pass, setPass] = useState<
    { export: string | null; latest_export: string | null;
      events: number; completed: number } | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const audioRef = useRef<Map<string, HTMLAudioElement>>(new Map())
  // One transport drives both tracks in lockstep; solo is a mute, not a
  // separate player, so isolating a track never desynchronizes the position.
  const [playing, setPlaying] = useState(false)
  const [position, setPosition] = useState(0)
  const [duration, setDuration] = useState(0)
  const [solo, setSolo] = useState<"both" | "participant_raw" | "assistant">("both")
  const deck = useRef<{ p: HTMLAudioElement | null; a: HTMLAudioElement | null }>(
    { p: null, a: null })
  const stopAt = useRef<number | null>(null)

  const event = events[index]
  const questionsFor = (item: Event): Question[] =>
    (item?.gap_key ? GAP_QUESTIONS : QUESTIONS)
  const complete = (item: Event): boolean =>
    Boolean(item?.verdict) && unanswered(item.verdict, questionsFor(item)).length === 0
  const done = events.filter(complete).length
  const outstanding = events.filter((e) => e.verdict && !complete(e)).length
  const questions = mode === "gap" ? GAP_QUESTIONS : QUESTIONS
  const keyOf = (item: Event): string => item?.gap_key || item?.event_key

  // A repinned queue mixes a few new candidates into many already judged, so
  // stepping by one walks back into finished work. Advance to what is not done.
  const nextUnverified = useCallback((from: number, rows: Event[]): number => {
    for (let i = from + 1; i < rows.length; i += 1) {
      const item = rows[i]
      const list = item?.gap_key ? GAP_QUESTIONS : QUESTIONS
      if (!item.verdict || unanswered(item.verdict, list).length) return i
    }
    return Math.min(from + 1, rows.length - 1)
  }, [])

  const load = useCallback(async (next: Mode = mode) => {
    setBusy(true); setErr(null)
    try {
      const data = next === "gap"
        ? await adminApi.reviewGapQueue(token, studyId)
        : await adminApi.reviewQueue(token, studyId)
      const rows: Event[] = data.gaps || data.events || []
      setMode(next)
      setEvents(rows)
      setExportName(data.export || "")
      setLatestExport(data.latest_export || "")
      setCorrectionMs(data.participant_capture_latency_correction_ms || 0)
      const first = rows.findIndex((e: Event) => !e.verdict
        || unanswered(e.verdict, e.gap_key ? GAP_QUESTIONS : QUESTIONS).length)
      setIndex(first === -1 ? 0 : first)
      setStarted(true)
    } catch (e: any) { setErr(e?.message || String(e)) }
    finally { setBusy(false) }
  }, [token, studyId, mode])

  // Audio is cached per session+track: 378 events span far fewer sessions, so
  // most items reuse an already-fetched file.
  const audioFor = useCallback(async (sessionId: string, track: string) => {
    const key = `${sessionId}:${track}`
    const cached = audioRef.current.get(key)
    if (cached) return cached
    const url = await adminApi.reviewAudio(token, studyId, sessionId, track)
    const element = new Audio(url)
    element.preload = "auto"
    // Seeking before metadata loads is silently discarded and playback starts
    // at 0 — the wrong moment, and easy to mistake for broken audio.
    await new Promise<void>((resolve) => {
      if (element.readyState >= 1) { resolve(); return }
      element.addEventListener("loadedmetadata", () => resolve(), { once: true })
      element.addEventListener("error", () => resolve(), { once: true })
    })
    audioRef.current.set(key, element)
    return element
  }, [token, studyId])

  const pauseAll = useCallback(() => {
    deck.current.p?.pause()
    deck.current.a?.pause()
    stopAt.current = null
    setPlaying(false)
  }, [])

  // Event times carry the frozen capture correction; the raw microphone file
  // is in capture time, so it sits correction-late relative to the timeline.
  const seekTo = useCallback((t: number) => {
    const { p, a } = deck.current
    if (p) p.currentTime = Math.max(0, t + correctionMs / 1000)
    if (a) a.currentTime = Math.max(0, t)
    setPosition(t)
  }, [correctionMs])

  const playFrom = useCallback(async (t: number, until: number | null = null) => {
    const { p, a } = deck.current
    if (!p || !a) return
    seekTo(t)
    stopAt.current = until
    p.muted = solo === "assistant"
    a.muted = solo === "participant_raw"
    try { await Promise.all([p.play(), a.play()]) }
    catch (e: any) { setErr(e?.message || String(e)) }
    setPlaying(true)
  }, [seekTo, solo])

  const playEvent = useCallback(() => {
    if (!event) return
    const [from, to] = eventWindow(event)
    playFrom(from / 1000, to / 1000)
  }, [event, playFrom])

  // Load both tracks for the current item and park the transport at the
  // event window so the first play needs no seeking.
  useEffect(() => {
    deck.current = { p: null, a: null }
    setPlaying(false); setPosition(0); setDuration(0)
    if (!event) return
    let cancelled = false
    Promise.all([
      audioFor(event.session_id, "participant_raw"),
      audioFor(event.session_id, "assistant"),
    ]).then(([p, a]) => {
      if (cancelled) return
      deck.current = { p, a }
      setDuration(Math.max((p.duration || 0) - correctionMs / 1000, a.duration || 0))
      seekTo(eventWindow(event)[0] / 1000)
    }).catch((e: any) => setErr(e?.message || String(e)))
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keyOf(event), audioFor, correctionMs])

  // The participant element is the clock; the assistant follows, and a
  // window play pauses itself at the window's end.
  useEffect(() => {
    if (!playing) return
    const id = window.setInterval(() => {
      const { p, a } = deck.current
      if (!p) return
      const t = Math.max(0, p.currentTime - correctionMs / 1000)
      setPosition(t)
      if (a && Math.abs(a.currentTime - t) > 0.3) a.currentTime = t
      if (stopAt.current != null && t >= stopAt.current) pauseAll()
    }, 100)
    return () => window.clearInterval(id)
  }, [playing, correctionMs, pauseAll])

  useEffect(() => {
    const { p, a } = deck.current
    if (p) p.muted = solo === "assistant"
    if (a) a.muted = solo === "participant_raw"
  }, [solo])

  // Warm the next item's audio so the reviewer never waits on a fetch
  // mid-pass. Verification is by ear alone: no transcript is shown, because
  // interval ASR stamps stock phrases on sub-speech blips and the assistant's
  // text times run ahead of its audio - both misread as evidence.
  useEffect(() => {
    if (!event) return
    setAnswers(withInapplicableCleared({ ...(event.verdict || {}) }, questions))
    setNote(event.verdict?.verification_note || "")
    // The detector's boundaries are the starting proposal: confirming a gap
    // unchanged records exactly what it measured.
    setBounds({ start: event.verdict?.verified_gap_start_ms ?? event.gap_start_ms,
                end: event.verdict?.verified_gap_end_ms ?? event.gap_end_ms })
    setUnlocked(null)
    const upcoming = events[index + 1]
    if (upcoming) {
      audioFor(upcoming.session_id, "participant_raw").catch(() => {})
      audioFor(upcoming.session_id, "assistant").catch(() => {})
    }
    return () => { pauseAll() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keyOf(event), token, studyId])

  const save = useCallback(async () => {
    if (!event) return
    if (complete(event) && unlocked !== keyOf(event)) return
    if (unanswered(answers, questions).length) return
    setBusy(true); setErr(null)
    try {
      const start = Number(bounds.start)
      const end = Number(bounds.end)
      const body = mode === "gap" ? {
        ...answers, gap_key: event.gap_key, session_id: event.session_id,
        gap_id: event.gap_id, verifier_initials: reviewer,
        verification_note: note,
        verified_gap_start_ms: start, verified_gap_end_ms: end,
        verified_gap_duration_ms: end - start,
      } : {
        ...answers, event_key: event.event_key, session_id: event.session_id,
        episode_id: event.episode_id, verifier_initials: reviewer,
        verification_note: note,
      }
      if (mode === "gap") await adminApi.reviewGapVerdict(token, studyId, body)
      else await adminApi.reviewVerdict(token, studyId, body)
      setEvents((rows) => rows.map(
        (row, i) => (i === index ? { ...row, verdict: body } : row)))
      setIndex(nextUnverified(index, events))
    } catch (e: any) { setErr(e?.message || String(e)) }
    finally { setBusy(false) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [answers, bounds, event, events, index, mode, nextUnverified, note,
      questions, reviewer, studyId, token, unlocked])

  useEffect(() => {
    if (!started) return
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return
      if (e.code === "Space") {
        e.preventDefault()
        if (playing) { pauseAll(); return }
        // Resume mid-window; replay the window once it has finished (or the
        // position was scrubbed outside it).
        const current = events[index]
        if (!current) return
        const [from, to] = eventWindow(current)
        if (position >= to / 1000 - 0.05 || position < from / 1000 - 0.05) playEvent()
        else playFrom(position, to / 1000)
      }
      if (e.key === "Enter") save()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [started, playing, position, events, index, playEvent, playFrom, pauseAll, save])

  useEffect(() => {
    if (started) return
    let live = true
    adminApi.reviewPass(token, studyId)
      .then((p: any) => { if (live) setPass(p) })
      .catch(() => { if (live) setPass(null) })
    return () => { live = false }
  }, [started, token, studyId])

  if (!started) {
    return (
      <div className="max-w-md">
        <p className="mb-3 text-sm text-muted-foreground">
          Manual verification of the automatic turn-taking candidates. The voice
          condition is withheld — judge only what the two tracks show.
        </p>
        <div className="mb-3 flex items-center gap-2">
          {([["overlap", "Overlaps & barge-ins"],
             ["gap", "Response gaps"]] as const).map(([value, label]) => (
            <Button key={value} size="sm"
              variant={mode === value ? "default" : "secondary"}
              onClick={() => setMode(value)}>{label}</Button>
          ))}
        </div>
        <label className="text-xs text-muted-foreground">Your initials</label>
        <Input value={reviewer} className="mb-3 mt-1 w-40"
          onChange={(e) => { setReviewer(e.target.value); localStorage.setItem("review_initials", e.target.value) }} />
        {pass && (
          <div className="mb-3 rounded border p-3 text-xs text-muted-foreground">
            <div>queue: {pass.export || "none pinned"}
              {pass.events ? ` · ${pass.completed} of ${pass.events} verified` : ""}</div>
            {pass.latest_export && pass.latest_export !== pass.export ? (
              <div className="mt-2 flex items-center gap-2">
                <span>newer export available: {pass.latest_export}</span>
                <Button size="sm" variant="outline" disabled={busy}
                  onClick={async () => {
                    setBusy(true); setErr(null)
                    try {
                      const r: any = await adminApi.reviewRepin(token, studyId)
                      setPass({ export: r.export, latest_export: r.export,
                                events: r.events, completed: r.completed })
                    } catch (e: any) { setErr(e?.message || String(e)) }
                    finally { setBusy(false) }
                  }}>
                  Load newer export
                </Button>
              </div>
            ) : (
              <div className="mt-1">up to date</div>
            )}
          </div>
        )}
        <Button disabled={!reviewer.trim() || busy} onClick={() => load(mode)}>
          {busy ? "Loading…" : "Start verification"}
        </Button>
        {err && <p className="mt-3 text-sm text-destructive">{err}</p>}
      </div>
    )
  }

  if (!event) return <p className="text-sm text-muted-foreground">Queue is empty.</p>

  const [from, to] = eventWindow(event)
  const missing = unanswered(answers, questions)
  const locked = complete(event) && unlocked !== keyOf(event)
  const span = (start: any, end: any): string => {
    const a = parseFloat(start)
    const b = parseFloat(end)
    if (!Number.isFinite(a) || !Number.isFinite(b)) return "—"
    return `${(a / 1000).toFixed(1)}–${(b / 1000).toFixed(1)}s (${(b - a).toFixed(0)}ms)`
  }
  const facts: [string, any][] = mode === "gap"
    ? [["session", event.session_id], ["gap", event.gap_id],
       ["direction", `${event.from_speaker} → ${event.to_speaker}`],
       ["detected", span(event.gap_start_ms, event.gap_end_ms)]]
    : [["session", event.session_id], ["episode", event.episode_id],
       ["initiator", event.initiator],
       ["overlap at", span(event.overlap_start_ms, event.overlap_end_ms)]]
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <Badge variant="secondary">{done} / {events.length} verified</Badge>
        <span className="text-muted-foreground">item {index + 1} · {exportName}</span>
        {event?.verdict && complete(event) && (
          <Badge variant="outline">already verified</Badge>
        )}
        {outstanding > 0 && (
          <Badge variant="secondary">{outstanding} verified but missing an answer</Badge>
        )}
        {latestExport && latestExport !== exportName && (
          <Button size="sm" variant="outline" disabled={busy}
            onClick={async () => {
              setBusy(true); setErr(null)
              try {
                const r: any = await adminApi.reviewRepin(token, studyId)
                await load()
                setErr(`Queue advanced to ${r.export}: ${r.events} events, `
                  + `${r.completed} already verified.`)
              } catch (e: any) { setErr(e?.message || String(e)) }
              finally { setBusy(false) }
            }}>
            Load newer export
          </Button>
        )}
        <Button size="sm" variant="secondary" disabled={busy}
          onClick={async () => {
            setBusy(true); setErr(null)
            try {
              await adminApi.download(token, adminApi.datasetUrl(studyId),
                                      `study${studyId}_dataset.zip`)
            } catch (e: any) { setErr(e?.message || String(e)) }
            finally { setBusy(false) }
          }}>
          Download dataset
        </Button>
        <span className="ml-auto text-xs text-muted-foreground">
          space = play/pause · enter = save
        </span>
      </div>

      <div className="rounded-lg border p-3">
        <div className="grid gap-2 text-xs sm:grid-cols-4">
          {facts.map(([label, value]) => (
            <div key={String(label)}>
              <div className="font-medium">{String(label)}</div>
              <div className="font-mono text-muted-foreground">{String(value ?? "—")}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Button size="sm" disabled={!duration}
            onClick={() => (playing ? pauseAll() : playFrom(position, to / 1000))}>
            {playing ? "⏸ Pause" : "▶ Play"}
          </Button>
          <Button size="sm" variant="secondary" disabled={!duration}
            onClick={playEvent}>↺ Replay</Button>
          <input type="range" min={from / 1000} max={to / 1000} step={0.05}
            value={Math.min(Math.max(position, from / 1000), to / 1000)}
            className="min-w-40 flex-1"
            onChange={(e) => seekTo(parseFloat(e.target.value))} />
          <span className="font-mono text-xs text-muted-foreground">
            {position.toFixed(1)}s of {(from / 1000).toFixed(1)}–{(to / 1000).toFixed(1)}s
          </span>
        </div>
        <div className="mt-2 flex items-center gap-2">
          <span className="text-xs text-muted-foreground">hear</span>
          {([["both", "Both"], ["participant_raw", "Participant"],
             ["assistant", "Assistant"]] as const).map(([value, label]) => (
            <Button key={value} size="sm"
              variant={solo === value ? "default" : "secondary"}
              onClick={() => setSolo(value)}>{label}</Button>
          ))}
        </div>
      </div>

      <div className="rounded-lg border p-3">
        {questions.map((q) => {
          const enabled = applicable(q.key, answers, questions)
          return (
            <div key={q.key} className={enabled ? "mb-3" : "mb-3 opacity-40"}>
              <div className="mb-1 text-sm font-medium">
                {q.label}
                <span className="ml-2 text-xs text-muted-foreground">
                  {enabled ? q.hint : "n/a — needs “yes” above"}
                </span>
              </div>
              <div className="flex gap-2">
                {[["1", "Yes"], ["0", "No"]].map(([value, label]) => (
                  <Button key={value} size="sm" disabled={!enabled}
                    variant={answers[q.key] === value ? "default" : "secondary"}
                    onClick={() => setAnswers(
                      (a) => withInapplicableCleared({ ...a, [q.key]: value }, questions))}>
                    {label}
                  </Button>
                ))}
              </div>
            </div>
          )
        })}
        {mode === "gap" && answers.verified_positive_gap === "1" && (
          <div className="mb-3 rounded border p-2">
            <div className="mb-1 text-sm font-medium">
              Boundaries
              <span className="ml-2 text-xs text-muted-foreground">
                scrub to the moment, then set — leave as-is to confirm the detector
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="secondary"
                onClick={() => setBounds((b) => ({ ...b, start: position * 1000 }))}>
                Set start = {position.toFixed(2)}s
              </Button>
              <Button size="sm" variant="secondary"
                onClick={() => setBounds((b) => ({ ...b, end: position * 1000 }))}>
                Set end = {position.toFixed(2)}s
              </Button>
              <Button size="sm" variant="ghost"
                onClick={() => setBounds({ start: event.gap_start_ms,
                                           end: event.gap_end_ms })}>
                Reset
              </Button>
              <span className="font-mono text-xs text-muted-foreground">
                {span(bounds.start, bounds.end)}
              </span>
            </div>
          </div>
        )}
        <label className="text-xs text-muted-foreground">Note (optional)</label>
        <Input value={note} onChange={(e) => setNote(e.target.value)} className="mt-1" />
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Button size="sm" variant="ghost" disabled={index === 0}
            onClick={() => setIndex((i) => i - 1)}>← Previous</Button>
          {locked ? (
            <Button size="sm" variant="secondary"
              onClick={() => setUnlocked(keyOf(event))}>
              Re-judge this item
            </Button>
          ) : (
            <Button size="sm" disabled={busy || missing.length > 0} onClick={save}>
              Save &amp; next
            </Button>
          )}
          {!locked && missing.length > 0 && (
            <span className="text-xs text-muted-foreground">
              answer {missing.length} more to save
            </span>
          )}
          <Button size="sm" variant="secondary"
            onClick={() => setIndex(nextUnverified(index, events))}>Skip</Button>
          {err && <span className="text-sm text-destructive">{err}</span>}
        </div>
      </div>
    </div>
  )
}
