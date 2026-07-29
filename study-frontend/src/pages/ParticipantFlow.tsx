import { useCallback, useEffect, useState, type ReactNode } from "react"
import { Button } from "@shared/ui/button"
import { Input } from "@shared/ui/input"
import { Spinner } from "@shared/ui/spinner"
import { CheckCircle2 } from "lucide-react"
import { api, type EnterResult, type RunState } from "@/api"
import { QuestionnaireForm, type QItem } from "@/components/QuestionnaireForm"
import { ScenarioCall } from "@/components/ScenarioCall"
import { AudioCheck } from "@/components/AudioCheck"

type Phase =
  | "code" | "welcome"
  | "eligibility" | "ineligible" | "consent" | "declined" | "background" | "audio_check"
  | "practice_intro" | "main_intro" | "scenario" | "post"
  | "pre_playback" | "playback" | "debrief" | "final" | "completion"

function mergePostItems(shared: QItem[], scenarioItems: QItem[]): QItem[] {
  const merged = [...shared]
  for (const item of scenarioItems) {
    const anchor = item.insert_after
    const index = anchor ? merged.findIndex(candidate => candidate.id === anchor) : -1
    if (index < 0) merged.push(item)
    else merged.splice(index + 1, 0, item)
  }
  return merged
}

function clearPlaybackCounts() {
  for (let index = sessionStorage.length - 1; index >= 0; index -= 1) {
    const key = sessionStorage.key(index)
    if (key?.startsWith("hmo:playback-count:")) sessionStorage.removeItem(key)
  }
}

export function ParticipantFlow() {
  const [code, setCode] = useState("")
  const [data, setData] = useState<EnterResult | null>(null)
  const [phase, setPhase] = useState<Phase>("code")
  const [scenarioIdx, setScenarioIdx] = useState(0)
  const [postSessionId, setPostSessionId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [deadline, setDeadline] = useState<number | null>(null)
  const [now, setNow] = useState(Date.now())

  const q = (kind: keyof NonNullable<EnterResult["questionnaires"]>): QItem[] =>
    ((data?.questionnaires as any)?.[kind] ?? []) as QItem[]

  // Header run-countdown tick.
  useEffect(() => {
    if (!deadline) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [deadline])

  const remaining = deadline ? Math.max(0, Math.floor((deadline - now) / 1000)) : null
  useEffect(() => {
    if (remaining === 0 && deadline) { setDeadline(null); setPhase("welcome"); refreshRun() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [remaining])

  const doEnter = useCallback(async (c: string) => {
    setBusy(true); setError(null)
    try {
      const res = await api.enter(c)
      setData(res); setCode(c)
      setPhase("welcome")
    } catch (e: any) {
      setError(e?.message || "Invalid code")
    } finally { setBusy(false) }
  }, [])

  const refreshRun = useCallback(async () => {
    if (!code) return
    try { setData(await api.enter(code)) } catch { /* ignore */ }
  }, [code])

  // Auto-enter from ?code=
  useEffect(() => {
    const c = new URLSearchParams(window.location.search).get("code")
    if (c) doEnter(c)
  }, [doEnter])

  const handleErr = (e: any) => {
    if (e?.status === 440) { setPhase("welcome"); refreshRun(); return true }
    setError(e?.message || String(e))
    return false
  }

  // Where to land after run/start. Uses the FRESH run (not the stale enter state):
  // a restart yields an empty current_step -> consent (first phase). Per-scenario
  // engine prepare happens later inside ScenarioCall.
  const goToRunStep = useCallback((run: RunState) => {
    const step = run.current_step || {}
    const p = step.phase as Phase | undefined
    if (p === "background") { setPhase("background"); return }
    if (p === "audio_check") { setPhase("audio_check"); return }
    if (p === "practice_intro") { setScenarioIdx(0); setPhase("practice_intro"); return }
    if (p === "main_intro") {
      setScenarioIdx(Math.max(1, (step.scenario_order ?? 2) - 1))
      setPhase("main_intro")
      return
    }
    if (p === "consent") { setPhase("consent"); return }
    if (p === "eligibility") { setPhase("eligibility"); return }
    if (p === "scenario" || p === "post") {
      setScenarioIdx(Math.max(0, (step.scenario_order ?? 1) - 1))
      setPostSessionId(typeof step.session_id === "string" ? step.session_id : null)
      setPhase(p)
      return
    }
    if (p === "final") { setPhase("final"); return }
    if (p === "pre_playback" || p === "playback" || p === "debrief") { setPhase(p); return }
    setScenarioIdx(0)
    setPhase(q("eligibility").length ? "eligibility" : "consent")
  }, [data]) // eslint-disable-line react-hooks/exhaustive-deps

  const setStep = useCallback((current_step: Record<string, any>, completed: Record<string, any> = {}) => {
    if (code) api.progress(code, current_step, completed).catch(() => {})
  }, [code])

  const startRun = useCallback(async (mode: "resume" | "restart") => {
    setBusy(true); setError(null)
    try {
      const res = await api.runStart(code, mode)
      setData(d => (d ? { ...d, run: res.run } : d))
      const secs = res?.run?.remaining_seconds ?? 3600
      setDeadline(Date.now() + secs * 1000)
      if (mode === "restart") {
        clearPlaybackCounts()
        const first = q("eligibility").length ? "eligibility" : "consent"
        setScenarioIdx(0); setPostSessionId(null); setStep({ phase: first }); setPhase(first)
      } else {
        goToRunStep(res.run)
      }
    } catch (e: any) {
      setError(e?.message || "Could not start")
    } finally { setBusy(false) }
  }, [code, data, goToRunStep, setStep]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- render ----------
  if (phase === "code") {
    return (
      <Centered>
        <h1 className="text-2xl font-bold">Welcome</h1>
        <p className="text-base text-muted-foreground">Enter your participant code to begin.</p>
        <div className="flex w-full max-w-xs gap-2">
          <Input value={code} onChange={e => setCode(e.target.value.toUpperCase())}
                 className="text-base" placeholder="e.g. A7X9K2" onKeyDown={e => e.key === "Enter" && doEnter(code)} />
          <Button className="text-base" onClick={() => doEnter(code)} disabled={busy || !code}>{busy ? "…" : "Start"}</Button>
        </div>
        {error && <p className="text-base text-destructive">{error}</p>}
      </Centered>
    )
  }

  if (phase === "welcome" && data) {
    const st = data.run.status
    return (
      <Centered>
        <h1 className="text-2xl font-bold">{data.study_name}</h1>
        {st === "submitted" ? (
          <>
            <p className="text-base">You have already completed this study. Thank you!</p>
          </>
        ) : st === "in_progress" || st === "expired" ? (
          <>
            <p className="max-w-md text-center text-base leading-7 text-muted-foreground">
              {st === "expired"
                ? "Your previous session expired. You can continue where you left off (your completed scenarios are kept) or restart."
                : "You have a session in progress. Continue where you left off, or restart."}
            </p>
            <div className="flex gap-2">
              <Button onClick={() => startRun("resume")} disabled={busy}>Continue</Button>
              <Button variant="secondary" onClick={() => startRun("restart")} disabled={busy}>Restart</Button>
            </div>
          </>
        ) : (
          <>
            {data.welcome_text
              ? <div className="max-h-[60vh] max-w-2xl overflow-auto whitespace-pre-wrap text-left text-base leading-7">{data.welcome_text}</div>
              : <p className="max-w-md text-center text-base leading-7 text-muted-foreground">
                  You will complete {data.scenarios.length} conversation scenarios, each followed by a brief
                  questionnaire. You have one hour to finish.
                </p>}
            {data.estimated_duration && <p className="text-sm text-muted-foreground">Estimated duration: {data.estimated_duration}</p>}
            <Button className="text-base" onClick={() => startRun("restart")} disabled={busy}>Begin</Button>
          </>
        )}
        {error && <p className="text-base text-destructive">{error}</p>}
      </Centered>
    )
  }

  const Frame = (children: ReactNode) => (
    <div className="mx-auto flex min-h-screen max-w-4xl flex-col px-4 py-6">
      <Header remaining={remaining} />
      <div className="flex-1">{children}</div>
    </div>
  )

  if (phase === "eligibility" && data) {
    return Frame(
      <QuestionnaireForm title="Eligibility" items={q("eligibility")} busy={busy}
        onSubmit={async (ans) => {
          if (ans.eligibility_18 !== "Yes") { setPhase("ineligible"); return }
          setBusy(true)
          try {
            await api.questionnaire(null, code, "eligibility", ans)
            setStep({ phase: "consent" })
            setPhase("consent")
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  if (phase === "ineligible") {
    return <Centered><h1 className="text-2xl font-bold">Thank you for your interest</h1><p className="text-base text-muted-foreground">This study is limited to participants aged 18 or older.</p></Centered>
  }

  if (phase === "declined") {
    return <Centered><h1 className="text-2xl font-bold">Thank you</h1><p className="text-base text-muted-foreground">You have chosen not to participate. No study interaction will begin.</p></Centered>
  }

  if (phase === "background" && data) {
    return Frame(
      <QuestionnaireForm title="Background questionnaire" items={q("background")} submitLabel="Continue to audio check" busy={busy}
        onSubmit={async (ans) => {
          setBusy(true)
          try {
            await api.questionnaire(null, code, "background", ans)
            setData(await api.enter(code))
            setStep({ phase: "audio_check" })
            setPhase("audio_check")
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  if (phase === "consent" && data) {
    return Frame(
      <QuestionnaireForm title="Participant information and consent" items={q("consent")} submitLabel="Continue" busy={busy}
        onSubmit={async (ans) => {
          if (ans.consent_decision !== "I consent and wish to continue.") { setPhase("declined"); return }
          setBusy(true)
          try {
            await api.questionnaire(null, code, "consent", ans)
            setStep({ phase: "background" })
            setPhase("background")
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  if (phase === "audio_check" && data) {
    return Frame(
      <AudioCheck code={code} items={q("audio_check")}
        onDone={async (ans) => {
          setBusy(true)
          try {
            await api.questionnaire(null, code, "audio_check", ans)
            setScenarioIdx(0)
            setStep({ phase: "practice_intro" })
            setPhase("practice_intro")
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  if (phase === "practice_intro" && data) {
    return Frame(
      <TransitionScreen
        title="Practice interaction"
        text={data.practice_intro_text ||
          "You will now complete a short practice conversation to become familiar with the study tasks and controls. The practice recording is retained for technical checks but is not included in the main analysis. Read the scenario carefully, then continue when you are ready."}
        onContinue={() => {
          setScenarioIdx(0)
          setStep({ phase: "scenario", scenario_order: data.scenarios[0].scenario_order })
          setPhase("scenario")
        }}
      />
    )
  }

  if (phase === "main_intro" && data) {
    const scenario = data.scenarios[scenarioIdx]
    return Frame(
      <TransitionScreen
        title="Main study"
        text={data.main_intro_text ||
          "The practice interaction is complete. You will now begin the main study conversations. Each conversation presents a different situation and aim. Read each scenario carefully and speak naturally using your own words. Continue when you are ready to begin."}
        onContinue={() => {
          setStep({ phase: "scenario", scenario_order: scenario.scenario_order })
          setPhase("scenario")
        }}
      />
    )
  }

  if (phase === "scenario" && data) {
    const scenario = data.scenarios[scenarioIdx]
    return Frame(
      <ScenarioCall code={code} scenario={scenario} onDone={(sessionId) => {
        setPostSessionId(sessionId)
        setStep({ phase: "post", scenario_order: scenario.scenario_order, session_id: sessionId })
        setPhase("post")
      }} />
    )
  }

  if (phase === "post" && data) {
    const scenario = data.scenarios[scenarioIdx]
    const sid = postSessionId
    const isPractice = scenario.study_role === "practice"
    const postItems = isPractice
      ? ((scenario.post_items as QItem[]) || [])
      : mergePostItems(q("post"), (scenario.post_items as QItem[]) || [])
    return Frame(
      <QuestionnaireForm title={isPractice ? "Practice check" : `After scenario ${scenario.scenario_order - 1}`} items={postItems}
        submitLabel={scenarioIdx < data.scenarios.length - 1 ? "Next scenario" : "Final questions"} busy={busy}
        onSubmit={async (ans) => {
          setBusy(true)
          try {
            if (!sid) throw new Error("Could not identify the completed scenario session")
            await api.questionnaire(sid, code, isPractice ? "practice_post" : "post", ans)
            if (scenarioIdx < data.scenarios.length - 1) {
              const next = scenarioIdx + 1
              setScenarioIdx(next)
              setPostSessionId(null)
              const nextPhase = isPractice ? "main_intro" : "scenario"
              setStep({ phase: nextPhase, scenario_order: data.scenarios[next].scenario_order })
              setPhase(nextPhase)
            } else {
              const nextPhase = q("pre_playback").length ? "pre_playback" : "final"
              setStep({ phase: nextPhase })
              setPhase(nextPhase)
            }
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  const analyticalScenarioOptions = data?.scenarios
    .filter(s => s.study_role !== "practice")
    .map(s => s.title)
    .filter(Boolean) || []

  if (phase === "pre_playback" && data) {
    return Frame(
      <QuestionnaireForm title="Post-session questionnaire" items={q("pre_playback")}
        submitLabel="Continue to recording" busy={busy} scenarioOptions={analyticalScenarioOptions}
        onSubmit={async (ans) => {
          setBusy(true)
          try {
            await api.questionnaire(null, code, "pre_playback", ans)
            setStep({ phase: "playback" }); setPhase("playback")
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  if (phase === "playback" && data) {
    return Frame(
      <QuestionnaireForm title="Recording and voice ratings" items={q("playback")}
        submitLabel="Continue to debriefing" busy={busy}
        playbackUrl={(item) => api.playbackUrl(
          code, item.scenario_order, item.track, item.condition, item.max_duration_s)}
        onSubmit={async (ans) => {
          setBusy(true)
          try {
            await api.questionnaire(null, code, "playback", ans)
            setStep({ phase: "debrief" }); setPhase("debrief")
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  if (phase === "debrief" && data) {
    return Frame(
      <QuestionnaireForm title="Final comment and debriefing" items={q("debrief")}
        submitLabel="Submit study" busy={busy}
        onSubmit={async (ans) => {
          setBusy(true)
          try {
            await api.questionnaire(null, code, "debrief", ans)
            await api.submit(code)
            setDeadline(null); setPhase("completion")
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  if (phase === "final" && data) {
    return Frame(
      <QuestionnaireForm title="Final questionnaire" items={q("final")} submitLabel="Submit study" busy={busy}
        scenarioOptions={analyticalScenarioOptions} playbackUrl={(item) => api.playbackUrl(
          code, item.scenario_order, item.track, item.condition, item.max_duration_s)}
        onSubmit={async (ans) => {
          setBusy(true)
          try {
            await api.questionnaire(null, code, "final", ans)
            await api.submit(code)
            setDeadline(null)
            setPhase("completion")
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  if (phase === "completion") {
    return (
      <Centered>
        <CheckCircle2 className="size-12 text-primary" />
        <h1 className="text-2xl font-bold">Thank you!</h1>
        <p className="text-base text-muted-foreground">Your responses have been saved. You may close this window.</p>
      </Centered>
    )
  }

  return <Centered><Spinner /></Centered>
}

function Header({ remaining }: { remaining: number | null }) {
  return (
    <header className="mb-6 flex items-center justify-between">
      <img src="/KTH_Logo.jpg" alt="KTH" className="h-12" />
      {remaining !== null && (
        <div className="text-sm text-muted-foreground">
          Time left: <span className="font-mono tabular-nums">{Math.floor(remaining / 60)}:{(remaining % 60).toString().padStart(2, "0")}</span>
        </div>
      )}
    </header>
  )
}

function TransitionScreen({ title, text, onContinue }: {
  title: string
  text: string
  onContinue: () => void
}) {
  return (
    <main className="mx-auto max-w-xl py-10 sm:py-16">
      <h1 className="text-2xl font-semibold">{title}</h1>
      <div className="mt-4 whitespace-pre-wrap text-base leading-7 text-muted-foreground">
        {text}
      </div>
      <Button className="mt-7 text-base" onClick={onContinue}>Continue</Button>
    </main>
  )
}

function Centered({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 px-4 text-center">
      {children}
    </div>
  )
}
