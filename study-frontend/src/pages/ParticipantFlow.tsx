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
  | "consent" | "background" | "scenario" | "post" | "final" | "completion"

function sessionIdFor(pid: string, order: number) {
  return `${pid}_S${String(order).padStart(2, "0")}`
}

export function ParticipantFlow() {
  const [code, setCode] = useState("")
  const [data, setData] = useState<EnterResult | null>(null)
  const [phase, setPhase] = useState<Phase>("code")
  const [scenarioIdx, setScenarioIdx] = useState(0)
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
    if (p === "consent") { setPhase("consent"); return }
    if (p === "scenario" || p === "post") {
      setScenarioIdx(Math.max(0, (step.scenario_order ?? 1) - 1))
      setPhase(p)
      return
    }
    if (p === "final") { setPhase("final"); return }
    setScenarioIdx(0)
    setPhase("background")
  }, [])

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
        setScenarioIdx(0); setStep({ phase: "background" }); setPhase("background")
      } else {
        goToRunStep(res.run)
      }
    } catch (e: any) {
      setError(e?.message || "Could not start")
    } finally { setBusy(false) }
  }, [code, goToRunStep, setStep])

  // ---------- render ----------
  if (phase === "code") {
    return (
      <Centered>
        <h1 className="text-2xl font-bold">Welcome</h1>
        <p className="text-sm text-muted-foreground">Enter your participant code to begin.</p>
        <div className="flex w-full max-w-xs gap-2">
          <Input value={code} onChange={e => setCode(e.target.value.toUpperCase())}
                 placeholder="e.g. A7X9K2" onKeyDown={e => e.key === "Enter" && doEnter(code)} />
          <Button onClick={() => doEnter(code)} disabled={busy || !code}>{busy ? "…" : "Start"}</Button>
        </div>
        {error && <p className="text-sm text-destructive">{error}</p>}
      </Centered>
    )
  }

  if (phase === "welcome" && data) {
    const st = data.run.status
    return (
      <Centered>
        <h1 className="text-2xl font-bold">{data.study_name}</h1>
        <p className="text-sm text-muted-foreground">Participant {data.participant_id}</p>
        {st === "submitted" ? (
          <>
            <p className="text-sm">You have already completed this study. Thank you!</p>
          </>
        ) : st === "in_progress" || st === "expired" ? (
          <>
            <p className="max-w-md text-center text-sm text-muted-foreground">
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
              ? <div className="max-h-[60vh] max-w-2xl overflow-auto whitespace-pre-wrap text-left text-sm">{data.welcome_text}</div>
              : <p className="max-w-md text-center text-sm text-muted-foreground">
                  You will complete {data.scenarios.length} conversation scenarios, each followed by a brief
                  questionnaire. You have one hour to finish.
                </p>}
            {data.estimated_duration && <p className="text-xs text-muted-foreground">Estimated duration: {data.estimated_duration}</p>}
            <Button onClick={() => startRun("restart")} disabled={busy}>Begin</Button>
          </>
        )}
        {error && <p className="text-sm text-destructive">{error}</p>}
      </Centered>
    )
  }

  const Frame = (children: ReactNode) => (
    <div className="mx-auto flex min-h-screen max-w-4xl flex-col px-4 py-6">
      <Header remaining={remaining} />
      <div className="flex-1">{children}</div>
    </div>
  )

  if (phase === "background" && data) {
    return Frame(
      <QuestionnaireForm title="Background questionnaire" items={q("background")} submitLabel="Continue" busy={busy}
        onSubmit={async (ans) => {
          setBusy(true)
          try {
            await api.questionnaire(null, code, "background", ans)
            setStep({ phase: "consent" })
            setPhase("consent")
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  if (phase === "consent" && data) {
    return Frame(
      <AudioCheck code={code} items={q("consent")}
        onDone={async (ans) => {
          setBusy(true)
          try {
            await api.questionnaire(null, code, "consent", ans)
            setScenarioIdx(0)
            setStep({ phase: "scenario", scenario_order: 1 })
            setPhase("scenario")
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  if (phase === "scenario" && data) {
    const scenario = data.scenarios[scenarioIdx]
    return Frame(
      <ScenarioCall code={code} scenario={scenario} onDone={() => {
        setStep({ phase: "post", scenario_order: scenario.scenario_order })
        setPhase("post")
      }} />
    )
  }

  if (phase === "post" && data) {
    const scenario = data.scenarios[scenarioIdx]
    const sid = sessionIdFor(data.participant_id, scenario.scenario_order)
    const postItems = [...q("post"), ...((scenario.post_items as QItem[]) || [])]
    return Frame(
      <QuestionnaireForm title={`After scenario ${scenario.scenario_order}`} items={postItems}
        submitLabel={scenarioIdx < data.scenarios.length - 1 ? "Next scenario" : "Final questions"} busy={busy}
        onSubmit={async (ans) => {
          setBusy(true)
          try {
            await api.questionnaire(sid, code, "post", ans)
            if (scenarioIdx < data.scenarios.length - 1) {
              const next = scenarioIdx + 1
              setScenarioIdx(next)
              setStep({ phase: "scenario", scenario_order: data.scenarios[next].scenario_order })
              setPhase("scenario")
            } else {
              setStep({ phase: "final" })
              setPhase("final")
            }
          } catch (e) { handleErr(e) } finally { setBusy(false) }
        }} />
    )
  }

  if (phase === "final" && data) {
    const scenarioOptions = data.scenarios.map(s => s.title).filter(Boolean)
    return Frame(
      <QuestionnaireForm title="Final questionnaire" items={q("final")} submitLabel="Submit study" busy={busy}
        scenarioOptions={scenarioOptions} playbackUrl={(item) => api.playbackUrl(code, item.scenario_order, item.track)}
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
        <p className="text-sm text-muted-foreground">Your responses have been saved. You may close this window.</p>
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

function Centered({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 px-4 text-center">
      {children}
    </div>
  )
}
