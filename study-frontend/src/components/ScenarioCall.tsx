import { useCallback, useEffect, useRef, useState } from "react"
import { AlertDialog } from "@base-ui/react/alert-dialog"
import { Button, buttonVariants } from "@shared/ui/button"
import { Badge } from "@shared/ui/badge"
import { Spinner } from "@shared/ui/spinner"
import { Phone, AlertTriangle, CheckCircle2, XCircle, Target, MessageCircle, Info } from "lucide-react"
import ReactMarkdown, { type Components } from "react-markdown"
import { useStudyConversation } from "@/hooks/useStudyConversation"
import { api, streamPrepare, type ScenarioInfo, type PrepareState } from "@/api"

function fmt(s: number) {
  const m = Math.floor(s / 60), r = s % 60
  return `${m}:${r.toString().padStart(2, "0")}`
}

type Phase = "preparing" | "ready" | "connecting" | "active" | "processing" | "error"

export function ScenarioCall({ code, scenario, onDone }: {
  code: string; scenario: ScenarioInfo; onDone: (sessionId: string) => void
}) {
  const conv = useStudyConversation()
  const [phase, setPhase] = useState<Phase>("preparing")
  const [prepare, setPrepare] = useState<PrepareState | null>(null)
  const [remaining, setRemaining] = useState(scenario.time_limit_s)
  const [errMsg, setErrMsg] = useState<string | null>(null)
  const [readyConfirmed, setReadyConfirmed] = useState(false)
  const [finalCheckOpen, setFinalCheckOpen] = useState(false)
  const sessionIdRef = useRef<string | null>(null)
  const started = useRef(false)
  const finalAccountPrompt = scenario.final_account_prompt?.trim() || ""
  const scenarioLabel = scenario.study_role === "practice"
    ? "Practice interaction"
    : `Scenario ${Math.max(1, scenario.scenario_order - 1)}`

  // On mount: create the session + prepare the engine this scenario needs.
  useEffect(() => {
    if (started.current) return
    started.current = true
    let stop = () => {}
    ;(async () => {
      try {
        const res = await api.sessionStart(code, scenario.scenario_order)
        sessionIdRef.current = res.session_id
        if (res.prepare?.status === "ready") { setPhase("ready"); return }
        setPrepare(res.prepare)
        stop = streamPrepare((s) => {
          setPrepare(s)
          if (s.status === "ready") setPhase("ready")
          if (s.status === "error") { setErrMsg(s.error || "Preparation failed"); setPhase("error") }
        })
      } catch (e: any) {
        setErrMsg(e?.message || "Could not start the session."); setPhase("error")
      }
    })()
    return () => stop()
  }, [code, scenario.scenario_order])

  useEffect(() => {
    if (conv.status === "active" && phase === "connecting") setPhase("active")
    if (conv.status === "error" && (phase === "connecting" || phase === "active")) {
      setErrMsg(conv.error || "Could not connect to the assistant."); setPhase("error")
    }
  }, [conv.status, conv.error, phase])

  useEffect(() => {
    if (phase !== "active") return
    const t = setInterval(() => setRemaining(r => Math.max(0, r - 1)), 1000)
    return () => clearInterval(t)
  }, [phase])

  const startCall = useCallback(async () => {
    const sid = sessionIdRef.current
    if (!sid) return
    setErrMsg(null); setPhase("connecting")
    try { await conv.start(sid) } catch (e: any) { setErrMsg(e?.message || "Could not start the call."); setPhase("error") }
  }, [conv])

  const endCall = useCallback(async (reason: string) => {
    setFinalCheckOpen(false)
    setPhase("processing")
    const sid = sessionIdRef.current
    try {
      const arts = await conv.stopAndAssemble()
      if (sid) { await api.saveSession(sid, arts); await api.sessionEnd(sid, reason) }
      if (!sid) throw new Error("The session identifier was lost before saving")
      onDone(sid)
    } catch (e: any) {
      setErrMsg("Saving failed: " + (e?.message || e) + ". You can continue."); setPhase("error")
    }
  }, [conv, onDone])

  const skipAfterError = useCallback(async () => {
    const sid = sessionIdRef.current
    if (!sid) return
    try { await api.sessionEnd(sid, "technical_problem") } catch { /* preserve forward progress */ }
    onDone(sid)
  }, [onDone])

  // Auto-end when the scenario time limit is reached.
  useEffect(() => {
    if (phase === "active" && remaining === 0) endCall("time_up")
  }, [phase, remaining, endCall])

  const statusBadge = () => {
    if (conv.error) return <Badge variant="destructive">Connection error</Badge>
    if (phase === "active") return <Badge>Connected</Badge>
    if (phase === "connecting") return <Badge variant="secondary">Connecting…</Badge>
    return <Badge variant="secondary">Not connected</Badge>
  }

  const requestGoalEnd = () => {
    if (finalAccountPrompt) {
      setFinalCheckOpen(true)
      return
    }
    void endCall("goal_reached")
  }

  return (
    <>
      <div className="grid items-start gap-5 md:grid-cols-[minmax(0,1fr)_320px]">
      <section className="overflow-hidden rounded-lg border bg-card">
        <header className="border-b px-5 py-4 sm:px-6">
          <p className="text-sm font-semibold uppercase text-muted-foreground">{scenarioLabel}</p>
          <h1 className="mt-1 text-xl font-semibold leading-tight sm:text-2xl">
            {scenario.title || "Your conversation"}
          </h1>
        </header>

        <div className="px-5 py-5 sm:px-6">
          <section aria-labelledby="scenario-situation">
            <SectionLabel id="scenario-situation">Your situation</SectionLabel>
            <ScenarioText value={scenario.role} className="text-lg font-medium leading-7" />
            <ScenarioText
              value={scenario.current_situation}
              className="mt-2 text-base leading-7 text-muted-foreground"
            />
          </section>

          <section className="-mx-5 mt-5 border-y bg-primary/5 px-5 py-4 sm:-mx-6 sm:px-6" aria-labelledby="scenario-goal">
            <div className="flex gap-3">
              <Target className="mt-0.5 size-5 shrink-0 text-primary" aria-hidden="true" />
              <div>
                <SectionLabel id="scenario-goal" className="text-primary">Your aim</SectionLabel>
                <ScenarioText value={scenario.goal} className="text-lg font-medium leading-7" />
              </div>
            </div>
          </section>

          <div className="divide-y">
            <section className="py-5" aria-labelledby="scenario-opening">
              <div className="flex gap-3">
                <MessageCircle className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
                <div className="min-w-0">
                  <SectionLabel id="scenario-opening">Start here</SectionLabel>
                  <p className="whitespace-pre-wrap text-base italic leading-7">“{scenario.suggested_first_line}”</p>
                  {scenario.opening_details && (
                    <ScenarioText
                      value={scenario.opening_details}
                      className="mt-2 text-sm leading-6 text-muted-foreground"
                    />
                  )}
                </div>
              </div>
            </section>

            <section className="py-5" aria-labelledby="scenario-details">
              <div className="flex gap-3">
                <Info className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
                <div className="min-w-0 flex-1">
                  <SectionLabel id="scenario-details">Before ending</SectionLabel>
                  {finalAccountPrompt && (
                    <p className="text-base font-medium leading-7">Ask the assistant for a final summary.</p>
                  )}
                  <ScenarioText
                    value={scenario.additional_details}
                    components={checklistMarkdownComponents}
                    className={`${finalAccountPrompt ? "mt-2 " : ""}text-base leading-7 text-muted-foreground`}
                  />
                </div>
              </div>
            </section>
          </div>

          {(scenario.extra_fields || []).map((f, i) => <Field key={i} label={f.label} value={f.value} />)}

          {scenario.how_to_interact && (
            <ScenarioText
              value={scenario.how_to_interact}
              className="border-t pt-4 text-sm leading-6 text-muted-foreground"
            />
          )}
        </div>
      </section>

      <div className="flex flex-col gap-4 rounded-lg border bg-card p-5 md:sticky md:top-5">
        <div className="flex items-center justify-between">
          {statusBadge()}
          <span className="font-mono text-lg tabular-nums">{fmt(remaining)}</span>
        </div>

        {phase === "preparing" && (
          <div className="flex flex-col gap-2 py-2">
            <div className="flex items-center gap-2 text-base text-muted-foreground"><Spinner /> Preparing your session…</div>
            {(prepare?.steps || []).map((s, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                {s.state === "done" ? <CheckCircle2 className="size-3.5 text-primary" />
                  : s.state === "error" ? <XCircle className="size-3.5 text-destructive" />
                  : <Spinner className="size-3.5" />}
                <span className={s.state === "error" ? "text-destructive" : ""}>{s.label}</span>
              </div>
            ))}
          </div>
        )}

        {phase === "ready" && (
          <div className="flex flex-col gap-3">
            <label className="flex cursor-pointer items-start gap-2 text-base leading-6">
              <input
                type="checkbox"
                checked={readyConfirmed}
                onChange={event => setReadyConfirmed(event.target.checked)}
                className="mt-0.5 size-4 shrink-0 accent-primary"
              />
              <span>I have read the scenario and I am ready to begin.</span>
            </label>
            <Button className="w-full gap-2 text-base" onClick={startCall} disabled={!readyConfirmed}>
              <Phone className="size-4" /> Start Call
            </Button>
          </div>
        )}

        {phase === "connecting" && (
          <div className="flex items-center justify-center gap-2 py-6 text-base text-muted-foreground"><Spinner /> Connecting to the assistant…</div>
        )}

        {phase === "active" && (
          <div className="flex flex-col gap-2">
            {finalAccountPrompt && (
              <div className="mb-1 flex items-start gap-2 border-y border-primary/40 py-2.5 text-sm leading-5">
                <Info className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden="true" />
                <span>Final summary required before ending.</span>
              </div>
            )}
            <p className="text-center text-base text-muted-foreground">Speak with the assistant. End the call when you are done.</p>
            <Button
              className="w-full gap-2 text-base"
              onClick={requestGoalEnd}
            >
              <CheckCircle2 className="size-4" /> End Call — goal reached
            </Button>
            <Button variant="secondary" className="w-full gap-2 text-base" onClick={() => endCall("give_up")}><XCircle className="size-4" /> End Call — give up</Button>
          </div>
        )}

        {phase === "processing" && (
          <div className="flex items-center justify-center gap-2 py-6 text-base text-muted-foreground"><Spinner /> Saving your session…</div>
        )}

        {errMsg && <p className="text-base text-destructive">{errMsg}</p>}
        {phase === "error" && (
          <div className="flex flex-col gap-2">
            <Button variant="secondary" className="w-full" onClick={() => window.location.reload()}>Try again</Button>
            <Button variant="ghost" className="w-full" onClick={skipAfterError}>Skip / continue</Button>
          </div>
        )}

        {(phase === "active" || phase === "connecting") && (
          <Button variant="ghost" className="mt-2 w-full gap-2 text-muted-foreground" onClick={() => endCall("technical_problem")}>
            <AlertTriangle className="size-4" /> Technical problem
          </Button>
        )}
        </div>
      </div>

      {finalAccountPrompt && (
        <FinalAccountDialog
          open={finalCheckOpen}
          prompt={finalAccountPrompt}
          onOpenChange={setFinalCheckOpen}
          onConfirm={() => {
            setFinalCheckOpen(false)
            void endCall("goal_reached")
          }}
        />
      )}
    </>
  )
}

function FinalAccountDialog({ open, prompt, onOpenChange, onConfirm }: {
  open: boolean
  prompt: string
  onOpenChange: (open: boolean) => void
  onConfirm: () => void
}) {
  return (
    <AlertDialog.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialog.Portal>
        <AlertDialog.Backdrop className="fixed inset-0 z-50 bg-black/65" />
        <AlertDialog.Viewport className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <AlertDialog.Popup className="w-full max-w-lg rounded-lg border bg-background p-5 shadow-2xl outline-none sm:p-6">
            <AlertDialog.Title className="text-xl font-semibold">Before you end</AlertDialog.Title>
            <AlertDialog.Description className="mt-2 text-base leading-6 text-muted-foreground">
              Have you asked this question and heard the assistant’s answer?
            </AlertDialog.Description>
            <blockquote className="mt-4 border-l-2 border-primary pl-4 text-lg font-medium leading-7">
              “{prompt}”
            </blockquote>
            <div className="mt-6 flex flex-col gap-2 sm:flex-row">
              <AlertDialog.Close className={buttonVariants({ className: "h-10 flex-1 text-base" })}>
                Return to conversation
              </AlertDialog.Close>
              <Button variant="secondary" className="h-10 flex-1 whitespace-normal text-base" onClick={onConfirm}>
                Yes — End call
              </Button>
            </div>
          </AlertDialog.Popup>
        </AlertDialog.Viewport>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  )
}

function Field({ label, value }: { label: string; value: string }) {
  if (!value) return null
  return (
    <div className="border-t py-4">
      <SectionLabel>{label}</SectionLabel>
      <ScenarioText value={value} className="text-base leading-7" />
    </div>
  )
}

function SectionLabel({ id, className = "", children }: {
  id?: string; className?: string; children: React.ReactNode
}) {
  return (
    <h2 id={id} className={`mb-1 text-sm font-semibold uppercase text-muted-foreground ${className}`}>
      {children}
    </h2>
  )
}

const markdownComponents: Components = {
  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
  ul: ({ children }) => (
    <ul className="my-2 list-disc space-y-1 pl-5 first:mt-0 last:mb-0">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="my-2 list-decimal space-y-1 pl-5 first:mt-0 last:mb-0">{children}</ol>
  ),
  li: ({ children }) => <li className="pl-0.5">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
}

const checklistMarkdownComponents: Components = {
  ...markdownComponents,
  ul: ({ children }) => (
    <ul className="my-2 grid list-disc gap-x-8 gap-y-1 pl-5 first:mt-0 last:mb-0 sm:grid-cols-2">
      {children}
    </ul>
  ),
}

function ScenarioText({ value, className = "", components = markdownComponents }: {
  value: string; className?: string; components?: Components
}) {
  if (!value) return null
  return (
    <div className={className}>
      <ReactMarkdown components={components}>{value}</ReactMarkdown>
    </div>
  )
}
