import { useCallback, useEffect, useRef, useState } from "react"
import { Button } from "@shared/ui/button"
import { Badge } from "@shared/ui/badge"
import { Spinner } from "@shared/ui/spinner"
import { Phone, AlertTriangle, CheckCircle2, XCircle } from "lucide-react"
import { useStudyConversation } from "@/hooks/useStudyConversation"
import { api, streamPrepare, type ScenarioInfo, type PrepareState } from "@/api"

function fmt(s: number) {
  const m = Math.floor(s / 60), r = s % 60
  return `${m}:${r.toString().padStart(2, "0")}`
}

type Phase = "preparing" | "ready" | "connecting" | "active" | "processing" | "error"

export function ScenarioCall({ code, scenario, onDone }: {
  code: string; scenario: ScenarioInfo; onDone: () => void
}) {
  const conv = useStudyConversation()
  const [phase, setPhase] = useState<Phase>("preparing")
  const [prepare, setPrepare] = useState<PrepareState | null>(null)
  const [remaining, setRemaining] = useState(scenario.time_limit_s)
  const [errMsg, setErrMsg] = useState<string | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const started = useRef(false)

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
    setPhase("processing")
    const sid = sessionIdRef.current
    try {
      const arts = await conv.stopAndAssemble()
      if (sid) { await api.saveSession(sid, arts); await api.sessionEnd(sid, reason) }
      onDone()
    } catch (e: any) {
      setErrMsg("Saving failed: " + (e?.message || e) + ". You can continue."); setPhase("error")
    }
  }, [conv, onDone])

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

  return (
    <div className="grid gap-5 md:grid-cols-[1fr_320px]">
      <div className="rounded-xl border bg-card p-5">
        <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Scenario {scenario.scenario_order}{scenario.title ? ` — ${scenario.title}` : ""}
        </div>
        <Field label="Your role" value={scenario.role} />
        <Field label="Current situation" value={scenario.current_situation} />
        <Field label="Your goal" value={scenario.goal} />
        <Field label="How to interact" value={scenario.how_to_interact} />
        <Field label="Suggested first line" value={scenario.suggested_first_line} />
        <Field label="Additional details" value={scenario.additional_details} />
        {(scenario.extra_fields || []).map((f, i) => <Field key={i} label={f.label} value={f.value} />)}
      </div>

      <div className="flex flex-col gap-4 rounded-xl border bg-card p-5">
        <div className="flex items-center justify-between">
          {statusBadge()}
          <span className="font-mono text-lg tabular-nums">{fmt(remaining)}</span>
        </div>

        {phase === "preparing" && (
          <div className="flex flex-col gap-2 py-2">
            <div className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner /> Preparing your session…</div>
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
          <Button className="w-full gap-2" onClick={startCall}><Phone className="size-4" /> Start Call</Button>
        )}

        {phase === "connecting" && (
          <div className="flex items-center justify-center gap-2 py-6 text-sm text-muted-foreground"><Spinner /> Connecting to the assistant…</div>
        )}

        {phase === "active" && (
          <div className="flex flex-col gap-2">
            <p className="text-center text-sm text-muted-foreground">Speak with the assistant. End the call when you are done.</p>
            <Button className="w-full gap-2" onClick={() => endCall("goal_reached")}><CheckCircle2 className="size-4" /> End Call — goal reached</Button>
            <Button variant="secondary" className="w-full gap-2" onClick={() => endCall("give_up")}><XCircle className="size-4" /> End Call — give up</Button>
          </div>
        )}

        {phase === "processing" && (
          <div className="flex items-center justify-center gap-2 py-6 text-sm text-muted-foreground"><Spinner /> Saving your session…</div>
        )}

        {errMsg && <p className="text-sm text-destructive">{errMsg}</p>}
        {phase === "error" && (
          <div className="flex flex-col gap-2">
            <Button variant="secondary" className="w-full" onClick={() => window.location.reload()}>Try again</Button>
            <Button variant="ghost" className="w-full" onClick={onDone}>Skip / continue</Button>
          </div>
        )}

        {(phase === "active" || phase === "connecting") && (
          <Button variant="ghost" className="mt-2 w-full gap-2 text-muted-foreground" onClick={() => endCall("technical_problem")}>
            <AlertTriangle className="size-4" /> Technical problem
          </Button>
        )}
      </div>
    </div>
  )
}

function Field({ label, value }: { label: string; value: string }) {
  if (!value) return null
  return (
    <div className="mb-3">
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <div className="whitespace-pre-wrap text-sm">{value}</div>
    </div>
  )
}
