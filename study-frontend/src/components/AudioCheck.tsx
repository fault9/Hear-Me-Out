import { useCallback, useEffect, useRef, useState } from "react"
import { Button } from "@shared/ui/button"
import { Spinner } from "@shared/ui/spinner"
import { CheckCircle2, Mic } from "lucide-react"
import { useStudyConversation } from "@/hooks/useStudyConversation"
import { api, streamPrepare } from "@/api"
import { QuestionnaireForm, type QItem } from "@/components/QuestionnaireForm"

type Phase = "idle" | "preparing" | "testing" | "passed" | "error"
const MIC_THRESHOLD = 0.03

export function AudioCheck({ code, items, onDone }: {
  code: string; items: QItem[]; onDone: (answers: Record<string, any>) => void
}) {
  const conv = useStudyConversation()
  const [phase, setPhase] = useState<Phase>("idle")
  const [micOk, setMicOk] = useState(false)
  const [heardOk, setHeardOk] = useState(false)
  const [overridden, setOverridden] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const stopPrepare = useRef<() => void>(() => {})

  // Mic detected once the level crosses the threshold.
  useEffect(() => { if (conv.amplitude > MIC_THRESHOLD) setMicOk(true) }, [conv.amplitude])

  // Reflect connection errors.
  useEffect(() => {
    if (conv.error && (phase === "testing" || phase === "preparing")) { setErr(conv.error); setPhase("error") }
  }, [conv.error, phase])

  // Timeout if the assistant never handshakes.
  useEffect(() => {
    if (phase !== "testing" || conv.handshakeReceived) return
    const t = setTimeout(() => { setErr("The assistant did not respond in time."); setPhase("error") }, 20000)
    return () => clearTimeout(t)
  }, [phase, conv.handshakeReceived])

  // Both checks pass → stop the call, reveal the confirmations.
  useEffect(() => {
    if (phase === "testing" && micOk && heardOk) { setPhase("passed"); conv.teardown() }
  }, [phase, micOk, heardOk, conv])

  // Always tear down on unmount.
  useEffect(() => () => conv.teardown(), []) // eslint-disable-line

  const start = useCallback(async () => {
    setErr(null); setMicOk(false); setHeardOk(false); setPhase("preparing")
    try {
      const res = await api.audioCheckStart(code)
      const begin = () => { setPhase("testing"); conv.start(res.session_id) }
      if (res.prepare?.status === "ready") begin()
      else stopPrepare.current = streamPrepare(s => {
        if (s.status === "ready") { stopPrepare.current(); begin() }
        if (s.status === "error") { setErr(s.error || "Preparation failed"); setPhase("error") }
      })
    } catch (e: any) { setErr(e?.message || "Could not start the audio check."); setPhase("error") }
  }, [code, conv])

  const canContinue = phase === "passed" || overridden
  const level = Math.min(100, Math.round(conv.amplitude * 400))

  return (
    <div className="mx-auto w-full max-w-2xl">
      <h2 className="mb-4 text-xl font-semibold tracking-tight">Audio check</h2>

      <div className="rounded-xl border bg-card p-5">
        {phase === "idle" && (
          <div className="flex flex-col items-start gap-3">
            <p className="text-sm text-muted-foreground">
              We'll quickly check your microphone and that you can hear the assistant. When you start,
              say <b>"Hello, can you hear me?"</b> — the level meter should move and you should hear a short reply.
            </p>
            <Button className="gap-2" onClick={start}><Mic className="size-4" /> Start audio check</Button>
          </div>
        )}

        {phase === "preparing" && (
          <div className="flex items-center gap-2 py-4 text-sm text-muted-foreground"><Spinner /> Preparing…</div>
        )}

        {(phase === "testing" || phase === "passed") && (
          <div className="flex flex-col gap-4">
            <p className="text-sm text-muted-foreground">Say "Hello, can you hear me?" and wait for a reply.</p>
            <div>
              <div className="mb-1 flex items-center gap-2 text-sm">
                <Mic className="size-4" /> Microphone {micOk && <CheckCircle2 className="size-4 text-primary" />}
              </div>
              <div className="h-3 w-full overflow-hidden rounded-full bg-muted">
                <div className="h-full rounded-full bg-primary transition-[width] duration-100" style={{ width: `${level}%` }} />
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <div className="text-sm">
                {conv.modelAudioReceived ? "The assistant is responding — did you hear it?" : "Waiting for the assistant to reply…"}
              </div>
              <Button variant={heardOk ? "default" : "secondary"} className="w-fit gap-2" disabled={!conv.modelAudioReceived}
                onClick={() => setHeardOk(true)}>
                {heardOk && <CheckCircle2 className="size-4" />} Yes, I heard the assistant clearly
              </Button>
            </div>
          </div>
        )}

        {phase === "error" && (
          <div className="flex flex-col items-start gap-3">
            <p className="text-sm text-destructive">{err || "The audio check could not complete."}</p>
            <div className="flex items-center gap-2">
              <Button variant="secondary" onClick={start}>Retry</Button>
            </div>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={overridden} onChange={e => { setOverridden(e.target.checked); if (e.target.checked) conv.teardown() }} />
              I confirm my microphone and audio are working (skip the automated check).
            </label>
          </div>
        )}

        {phase === "passed" && (
          <p className="mt-3 flex items-center gap-2 text-sm text-primary"><CheckCircle2 className="size-4" /> Audio check passed.</p>
        )}
      </div>

      {canContinue && (
        <div className="mt-6">
          <QuestionnaireForm title="Please confirm" items={items} submitLabel="Continue"
            onSubmit={(ans) => onDone({ ...ans, _audio_check: { mic_ok: micOk, heard_ok: heardOk, overridden } })} />
        </div>
      )}
    </div>
  )
}
