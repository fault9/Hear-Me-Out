import { useCallback, useEffect, useState, type ReactNode } from "react"
import { Button } from "@shared/ui/button"
import { Input } from "@shared/ui/input"
import { Badge } from "@shared/ui/badge"
import { adminApi } from "@/api"
import { ScenarioEditor } from "@/components/admin/ScenarioEditor"
import { QuestionnaireBuilder } from "@/components/admin/QuestionnaireBuilder"

type Tab = "scenarios" | "targets" | "questionnaires" | "participants" | "data"

export function StudyEditor({ token, studyId, onBack }: {
  token: string; studyId: number; onBack: () => void
}) {
  const [study, setStudy] = useState<any>(null)
  const [voices, setVoices] = useState<string[]>([])
  const [engines, setEngines] = useState<string[]>([])
  const [tab, setTab] = useState<Tab>("scenarios")
  const [err, setErr] = useState<string | null>(null)

  const reload = useCallback(async () => {
    try {
      const r = await adminApi.getStudy(token, studyId)
      setStudy(r.study)
    } catch (e: any) { setErr(e?.message || String(e)) }
  }, [token, studyId])

  useEffect(() => {
    reload()
    adminApi.voices(token).then(r => setVoices(r.voices)).catch(() => {})
    adminApi.engines(token).then(r => setEngines(r.engines)).catch(() => {})
  }, [reload, token])

  if (!study) return <div className="p-6 text-sm text-muted-foreground">Loading…</div>

  const targets: any[] = study.targets || []
  const scenarios: any[] = study.scenarios || []

  return (
    <div className="mx-auto max-w-4xl px-4 py-6">
      <div className="mb-4 flex items-center gap-3">
        <Button size="sm" variant="ghost" onClick={onBack}>← Studies</Button>
        <h1 className="text-xl font-bold">{study.name}</h1>
        <span className="text-xs text-muted-foreground">#{study.id}</span>
      </div>
      {err && <p className="mb-3 text-sm text-destructive">{err}</p>}

      <div className="mb-5 flex flex-wrap gap-1 border-b">
        {(["scenarios", "targets", "questionnaires", "participants", "data"] as Tab[]).map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-3 py-2 text-sm capitalize ${tab === t ? "border-b-2 border-primary font-medium" : "text-muted-foreground"}`}>
            {t}
          </button>
        ))}
      </div>

      {tab === "scenarios" && (
        <div className="flex flex-col gap-4">
          {scenarios.map((sc, i) => (
            <ScenarioEditor key={sc.id} token={token} studyId={studyId} scenario={sc} index={i}
              voices={voices} engines={engines} targets={targets} onChange={reload} />
          ))}
          <Button variant="secondary" onClick={async () => {
            await adminApi.addScenario(token, studyId, {
              order_idx: scenarios.length, title: `Scenario ${scenarios.length + 1}`,
              system_prompt: "You are a helpful conversational partner. Keep your replies concise.",
              voice_prompt: voices[0] || "NATF2.pt", time_limit_s: 300,
              scenario_card: {}, voice_schedule: [{ mode: "natural", start_s: 0, end_s: null }],
            })
            reload()
          }}>+ Add scenario</Button>
        </div>
      )}

      {tab === "targets" && (
        <TargetsPanel token={token} studyId={studyId} targets={targets} engines={engines} onChange={reload} />
      )}

      {tab === "questionnaires" && (
        <QuestionnaireBuilder token={token} studyId={studyId}
          questionnaires={study.questionnaires || {}} onChange={reload} />
      )}

      {tab === "participants" && (
        <ParticipantsPanel token={token} studyId={studyId}
          participants={study.participants || []} hasScenarios={scenarios.length > 0} onChange={reload} />
      )}

      {tab === "data" && <DataPanel token={token} studyId={studyId} />}
    </div>
  )
}

function TargetsPanel({ token, studyId, targets, engines, onChange }: any) {
  const [speakerId, setSpeakerId] = useState("")
  const [engine, setEngine] = useState(engines[0] || "meanvc")
  const [file, setFile] = useState<File | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-end gap-2 rounded-lg border p-3">
        <Labeled label="Speaker ID"><Input value={speakerId} onChange={e => setSpeakerId(e.target.value)} placeholder="e.g. p225" className="w-36" /></Labeled>
        <Labeled label="Engine">
          <select className="rounded-md border bg-background px-2 py-2 text-sm" value={engine} onChange={e => setEngine(e.target.value)}>
            {engines.map((en: string) => <option key={en} value={en}>{en}</option>)}
          </select>
        </Labeled>
        <input type="file" accept="audio/*" onChange={e => setFile(e.target.files?.[0] || null)} className="text-sm" />
        <Button size="sm" disabled={busy || !speakerId.trim() || !file} onClick={async () => {
          if (!file) return
          const id = speakerId.trim()
          setBusy(true); setErr(null)
          // One identifier names the voice: used as the link key, the display name,
          // and the target_speaker_id saved with each session.
          try { await adminApi.uploadTarget(token, studyId, id, id, id, engine, file); setSpeakerId(""); setFile(null); onChange() }
          catch (e: any) { setErr(e?.message || String(e)) } finally { setBusy(false) }
        }}>Upload voice</Button>
        {err && <span className="text-xs text-destructive">{err}</span>}
      </div>
      <p className="mb-4 text-xs text-muted-foreground">
        The Speaker ID names this voice — it appears in the scenario voice selectors and is saved with
        every session as <code>target_speaker_id</code>.
      </p>
      <ul className="space-y-1 text-sm">
        {targets.map((t: any) => (
          <li key={t.id} className="flex items-center gap-2">
            <Badge variant="secondary">{t.speaker_id}</Badge>
            <Badge>{t.engine}</Badge>
            <span className="text-xs text-muted-foreground">{t.wav_path?.split("/").pop()}</span>
            <Button size="sm" variant="ghost" className="ml-auto" onClick={async () => { await adminApi.deleteTarget(token, studyId, t.id); onChange() }}>Delete</Button>
          </li>
        ))}
        {targets.length === 0 && <li className="text-sm text-muted-foreground">No voices uploaded.</li>}
      </ul>
    </div>
  )
}

function ParticipantsPanel({ token, studyId, participants, hasScenarios, onChange }: any) {
  const [count, setCount] = useState(5)
  const [err, setErr] = useState<string | null>(null)
  return (
    <div>
      <div className="mb-4 flex items-end gap-2">
        <Labeled label="Count"><Input type="number" value={count} onChange={e => setCount(Number(e.target.value))} className="w-24" /></Labeled>
        <Button size="sm" disabled={!hasScenarios} onClick={async () => {
          setErr(null)
          try { await adminApi.generate(token, studyId, count); onChange() } catch (e: any) { setErr(e?.message || String(e)) }
        }}>Generate codes</Button>
        {!hasScenarios && <span className="text-xs text-muted-foreground">Add a scenario first.</span>}
        {err && <span className="text-xs text-destructive">{err}</span>}
      </div>
      <div className="max-h-72 overflow-auto rounded-md border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-left"><tr><th className="p-2">Participant</th><th className="p-2">Code</th></tr></thead>
          <tbody>
            {participants.map((p: any) => (
              <tr key={p.participant_id} className="border-t">
                <td className="p-2">{p.participant_id}</td>
                <td className="p-2 font-mono">{p.code}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function DataPanel({ token, studyId }: any) {
  const [runs, setRuns] = useState<any[]>([])
  const [sessions, setSessions] = useState<any[]>([])
  useEffect(() => {
    adminApi.runs(token, studyId).then(r => setRuns(r.runs || [])).catch(() => {})
    adminApi.sessions(token, studyId).then(r => setSessions(r.sessions || [])).catch(() => {})
  }, [token, studyId])
  const download = async (fmt: "json" | "zip") => {
    const r = await fetch(adminApi.exportUrl(studyId, fmt), { headers: { "X-Study-Admin-Token": token } })
    const blob = await r.blob()
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob)
    a.download = `study${studyId}_export.${fmt}`; a.click(); URL.revokeObjectURL(a.href)
  }
  return (
    <div>
      <div className="mb-4 flex gap-2">
        <Button size="sm" variant="secondary" onClick={() => download("json")}>Export JSON</Button>
        <Button size="sm" variant="secondary" onClick={() => download("zip")}>Export ZIP</Button>
        <Button size="sm" variant="ghost" onClick={() => adminApi.stopEngine(token)}>Stop VC engine</Button>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <Table title="Runs" head={["Participant", "Status", "Left"]}
          rows={runs.map(r => [r.participant_id, r.status, r.remaining_seconds ? `${Math.floor(r.remaining_seconds / 60)}m` : "—"])} />
        <Table title="Sessions" head={["Session", "Condition", "End"]}
          rows={sessions.map(s => [s.session_id, s.voice_condition, s.end_reason || "—"])} />
      </div>
    </div>
  )
}

function Table({ title, head, rows }: { title: string; head: string[]; rows: any[][] }) {
  return (
    <div>
      <h3 className="mb-1 text-sm font-medium">{title}</h3>
      <div className="max-h-72 overflow-auto rounded-md border text-sm">
        <table className="w-full">
          <thead className="bg-muted/50 text-left"><tr>{head.map(h => <th key={h} className="p-2">{h}</th>)}</tr></thead>
          <tbody>
            {rows.map((r, i) => <tr key={i} className="border-t">{r.map((c, j) => <td key={j} className="p-2 font-mono text-xs">{String(c)}</td>)}</tr>)}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function Labeled({ label, children }: { label: string; children: ReactNode }) {
  return <div className="flex flex-col gap-1"><label className="text-xs text-muted-foreground">{label}</label>{children}</div>
}
