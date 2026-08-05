import { useCallback, useEffect, useState, type ReactNode } from "react"
import { Button } from "@shared/ui/button"
import { Input } from "@shared/ui/input"
import { Badge } from "@shared/ui/badge"
import { adminApi } from "@/api"
import { ScenarioEditor } from "@/components/admin/ScenarioEditor"
import { QuestionnaireBuilder } from "@/components/admin/QuestionnaireBuilder"

type Tab = "settings" | "scenarios" | "targets" | "questionnaires" | "participants" | "data"

export function StudyEditor({ token, studyId, onBack }: {
  token: string; studyId: number; onBack: () => void
}) {
  const [study, setStudy] = useState<any>(null)
  const [voices, setVoices] = useState<string[]>([])
  const [engines, setEngines] = useState<string[]>([])
  const [tab, setTab] = useState<Tab>("settings")
  const [err, setErr] = useState<string | null>(null)
  const [liveRun, setLiveRun] = useState<any>(null)

  // Keep the header's slot indicator fresh: whoever holds the single session
  // window shows up here, releasable from any tab.
  useEffect(() => {
    let stop = false
    const check = async () => {
      try {
        const r = await adminApi.runs(token, studyId)
        const live = (r.runs || []).find(
          (x: any) => x.status === "in_progress" && x.remaining_seconds > 0)
        if (!stop) setLiveRun(live || null)
      } catch { /* header indicator is best-effort */ }
    }
    check()
    const t = setInterval(check, 15000)
    return () => { stop = true; clearInterval(t) }
  }, [token, studyId])

  const reload = useCallback(async () => {
    try {
      const r = await adminApi.getStudy(token, studyId)
      setStudy(r.study)
    } catch (e: any) { setErr(e?.message || String(e)) }
  }, [token, studyId])

  const copyPostToAll = async (items: any[]) => {
    const scs: any[] = (study?.scenarios || [])
    await Promise.all(scs.map(s => adminApi.setScenarioPostItems(token, studyId, s.id, items)))
    reload()
  }

  useEffect(() => {
    reload()
    adminApi.voices(token).then(r => setVoices(r.voices)).catch(() => {})
    adminApi.engines(token).then(r => setEngines(r.engines)).catch(() => {})
  }, [reload, token])

  if (!study) return <div className="p-6 text-sm text-muted-foreground">Loading…</div>

  const targets: any[] = study.targets || []
  const scenarios: any[] = study.scenarios || []
  const intakePaused = Boolean(study.settings?.intake_paused)
  const releaseLive = async () => {
    if (!liveRun) return
    if (!window.confirm(`Release ${liveRun.participant_id}'s session window? `
      + `Their progress is kept and they can resume later.`)) return
    try { await adminApi.releaseRun(token, studyId, liveRun.id); setLiveRun(null) }
    catch (e: any) { setErr(e?.message || String(e)) }
  }
  const toggleIntake = async () => {
    try {
      await adminApi.updateStudy(token, studyId, {
        name: study.name,
        settings: { ...(study.settings || {}), intake_paused: !intakePaused },
      })
      reload()
    } catch (e: any) { setErr(e?.message || String(e)) }
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-6">
      <div className="mb-4 flex items-center gap-3">
        <Button size="sm" variant="ghost" onClick={onBack}>← Studies</Button>
        <h1 className="text-xl font-bold">{study.name}</h1>
        <span className="text-xs text-muted-foreground">#{study.id}</span>
        {intakePaused && <Badge variant="destructive">Intake paused</Badge>}
        <div className="ml-auto flex items-center gap-2">
          {liveRun && (
            <Button size="sm" variant="secondary" onClick={releaseLive}>
              Release {liveRun.participant_id}
            </Button>
          )}
          <Button size="sm" variant={intakePaused ? "default" : "ghost"} onClick={toggleIntake}>
            {intakePaused ? "Reopen intake" : "Pause intake"}
          </Button>
        </div>
      </div>
      {err && <p className="mb-3 text-sm text-destructive">{err}</p>}

      <div className="mb-5 flex flex-wrap gap-1 border-b">
        {(["settings", "scenarios", "targets", "questionnaires", "participants", "data"] as Tab[]).map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-3 py-2 text-sm capitalize ${tab === t ? "border-b-2 border-primary font-medium" : "text-muted-foreground"}`}>
            {t}
          </button>
        ))}
      </div>

      {tab === "settings" && (
        <SettingsPanel token={token} studyId={studyId} study={study} onChange={reload} />
      )}

      {tab === "scenarios" && (
        <div className="flex flex-col gap-4">
          {scenarios.map((sc, i) => (
            <ScenarioEditor key={sc.id} token={token} studyId={studyId} scenario={sc} index={i}
              voices={voices} engines={engines} targets={targets} onChange={reload}
              counterbalancing={study.settings?.counterbalancing || {}}
              onGoToTargets={() => setTab("targets")} onCopyPostToAll={copyPostToAll} />
          ))}
          <Button variant="secondary" onClick={async () => {
            await adminApi.addScenario(token, studyId, {
              order_idx: scenarios.length, title: `Scenario ${scenarios.length + 1}`,
              system_prompt: "You are a helpful conversational partner. Keep your replies concise.",
              voice_prompt: voices[0] || "NATF2.pt", time_limit_s: 300,
              scenario_card: {
                how_to_interact: "Speak naturally and use your own words. You may repeat, rephrase, correct, or interrupt the assistant when necessary. You may end the interaction once you believe the goal has been achieved, or if you believe that you cannot make further progress.",
              },
              voice_schedule: [{ mode: "natural", start_s: 0, end_s: null }],
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
          questionnaires={study.questionnaires || {}} scenarios={scenarios} onChange={reload} />
      )}

      {tab === "participants" && (
        <ParticipantsPanel token={token} studyId={studyId}
          participants={study.participants || []} hasScenarios={scenarios.length > 0} onChange={reload} />
      )}

      {tab === "data" && <DataPanel token={token} studyId={studyId} />}
    </div>
  )
}

function SettingsPanel({ token, studyId, study, onChange }: any) {
  const s = study.settings || {}
  const [name, setName] = useState(study.name || "")
  const [welcome, setWelcome] = useState(s.welcome_text || "")
  const [duration, setDuration] = useState(s.estimated_duration || "")
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  // Re-sync fields when the study reloads (e.g. after a YAML import). Safe because
  // a reload only happens on a deliberate action, never mid-typing.
  useEffect(() => {
    const st = study.settings || {}
    setName(study.name || ""); setWelcome(st.welcome_text || ""); setDuration(st.estimated_duration || "")
  }, [study])
  return (
    <div className="flex flex-col gap-3">
      <Labeled label="Study name"><Input value={name} onChange={e => setName(e.target.value)} /></Labeled>
      <Labeled label="Welcome page text (shown to participants before the study)">
        <textarea className="min-h-[200px] w-full rounded-md border bg-background p-3 text-sm"
          value={welcome} onChange={e => setWelcome(e.target.value)} />
      </Labeled>
      <Labeled label="Estimated duration (e.g. 20–30 minutes)">
        <Input value={duration} onChange={e => setDuration(e.target.value)} className="w-64" />
      </Labeled>
      <div className="flex items-center gap-3">
        <Button disabled={busy} onClick={async () => {
          setBusy(true); setMsg(null)
          try {
            await adminApi.updateStudy(token, studyId, { name, settings: { ...s, welcome_text: welcome, estimated_duration: duration } })
            setMsg("Saved"); onChange()
          } catch (e: any) { setMsg(e?.message || String(e)) } finally { setBusy(false) }
        }}>{busy ? "Saving…" : "Save settings"}</Button>
        {msg && <span className="text-sm text-muted-foreground">{msg}</span>}
      </div>

      <YamlPanel token={token} studyId={studyId} onChange={onChange} />
    </div>
  )
}

function YamlPanel({ token, studyId, onChange }: any) {
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  return (
    <div className="mt-4 rounded-lg border p-4">
      <div className="mb-2 text-sm font-semibold">Template & YAML</div>
      <p className="mb-3 text-xs text-muted-foreground">
        Upload the target voices (Targets tab) <b>before importing</b> — the YAML references targets by
        Speaker ID and import fails if any are missing.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="secondary" onClick={() => adminApi.download(token, adminApi.templateUrl(), "pilot_study.yaml")}>Download template</Button>
        <Button size="sm" variant="secondary" onClick={() => adminApi.download(token, adminApi.yamlUrl(studyId), `study${studyId}.yaml`)}>Export this study</Button>
        <label className="cursor-pointer">
          <span className="inline-flex h-8 items-center rounded-md border px-3 text-sm">Import from YAML…</span>
          <input type="file" accept=".yaml,.yml,text/yaml,application/x-yaml" className="hidden"
            onChange={async e => {
              const f = e.target.files?.[0]; if (!f) return
              setBusy(true); setErr(null); setMsg(null)
              try { await adminApi.importYaml(token, studyId, f); setMsg("Imported"); onChange() }
              catch (ex: any) { setErr(ex?.message || String(ex)) } finally { setBusy(false); e.target.value = "" }
            }} />
        </label>
        {busy && <span className="text-xs text-muted-foreground">Importing…</span>}
        {msg && <span className="text-xs text-primary">{msg}</span>}
      </div>
      {err && <p className="mt-2 text-xs text-destructive">{err}</p>}
    </div>
  )
}

function TargetsPanel({ token, studyId, targets, engines, onChange }: any) {
  const [speakerId, setSpeakerId] = useState("")
  const [engine, setEngine] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const selectedEngine = engines.includes(engine) ? engine : (engines[0] || "")

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-end gap-2 rounded-lg border p-3">
        <Labeled label="Speaker ID"><Input value={speakerId} onChange={e => setSpeakerId(e.target.value)} placeholder="e.g. p225" className="w-36" /></Labeled>
        <Labeled label="Engine">
          <select className="rounded-md border bg-background px-2 py-2 text-sm" value={selectedEngine} onChange={e => setEngine(e.target.value)}>
            {engines.map((en: string) => <option key={en} value={en}>{en}</option>)}
          </select>
        </Labeled>
        <input type="file" accept="audio/*" onChange={e => setFile(e.target.files?.[0] || null)} className="text-sm" />
        <Button size="sm" disabled={busy || !speakerId.trim() || !file || !selectedEngine} onClick={async () => {
          if (!file) return
          const id = speakerId.trim()
          setBusy(true); setErr(null)
          // One identifier names the voice: used as the link key, the display name,
          // and the target_speaker_id saved with each session.
          try { await adminApi.uploadTarget(token, studyId, id, id, id, selectedEngine, file); setSpeakerId(""); setFile(null); onChange() }
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
  const [balance, setBalance] = useState<any>(null)
  useEffect(() => {
    adminApi.counterbalance(token, studyId).then(setBalance).catch((e: any) => setErr(e?.message || String(e)))
  }, [token, studyId, participants.length])
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
      {balance?.configured && (
        <div className="mb-4 rounded-md border p-3 text-xs">
          <div className="mb-1 font-semibold">Counterbalance allocation</div>
          <div className="font-mono">{Object.entries(balance.variant_counts || {}).map(([id, n]) => `${id}: ${n}`).join(" · ")}</div>
          {balance.awaiting_profile > 0 && <div className="mt-1">Awaiting gender response: {balance.awaiting_profile}</div>}
          {Object.entries(balance.stratum_variant_counts || {}).map(([stratum, counts]: [string, any]) => (
            <div key={stratum} className="mt-1 font-mono">{stratum}: {Object.entries(counts).map(([id, n]) => `${id}=${n}`).join(" · ")}</div>
          ))}
          {(balance.warnings || []).map((warning: string) => <p key={warning} className="mt-1 text-destructive">{warning}</p>)}
        </div>
      )}
      <div className="max-h-72 overflow-auto rounded-md border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-left"><tr><th className="p-2">Participant</th><th className="p-2">Code</th><th className="p-2">Gender group</th><th className="p-2">Variant</th><th className="p-2">Target</th></tr></thead>
          <tbody>
            {participants.map((p: any) => (
              <tr key={p.participant_id} className="border-t">
                <td className="p-2">{p.participant_id}</td>
                <td className="p-2 font-mono">{p.code}</td>
                <td className="p-2">{p.allocation_stratum || (p.allocation_status === "awaiting_profile" ? "Awaiting response" : "—")}</td>
                <td className="p-2 font-mono">{p.variant_id || (p.allocation_status === "awaiting_profile" ? "pending" : "manual")}</td>
                <td className="p-2 font-mono">{p.target_ref || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

type SessionView = "included" | "analytical" | "all"

function DataPanel({ token, studyId }: any) {
  const [runs, setRuns] = useState<any[]>([])
  const [sessions, setSessions] = useState<any[]>([])
  const [status, setStatus] = useState<any>(null)
  const [vcStatus, setVcStatus] = useState<any>(null)
  const [vcScope, setVcScope] = useState("all")
  const [sessionView, setSessionView] = useState<SessionView>("analytical")
  const load = () => {
    adminApi.runs(token, studyId).then(r => setRuns(r.runs || [])).catch(() => {})
    adminApi.sessions(token, studyId).then(r => setSessions(r.sessions || [])).catch(() => {})
    adminApi.analyzeStatus(token, studyId).then(setStatus).catch(() => {})
    adminApi.vcQualityStatus(token, studyId).then(setVcStatus).catch(() => {})
  }
  useEffect(() => { load() }, [token, studyId]) // eslint-disable-line
  useEffect(() => {
    if (!status?.running) return
    const poll = setInterval(async () => {
      const next = await adminApi.analyzeStatus(token, studyId)
      setStatus(next)
      if (!next.running) load()
    }, 1500)
    return () => clearInterval(poll)
  }, [status?.running, token, studyId]) // eslint-disable-line
  useEffect(() => {
    if (!vcStatus?.running) return
    const poll = setInterval(async () => {
      const next = await adminApi.vcQualityStatus(token, studyId)
      setVcStatus(next)
      if (!next.running) load()
    }, 1500)
    return () => clearInterval(poll)
  }, [vcStatus?.running, token, studyId]) // eslint-disable-line
  const download = async (fmt: "json" | "zip") => {
    const r = await fetch(adminApi.exportUrl(studyId, fmt), { headers: { "X-Study-Admin-Token": token } })
    const blob = await r.blob()
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob)
    a.download = `study${studyId}_export.${fmt}`; a.click(); URL.revokeObjectURL(a.href)
  }
  const [opErr, setOpErr] = useState<string | null>(null)
  const runAnalysis = async (force: boolean) => {
    setOpErr(null)
    try { setStatus(await adminApi.analyze(token, studyId, force)) }
    catch (e: any) { setOpErr(e?.message || String(e)) }
  }
  const analyticalSessions = sessions.filter(s => s.analysis_eligible !== false)
  // Only sessions that actually ended can be analyzed: abnormal terminations
  // never send the end call, and "incomplete" validity is terminal — neither
  // can ever finish the pipeline, so they don't count as pending.
  const pending = analyticalSessions.filter(
    s => s.ended_at != null
      && (s.technical_validity?.status || "pending") !== "incomplete"
      && (!s.metrics || !["complete", "skipped"].includes(s.vc_quality_status || "")
          || (s.technical_validity?.status || "pending") === "pending")).length
  const running = status?.running
  const vcRunning = vcStatus?.running
  const phaseLabel = status?.phase === "vc_quality" ? "VC quality" : "Transcription and timing"
  const runVCQuality = async (force: boolean) => {
    const body: { participant_id?: string; session_id?: string; force?: boolean } = { force }
    if (vcScope.startsWith("participant:")) body.participant_id = vcScope.slice("participant:".length)
    if (vcScope.startsWith("session:")) body.session_id = vcScope.slice("session:".length)
    setOpErr(null)
    try { setVcStatus(await adminApi.vcQuality(token, studyId, body)) }
    catch (e: any) { setOpErr(e?.message || String(e)) }
  }
  const participants = Array.from(new Set(
    analyticalSessions.map(s => s.participant_id)))
  const includedSessions = analyticalSessions.filter(s => s.analysis_included)
  // "pending" = analytical but not yet decidable (unanalyzed, or its run is still open)
  const pendingSessions = analyticalSessions.filter(s => !s.analysis_included
    && (s.analysis_exclusion_reasons || []).some(
      (r: string) => r === "technical_validity_pending" || r === "no_submitted_run"))
  const excludedCount = analyticalSessions.length - includedSessions.length - pendingSessions.length
  const practiceCount = sessions.length - analyticalSessions.length
  const visibleSessions = sessionView === "all" ? sessions
    : sessionView === "analytical" ? analyticalSessions
    : includedSessions

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <Button size="sm" variant="secondary" onClick={() => download("json")}>Export JSON</Button>
        <Button size="sm" variant="secondary" onClick={() => download("zip")}>Export ZIP</Button>
        <Button size="sm" variant="ghost" onClick={() => adminApi.stopEngine(token)}>Stop VC engine</Button>
      </div>
      {opErr && <p className="mb-3 text-sm text-destructive">{opErr}</p>}

      <div className="mb-4 rounded-lg border p-3">
        <div className="mb-1 text-sm font-semibold">Analysis pipeline</div>
        <p className="mb-2 text-xs text-muted-foreground">
          Processes every analytical capture, creates a technical-validity report, then runs VC-quality scoring.
          Dataset inclusion separately selects the latest valid attempt from the latest submitted run.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" disabled={running || vcRunning} onClick={() => runAnalysis(false)}>
            {running ? `${phaseLabel} ${status.done}/${status.total}…` : `Run analysis (${pending} pending)`}
          </Button>
          <Button size="sm" variant="ghost" disabled={running || vcRunning}
            onClick={() => runAnalysis(true)}>Create all new snapshots</Button>
          {running && status.current && <span className="text-xs text-muted-foreground">{status.current}</span>}
          {!running && status?.error && <span className="text-xs text-destructive">{status.error}</span>}
        </div>
      </div>

      <div className="mb-4 rounded-lg border p-3">
        <div className="mb-1 text-sm font-semibold">VC-quality rerun</div>
        <p className="mb-2 text-xs text-muted-foreground">
          Optional scoped rerun using the frozen target, raw microphone, transmitted audio, and route events.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <select className="rounded-md border bg-background px-2 py-1.5 text-sm" value={vcScope}
            onChange={e => setVcScope(e.target.value)} disabled={running || vcRunning}>
            <option value="all">All analytical sessions</option>
            {participants.map(pid => <option key={pid} value={`participant:${pid}`}>Participant {pid}</option>)}
            {analyticalSessions.map(s => <option key={s.session_id} value={`session:${s.session_id}`}>Session {s.session_id}</option>)}
          </select>
          <Button size="sm" disabled={running || vcRunning} onClick={() => runVCQuality(false)}>
            {vcRunning ? `Scoring ${vcStatus.done}/${vcStatus.total}…` : "Run VC quality"}
          </Button>
          <Button size="sm" variant="ghost" disabled={running || vcRunning}
            onClick={() => runVCQuality(true)}>Create new snapshot</Button>
          {vcRunning && vcStatus.current && <span className="text-xs text-muted-foreground">{vcStatus.current}</span>}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Table title="Runs" head={["Participant", "Status", "Left", ""]}
          rows={runs.map(r => [r.participant_id, r.status,
            r.remaining_seconds ? `${Math.floor(r.remaining_seconds / 60)}m` : "—",
            r.status === "in_progress" && r.remaining_seconds > 0 ? (
              <Button size="sm" variant="ghost" onClick={async () => {
                if (!window.confirm(`Release ${r.participant_id}'s session window? Their progress is kept and they can resume later.`)) return
                try { await adminApi.releaseRun(token, studyId, r.id); load() }
                catch (e: any) { setOpErr(e?.message || String(e)) }
              }}>Release</Button>
            ) : ""])} />
        <Table title="Sessions" head={["Session", "Condition", "Preprocess", "VC quality", "Technical", "Dataset"]}
          controls={<ViewToggle value={sessionView} onChange={setSessionView} />}
          footer={`${includedSessions.length} included · ${excludedCount} excluded · ${pendingSessions.length} pending · ${practiceCount} practice`}
          rows={visibleSessions.map(s => {
            const excluded = s.analysis_eligible === false
            const validity = s.technical_validity?.status || "pending"
            const reviewLabels = validity === "valid" ? [
              s.technical_validity?.valid_for_timing_reconstruction === false
                ? "timing review" : null,
              s.technical_validity?.valid_for_post_checkpoint_analysis === false
                ? "no post-checkpoint window" : null,
            ].filter(Boolean) : []
            const validityLabel = reviewLabels.length
              ? `valid; ${reviewLabels.join("; ")}`
              : validity
            const dataset = s.analysis_included
              ? "included"
              : (s.analysis_exclusion_reasons || []).join(", ") || "pending"
            return [s.session_id, s.voice_condition,
              excluded ? "excluded" : (s.metrics ? "analyzed" : "pending"),
              excluded ? "excluded" : (s.vc_quality_status || "pending"),
              excluded ? "excluded" : validityLabel,
              dataset]
          })} />
      </div>
    </div>
  )
}

function ViewToggle({ value, onChange }: { value: SessionView; onChange: (v: SessionView) => void }) {
  const views: [SessionView, string][] = [["included", "Included"], ["analytical", "Analytical"], ["all", "All"]]
  return (
    <div className="flex overflow-hidden rounded-md border">
      {views.map(([id, label]) => (
        <button key={id} onClick={() => onChange(id)}
          className={`px-2 py-0.5 text-xs ${value === id ? "bg-muted font-medium" : "text-muted-foreground hover:bg-muted/50"}`}>
          {label}
        </button>
      ))}
    </div>
  )
}

function Table({ title, head, rows, controls, footer }: {
  title: string; head: string[]; rows: any[][]; controls?: ReactNode; footer?: string
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium">{title}</h3>
        {controls}
      </div>
      <div className="max-h-72 overflow-auto rounded-md border text-sm">
        <table className="w-full">
          <thead className="bg-muted/50 text-left"><tr>{head.map(h => <th key={h} className="p-2">{h}</th>)}</tr></thead>
          <tbody>
            {rows.map((r, i) => <tr key={i} className="border-t">{r.map((c, j) => <td key={j} className="p-2 font-mono text-xs">{c}</td>)}</tr>)}
          </tbody>
        </table>
      </div>
      {footer && <p className="mt-1 text-xs text-muted-foreground">{footer}</p>}
    </div>
  )
}

export function Labeled({ label, children }: { label: string; children: ReactNode }) {
  return <div className="flex flex-col gap-1"><label className="text-xs text-muted-foreground">{label}</label>{children}</div>
}
