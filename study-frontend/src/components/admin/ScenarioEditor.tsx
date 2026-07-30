import { useMemo, useState } from "react"
import { Button } from "@shared/ui/button"
import { Input } from "@shared/ui/input"
import { adminApi } from "@/api"
import { ItemListEditor } from "@/components/admin/ItemListEditor"

type Preset = "natural" | "vc" | "natural_vc" | "vc_natural"

const PRESET_LABELS: Record<Preset, string> = {
  natural: "Natural only",
  vc: "VC only",
  natural_vc: "Natural → VC",
  vc_natural: "VC → Natural",
}

function scheduleFromPreset(preset: Preset, engine: string, target: string, switchS: number): any[] {
  if (preset === "natural") return [{ mode: "natural", start_s: 0, end_s: null }]
  if (preset === "vc") return [{ mode: "vc", engine, target_ref: target, start_s: 0, end_s: null }]
  if (preset === "natural_vc") return [
    { mode: "natural", start_s: 0, end_s: switchS },
    { mode: "vc", engine, target_ref: target, start_s: switchS, end_s: null }]
  return [
    { mode: "vc", engine, target_ref: target, start_s: 0, end_s: switchS },
    { mode: "natural", start_s: switchS, end_s: null }]
}

function presetFromSchedule(schedule: any[]): { preset: Preset; engine: string; target: string; switchS: number } {
  const segs = schedule || []
  const vc = segs.find(s => s.mode === "vc") || {}
  const engine = vc.engine || "meanvc"
  const target = vc.target_ref || ""
  if (segs.length <= 1) {
    const only = segs[0]?.mode || "natural"
    return { preset: only === "vc" ? "vc" : "natural", engine, target, switchS: 25 }
  }
  const first = segs[0]?.mode
  const switchS = segs[0]?.end_s ?? 25
  return { preset: first === "natural" ? "natural_vc" : "vc_natural", engine, target, switchS }
}

function describeSchedule(schedule: any[]): string {
  const parsed = presetFromSchedule(schedule)
  const suffix = parsed.preset === "natural_vc" || parsed.preset === "vc_natural"
    ? ` at ${parsed.switchS} s`
    : ""
  return `${PRESET_LABELS[parsed.preset]}${suffix}`
}

export function ScenarioEditor({ token, studyId, scenario, index, voices, engines, targets,
  counterbalancing, onChange, onGoToTargets, onCopyPostToAll }: any) {
  const [open, setOpen] = useState(false)
  const card = scenario.scenario_card || {}
  const init = useMemo(() => presetFromSchedule(scenario.voice_schedule), [scenario.id])
  const conditions = counterbalancing?.conditions || {}
  const variantSchedules = (counterbalancing?.variants || []).map((variant: any) => {
    const conditionId = variant.condition_assignment?.[index + 1]
    const condition = conditions[conditionId]
    const schedule = Array.isArray(condition) ? condition : condition?.voice_schedule
    return {
      variantId: String(variant.id || ""),
      conditionId: String(conditionId || ""),
      schedule: Array.isArray(schedule) ? schedule : [],
    }
  }).filter((row: any) => row.variantId && row.conditionId && row.schedule.length)
  const usesCounterbalancedSchedule = variantSchedules.length > 0

  const [title, setTitle] = useState(scenario.title || "")
  const [role, setRole] = useState(card.role || "")
  const [situation, setSituation] = useState(card.current_situation || "")
  const [goal, setGoal] = useState(card.goal || "")
  const [howTo, setHowTo] = useState(card.how_to_interact || "")
  const [firstLine, setFirstLine] = useState(card.suggested_first_line || "")
  const [additional, setAdditional] = useState(card.additional_details || "")
  const [prompt, setPrompt] = useState(scenario.system_prompt || "")
  const [extraFields, setExtraFields] = useState<{ label: string; value: string }[]>(card.extra_fields || [])
  const [postItems, setPostItems] = useState<any[]>(scenario.post_items || [])
  const [voicePrompt, setVoicePrompt] = useState(scenario.voice_prompt || voices[0] || "NATF2.pt")
  const [timeLimit, setTimeLimit] = useState(scenario.time_limit_s || 300)
  const [preset, setPreset] = useState<Preset>(init.preset)
  const [engine, setEngine] = useState(init.engine)
  const [target, setTarget] = useState(init.target)
  const [switchS, setSwitchS] = useState(init.switchS)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const needsTarget = preset !== "natural"
  const needsSwitch = preset === "natural_vc" || preset === "vc_natural"
  const engineTargets = targets.filter((t: any) => t.engine === engine)

  const save = async () => {
    const required: [string, string][] = [
      ["Title", title], ["System prompt", prompt], ["Your role", role],
      ["Current situation", situation], ["Your goal", goal], ["How to interact", howTo],
      ["Suggested first line", firstLine], ["Additional details", additional],
    ]
    const missing = required.filter(([, v]) => !v.trim()).map(([l]) => l)
    if (missing.length) { setErr("Please fill in: " + missing.join(", ")); return }
    if (needsTarget && !target) { setErr("Pick a target voice for this VC scenario."); return }
    setBusy(true); setErr(null)
    try {
      await adminApi.updateScenario(token, studyId, scenario.id, {
        order_idx: scenario.order_idx ?? index, title,
        scenario_card: {
          role, current_situation: situation, goal, how_to_interact: howTo,
          suggested_first_line: firstLine, additional_details: additional,
          extra_fields: extraFields.filter(f => f.label.trim()),
        },
        system_prompt: prompt, voice_prompt: voicePrompt, time_limit_s: Number(timeLimit),
        voice_schedule: usesCounterbalancedSchedule
          ? scenario.voice_schedule
          : scheduleFromPreset(preset, engine, target, Number(switchS)),
        post_items: postItems,
      })
      onChange()
    } catch (e: any) { setErr(e?.message || String(e)) } finally { setBusy(false) }
  }

  return (
    <div className="rounded-lg border">
      <button className="flex w-full items-center justify-between px-4 py-3 text-left" onClick={() => setOpen(o => !o)}>
        <span className="font-medium">{index + 1}. {title || "Untitled scenario"}</span>
        <span className="text-xs text-muted-foreground">
          {usesCounterbalancedSchedule
            ? "Assigned by counterbalancing"
            : `${PRESET_LABELS[preset]}${needsTarget ? ` · ${engine}` : ""}`}
        </span>
      </button>
      {open && (
        <div className="flex flex-col gap-3 border-t p-4">
          <p className="text-xs text-muted-foreground">All fields below are required.</p>
          <Field label="Title *"><Input value={title} onChange={e => setTitle(e.target.value)} /></Field>
          <Field label="Your role *"><Textarea value={role} onChange={setRole} /></Field>
          <Field label="Current situation *"><Textarea value={situation} onChange={setSituation} /></Field>
          <Field label="Your goal *"><Textarea value={goal} onChange={setGoal} /></Field>
          <Field label="How to interact *"><Textarea value={howTo} onChange={setHowTo} /></Field>
          <Field label="Suggested first line *"><Textarea value={firstLine} onChange={setFirstLine} /></Field>
          <Field label="Additional details *"><Textarea value={additional} onChange={setAdditional} /></Field>
          <Field label="System prompt (hidden from participant) *"><Textarea value={prompt} onChange={setPrompt} /></Field>

          <div className="rounded-md border p-3">
            <div className="mb-2 text-xs font-semibold uppercase text-muted-foreground">Extra fields (shown on the scenario card)</div>
            <div className="flex flex-col gap-2">
              {extraFields.map((f, i) => (
                <div key={i} className="flex gap-2">
                  <Input className="w-40" placeholder="Label" value={f.label}
                    onChange={e => setExtraFields(a => a.map((x, j) => j === i ? { ...x, label: e.target.value } : x))} />
                  <Input className="flex-1" placeholder="Value" value={f.value}
                    onChange={e => setExtraFields(a => a.map((x, j) => j === i ? { ...x, value: e.target.value } : x))} />
                  <Button size="sm" variant="ghost" onClick={() => setExtraFields(a => a.filter((_, j) => j !== i))}>✕</Button>
                </div>
              ))}
              <Button size="sm" variant="secondary" className="self-start"
                onClick={() => setExtraFields(a => [...a, { label: "", value: "" }])}>+ Add field</Button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Field label="Assistant voice">
              <Select value={voicePrompt} onChange={setVoicePrompt} options={voices} />
            </Field>
            <Field label="Time limit (s)">
              <Input type="number" value={timeLimit} onChange={e => setTimeLimit(Number(e.target.value))} />
            </Field>
          </div>

          <div className="rounded-md border p-3">
            <div className="mb-2 text-xs font-semibold uppercase text-muted-foreground">Voice schedule</div>
            {usesCounterbalancedSchedule ? (
              <div>
                <div className="divide-y border-y text-sm">
                  {variantSchedules.map((row: any) => (
                    <div key={row.variantId} className="grid grid-cols-[4rem_minmax(0,1fr)_minmax(0,1fr)] gap-3 py-2">
                      <span className="font-mono font-medium">{row.variantId}</span>
                      <span>{row.conditionId.replaceAll("_", " ")}</span>
                      <span className="text-muted-foreground">{describeSchedule(row.schedule)}</span>
                    </div>
                  ))}
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  The YAML assigns this schedule per counterbalance variant.
                  {counterbalancing?.target_assignment
                    ? " VC segments use the participant target assigned from the background questionnaire."
                    : ""}
                </p>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <Field label="Mode">
                    <Select value={preset} onChange={v => setPreset(v as Preset)}
                      options={["natural", "vc", "natural_vc", "vc_natural"]}
                      labels={PRESET_LABELS} />
                  </Field>
                  {needsTarget && (
                    <Field label="Engine"><Select value={engine} onChange={v => { setEngine(v); setTarget("") }} options={engines} /></Field>
                  )}
                  {needsTarget && (
                    <Field label="Target voice">
                      <Select value={target} onChange={setTarget}
                        options={engineTargets.map((t: any) => t.ref)}
                        labels={Object.fromEntries(engineTargets.map((t: any) => [t.ref, t.speaker_id]))}
                        placeholder="Select voice" />
                    </Field>
                  )}
                  {needsSwitch && (
                    <Field label="Switch at (s)"><Input type="number" value={switchS} onChange={e => setSwitchS(Number(e.target.value))} /></Field>
                  )}
                </div>
                {needsTarget && engineTargets.length === 0 && (
                  <p className="mt-2 text-xs text-destructive">
                    No {engine} voices uploaded —{" "}
                    <button type="button" className="underline" onClick={onGoToTargets}>go to the Targets tab</button> to add one.
                  </p>
                )}
              </>
            )}
          </div>

          <div className="rounded-md border p-3">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-xs font-semibold uppercase text-muted-foreground">Scenario-specific post questions</div>
              {onCopyPostToAll && (
                <Button size="sm" variant="ghost" onClick={() => onCopyPostToAll(postItems)}>Copy to all scenarios</Button>
              )}
            </div>
            <p className="mb-2 text-xs text-muted-foreground">Shown after this scenario, in addition to the shared post questions.</p>
            <ItemListEditor items={postItems} onChange={setPostItems} />
          </div>

          {err && <p className="text-sm text-destructive">{err}</p>}
          <div className="flex gap-2">
            <Button size="sm" disabled={busy} onClick={save}>{busy ? "Saving…" : "Save scenario"}</Button>
            <Button size="sm" variant="ghost" onClick={async () => { await adminApi.deleteScenario(token, studyId, scenario.id); onChange() }}>Delete</Button>
          </div>
        </div>
      )}
    </div>
  )
}

function Field({ label, children }: { label: string; children: any }) {
  return <div className="flex flex-col gap-1"><label className="text-xs text-muted-foreground">{label}</label>{children}</div>
}
function Textarea({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return <textarea className="min-h-[60px] w-full rounded-md border bg-background px-3 py-2 text-sm"
    value={value} onChange={e => onChange(e.target.value)} />
}
function Select({ value, onChange, options, labels, placeholder }: {
  value: string; onChange: (v: string) => void; options: string[]; labels?: Record<string, string>; placeholder?: string
}) {
  return (
    <select className="w-full rounded-md border bg-background px-2 py-2 text-sm" value={value} onChange={e => onChange(e.target.value)}>
      {placeholder && <option value="">{placeholder}</option>}
      {options.map(o => <option key={o} value={o}>{labels?.[o] ?? o}</option>)}
    </select>
  )
}
