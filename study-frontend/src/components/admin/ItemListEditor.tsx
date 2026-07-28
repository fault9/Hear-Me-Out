import { Button } from "@shared/ui/button"
import { Input } from "@shared/ui/input"

export const FIELD_TYPES = ["text", "textarea", "number", "radio", "select", "checkbox", "switch", "scale", "audio_playback", "notice"]
export type Item = Record<string, any>

let _uid = 0
export const newItemId = () => `q_${Date.now()}_${_uid++}`

const csv = (s: string) => s.split(",").map(x => x.trim()).filter(Boolean)

export function ItemListEditor({ items, onChange, scenarios = [] }: {
  items: Item[]; onChange: (items: Item[]) => void; scenarios?: { title: string }[]
}) {
  const hasOptions = (t: string) => t === "radio" || t === "select" || t === "checkbox"
  const update = (i: number, patch: Item) => onChange(items.map((it, j) => j === i ? { ...it, ...patch } : it))
  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir; if (j < 0 || j >= items.length) return
    const c = [...items];[c[i], c[j]] = [c[j], c[i]]; onChange(c)
  }

  return (
    <div className="flex flex-col gap-3">
      {items.map((it, i) => (
        <div key={it.id || i} className="rounded-lg border p-3">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <select className="rounded-md border bg-background px-2 py-1 text-sm" value={it.type || "text"}
              onChange={e => update(i, { type: e.target.value })}>
              {FIELD_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            {it.type !== "audio_playback" && it.type !== "notice" && (
              <label className="flex items-center gap-1 text-xs">
                <input type="checkbox" checked={!!it.required} onChange={e => update(i, { required: e.target.checked })} /> required
              </label>
            )}
            <span className="text-[10px] text-muted-foreground">id: {it.id}</span>
            <div className="ml-auto flex gap-1">
              <Button size="sm" variant="ghost" onClick={() => move(i, -1)}>↑</Button>
              <Button size="sm" variant="ghost" onClick={() => move(i, 1)}>↓</Button>
              <Button size="sm" variant="ghost" onClick={() => onChange(items.filter((_, j) => j !== i))}>✕</Button>
            </div>
          </div>

          <Input value={it.label || ""} onChange={e => update(i, { label: e.target.value })}
            placeholder={it.type === "audio_playback" ? "Instruction text shown above the player (optional)" : "Question label"} />

          {it.type === "audio_playback" && (
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-muted-foreground">Play</span>
              <select className="rounded-md border bg-background px-2 py-1 text-sm"
                value={it.scenario_order ?? ""} onChange={e => update(i, { scenario_order: e.target.value ? Number(e.target.value) : undefined })}>
                <option value="">Auto (VC→natural scenario)</option>
                {scenarios.map((s, j) => <option key={j} value={j + 1}>{`Scenario ${j + 1}: ${s.title || "(untitled)"}`}</option>)}
              </select>
              <select className="rounded-md border bg-background px-2 py-1 text-sm"
                value={it.track || "merged"} onChange={e => update(i, { track: e.target.value })}>
                <option value="merged">Full conversation</option>
                <option value="participant">Participant only</option>
              </select>
              <Input className="w-44" placeholder="Condition id (optional)"
                value={it.condition || ""} onChange={e => update(i, { condition: e.target.value || undefined })} />
              <Input className="w-32" type="number" min={1} max={30}
                placeholder="Max seconds"
                value={it.max_duration_s ?? ""}
                onChange={e => update(i, {
                  max_duration_s: e.target.value ? Number(e.target.value) : undefined,
                })} />
            </div>
          )}

          {hasOptions(it.type) && (
            <>
              <Input className="mt-2" placeholder="Options (comma separated)"
                value={(it.options || []).join(", ")} onChange={e => update(i, { options: csv(e.target.value) })} />
              <div className="mt-2 flex flex-wrap items-center gap-3 text-xs">
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={it.options_source === "scenarios"}
                    onChange={e => update(i, { options_source: e.target.checked ? "scenarios" : undefined })} /> options = scenario names
                </label>
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={!!it.allow_other} onChange={e => update(i, { allow_other: e.target.checked })} /> allow “other” text
                </label>
                {it.allow_other && <Input className="w-40" placeholder="Other label"
                  value={it.other_label || ""} onChange={e => update(i, { other_label: e.target.value })} />}
              </div>
            </>
          )}

          {(it.type === "number" || it.type === "scale") && (
            <div className="mt-2 flex flex-wrap gap-2">
              <Input type="number" className="w-20" placeholder="min" value={it.min ?? ""}
                onChange={e => update(i, { min: e.target.value === "" ? undefined : Number(e.target.value) })} />
              <Input type="number" className="w-20" placeholder="max" value={it.max ?? ""}
                onChange={e => update(i, { max: e.target.value === "" ? undefined : Number(e.target.value) })} />
              {it.type === "scale" && <>
                <Input className="flex-1" placeholder="min label" value={it.min_label || ""} onChange={e => update(i, { min_label: e.target.value })} />
                <Input className="flex-1" placeholder="max label" value={it.max_label || ""} onChange={e => update(i, { max_label: e.target.value })} />
              </>}
            </div>
          )}

          {it.type === "scale" && (
            <Input className="mt-2" placeholder="Extra options (comma separated: Not applicable, Not sure, …)"
              value={(it.extra_options || []).join(", ")} onChange={e => update(i, { extra_options: csv(e.target.value) })} />
          )}

          {it.type !== "audio_playback" && it.type !== "notice" && (
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-muted-foreground">Show only if</span>
              <Input className="w-40" placeholder="question id" value={it.show_if?.field || ""}
                onChange={e => update(i, { show_if: e.target.value ? { field: e.target.value, in: it.show_if?.in || [] } : undefined })} />
              <span className="text-muted-foreground">is one of</span>
              <Input className="flex-1" placeholder="values (comma separated)" value={(it.show_if?.in || []).join(", ")}
                onChange={e => update(i, { show_if: it.show_if?.field ? { field: it.show_if.field, in: csv(e.target.value) } : it.show_if })} />
            </div>
          )}
        </div>
      ))}
      <Button variant="secondary" onClick={() => onChange([...items, { id: newItemId(), type: "text", label: "", required: false }])}>
        + Add question
      </Button>
    </div>
  )
}
