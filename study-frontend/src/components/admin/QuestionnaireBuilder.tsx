import { useEffect, useState } from "react"
import { Button } from "@shared/ui/button"
import { adminApi } from "@/api"
import { ItemListEditor, type Item } from "@/components/admin/ItemListEditor"

const SECTIONS: { key: string; label: string }[] = [
  { key: "consent", label: "Audio check" },
  { key: "background", label: "Background" },
  { key: "post", label: "Post (shared)" },
  { key: "final", label: "Final" },
]

export function QuestionnaireBuilder({ token, studyId, questionnaires, scenarios = [], onChange }: {
  token: string; studyId: number; questionnaires: Record<string, Item[]>
  scenarios?: { title: string }[]; onChange: () => void
}) {
  const [q, setQ] = useState<Record<string, Item[]>>(() =>
    Object.fromEntries(SECTIONS.map(s => [s.key, [...(questionnaires[s.key] || [])]])))
  // Re-sync when the study reloads (e.g. after a YAML import).
  useEffect(() => {
    setQ(Object.fromEntries(SECTIONS.map(s => [s.key, [...(questionnaires[s.key] || [])]])))
  }, [questionnaires])
  const [active, setActive] = useState(SECTIONS[0].key)
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const save = async () => {
    setBusy(true); setMsg(null)
    try { await adminApi.setQuestionnaires(token, studyId, q); setMsg("Saved"); onChange() }
    catch (e: any) { setMsg(e?.message || String(e)) } finally { setBusy(false) }
  }

  return (
    <div>
      <p className="mb-3 text-xs text-muted-foreground">
        "Post (shared)" questions are asked after every scenario; scenario-specific post questions live
        on each scenario (Scenarios tab).
      </p>
      <div className="mb-4 flex flex-wrap gap-1">
        {SECTIONS.map(s => (
          <button key={s.key} onClick={() => setActive(s.key)}
            className={`rounded-md px-3 py-1.5 text-sm ${active === s.key ? "bg-primary text-primary-foreground" : "border"}`}>
            {s.label} <span className="opacity-60">({(q[s.key] || []).length})</span>
          </button>
        ))}
      </div>

      <ItemListEditor items={q[active] || []} scenarios={scenarios}
        onChange={items => setQ(prev => ({ ...prev, [active]: items }))} />

      <div className="mt-4 flex items-center gap-3">
        <Button disabled={busy} onClick={save}>{busy ? "Saving…" : "Save questionnaires"}</Button>
        {msg && <span className="text-sm text-muted-foreground">{msg}</span>}
      </div>
    </div>
  )
}
