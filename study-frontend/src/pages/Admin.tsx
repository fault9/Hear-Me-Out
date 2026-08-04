import { useCallback, useEffect, useState } from "react"
import { Button } from "@shared/ui/button"
import { Input } from "@shared/ui/input"
import { Badge } from "@shared/ui/badge"
import { adminApi } from "@/api"
import { StudyEditor } from "@/components/admin/StudyEditor"

const TOKEN_KEY = "study_admin_token"

export function Admin() {
  const [token, setToken] = useState(localStorage.getItem(TOKEN_KEY) || "")
  const [authed, setAuthed] = useState(false)
  const [studies, setStudies] = useState<any[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [newName, setNewName] = useState("")
  const [showArchived, setShowArchived] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const loadStudies = useCallback(async (tok: string) => {
    const r = await adminApi.listStudies(tok)
    setStudies(r.studies || [])
    setAuthed(true)
    localStorage.setItem(TOKEN_KEY, tok)
  }, [])

  const signIn = async (tok: string) => {
    setErr(null)
    try { await loadStudies(tok) } catch (e: any) { setErr(e?.message || "Sign-in failed") }
  }

  useEffect(() => { if (token) signIn(token) }, []) // eslint-disable-line

  if (!authed) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-3 px-4">
        <h1 className="text-xl font-semibold">Study admin</h1>
        <div className="flex w-full max-w-sm gap-2">
          <Input type="password" value={token} onChange={e => setToken(e.target.value)}
                 placeholder="Admin token" onKeyDown={e => e.key === "Enter" && signIn(token)} />
          <Button onClick={() => signIn(token)}>Sign in</Button>
        </div>
        {err && <p className="text-sm text-destructive">{err}</p>}
      </div>
    )
  }

  if (selected !== null) {
    return <StudyEditor token={token} studyId={selected} onBack={() => { setSelected(null); loadStudies(token) }} />
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      <div className="mb-6 flex items-center justify-between gap-3">
        <h1 className="text-2xl font-bold">Studies</h1>
        <Button size="sm" variant="ghost" onClick={() => setShowArchived(value => !value)}>
          {showArchived ? "Hide archived" : `Show archived (${studies.filter(s => s.archived).length})`}
        </Button>
      </div>

      <div className="mb-6 flex gap-2">
        <Input value={newName} onChange={e => setNewName(e.target.value)} placeholder="New study name" />
        <Button disabled={!newName.trim()} onClick={async () => {
          try {
            const r = await adminApi.createStudy(token, newName.trim())
            setNewName(""); await loadStudies(token); setSelected(r.study.id)
          } catch (e: any) { setErr(e?.message || String(e)) }
        }}>Create</Button>
      </div>
      {err && <p className="mb-3 text-sm text-destructive">{err}</p>}

      <div className="flex flex-col gap-2">
        {studies.filter(s => showArchived || !s.archived).map(s => (
          <div key={s.id} className="flex items-center justify-between rounded-lg border p-3">
            <div className="flex items-center gap-2">
              <button className="font-medium hover:underline" onClick={() => setSelected(s.id)}>{s.name}</button>
              {s.archived ? <Badge variant="secondary">archived</Badge> : null}
              <span className="text-xs text-muted-foreground">#{s.id}</span>
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="secondary" onClick={() => setSelected(s.id)}>Open</Button>
              {!s.archived && (
                <Button size="sm" variant="ghost" onClick={async () => {
                  // Type-to-confirm: archiving kills every participant code for a
                  // possibly-live study, so a click-through confirm is not enough.
                  const typed = window.prompt(
                    `Archiving “${s.name}” stops ALL of its participant codes from working. `
                    + `Its data is preserved and it can be restored later.\n\n`
                    + `Type the study name to confirm:`)
                  if (typed === null) return
                  if (typed.trim() !== s.name) {
                    window.alert("Name did not match — archive cancelled.")
                    return
                  }
                  await adminApi.archiveStudy(token, s.id); loadStudies(token)
                }}>Archive</Button>
              )}
              {s.archived && (
                <Button size="sm" variant="ghost" onClick={async () => {
                  await adminApi.restoreStudy(token, s.id); loadStudies(token)
                }}>Restore</Button>
              )}
            </div>
          </div>
        ))}
        {studies.filter(s => showArchived || !s.archived).length === 0 && (
          <p className="text-sm text-muted-foreground">
            {studies.length ? "No active studies." : "No studies yet — create one above."}
          </p>
        )}
      </div>
    </div>
  )
}
