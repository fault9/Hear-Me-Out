// ============================================================================
//  AUDIT RUNNER SECTION — automated matched-input presentations (soundboard)
// ----------------------------------------------------------------------------
//  UI for useSoundboardAudit: configure → generate & freeze the manifest →
//  run N fresh-conversation presentations unattended → download the results
//  zip. Rendered only inside the SoundboardPanel (soundboard mode), and the
//  panel only mounts with VC off, so the auto mode is inherently gated to
//  the soundboard's direct-to-PP path.
// ============================================================================

import { useState } from "react"
import { Button } from "@shared/ui/button"
import { Badge } from "@shared/ui/badge"
import { Spinner } from "@shared/ui/spinner"
import { Bot, Download, Play, Square, Snowflake } from "lucide-react"
import { useSoundboardAudit } from "@/hooks/useSoundboardAudit"
import type { Slot } from "@/lib/soundboardDb"
import type { useWebSocket } from "@shared/hooks/useWebSocket"
import type { useSoundboardPlayback } from "@/hooks/useSoundboardPlayback"

interface Props {
  ws: ReturnType<typeof useWebSocket>
  playback: ReturnType<typeof useSoundboardPlayback>
  slots: Slot[]
  onBeforeRun?: () => void
}

const STATUS_TONE: Record<string, string> = {
  ok: "text-emerald-500",
  no_response: "text-amber-500",
  handshake_timeout: "text-destructive",
  clip_error: "text-destructive",
  error: "text-destructive",
  aborted: "text-muted-foreground",
}

export function SoundboardAuditSection({ ws, playback, slots, onBeforeRun }: Props) {
  const [open, setOpen] = useState(false)
  const audit = useSoundboardAudit({ ws, playback, slots, onBeforeRun })
  const { config, setConfig, manifest, progress } = audit
  const running = progress.running

  return (
    <div className="mt-3 rounded-md border border-dashed p-2">
      <button
        className="flex w-full items-center gap-2 text-left text-xs font-semibold"
        onClick={() => setOpen((v) => !v)}
        disabled={running}
      >
        <Bot className="size-3.5" />
        Audit runner — automated presentations
        {running && <Spinner className="size-3" />}
        <span className="ml-auto text-[10px] font-normal text-muted-foreground">
          {open ? "hide" : "show"}
        </span>
      </button>

      {(open || running) && (
        <div className="mt-2 space-y-2">
          <p className="text-[10px] leading-4 text-muted-foreground">
            Runs every playable slot {config.reps}× in seeded random order, each
            in a FRESH PersonaPlex conversation (connect → greeting → clip → PP
            response → collect → disconnect). Freeze the manifest first; results
            (manifest + per-run PP/sent audio + timing log) download as one zip.
          </p>

          {/* config */}
          <div className="flex flex-wrap items-end gap-2">
            <label className="space-y-0.5 text-[10px]">
              <span className="block text-muted-foreground">Reps / slot</span>
              <input
                type="number" min={1} max={10} value={config.reps}
                disabled={running}
                onChange={(e) => setConfig({ ...config, reps: Math.max(1, Number(e.target.value) || 1) })}
                className="h-7 w-16 rounded-md border bg-background px-2 text-xs"
              />
            </label>
            <label className="space-y-0.5 text-[10px]">
              <span className="block text-muted-foreground">Seed</span>
              <input
                type="number" value={config.seed}
                disabled={running}
                onChange={(e) => setConfig({ ...config, seed: Number(e.target.value) || 0 })}
                className="h-7 w-24 rounded-md border bg-background px-2 text-xs"
              />
            </label>
            <label className="min-w-40 flex-1 space-y-0.5 text-[10px]">
              <span className="block text-muted-foreground">PP text prompt (recorded in manifest)</span>
              <input
                value={config.prompt}
                disabled={running}
                onChange={(e) => setConfig({ ...config, prompt: e.target.value })}
                className="h-7 w-full rounded-md border bg-background px-2 text-xs"
              />
            </label>
          </div>

          {/* production-engine discipline */}
          {audit.engineWarnings.length > 0 && (
            <div className="space-y-1 rounded-md border border-amber-500/40 bg-amber-500/10 p-2">
              {audit.engineWarnings.map((w) => (
                <p key={w} className="text-[10px] text-amber-600 dark:text-amber-400">{w}</p>
              ))}
              <label className="flex items-center gap-1.5 text-[10px]">
                <input
                  type="checkbox"
                  checked={config.allowNonProductionEngines}
                  disabled={running}
                  onChange={(e) => setConfig({ ...config, allowNonProductionEngines: e.target.checked })}
                />
                Exploratory run — allow non-production VC engines (recorded in
                the manifest; NOT valid as the frozen study audit)
              </label>
            </div>
          )}

          {/* manifest freeze + run controls */}
          <div className="flex flex-wrap items-center gap-2">
            <Button size="xs" variant="secondary" disabled={running}
                    onClick={() => void audit.generateManifest()}>
              <Snowflake className="size-3" /> Freeze manifest
            </Button>
            {manifest && (
              <>
                <Badge variant="outline" className="text-[9px] font-mono">
                  {manifest.presentations.length} runs · {manifest.manifest_sha256?.slice(0, 8)}
                </Badge>
                <Button size="xs" variant="ghost" disabled={running}
                        onClick={audit.downloadManifest} title="Download the frozen manifest JSON">
                  <Download className="size-3" />
                </Button>
              </>
            )}
            {!running ? (
              <Button size="xs" disabled={!manifest || ws.connected}
                      title={ws.connected ? "Stop the open conversation first" : undefined}
                      onClick={() => void audit.start()}>
                <Play className="size-3" /> Run
              </Button>
            ) : (
              <Button size="xs" variant="destructive" onClick={audit.abort}>
                <Square className="size-3" /> Abort
              </Button>
            )}
            {audit.resultsZip && !running && (
              <Button size="xs" variant="secondary" onClick={audit.downloadResults}>
                <Download className="size-3" /> Results zip
              </Button>
            )}
          </div>

          {/* server persistence status (STUDY_DATA_ROOT/audit/…) */}
          {audit.uploadState.status !== "idle" && (
            <p className="flex items-center gap-1.5 text-[10px]">
              {audit.uploadState.status === "uploading" && (
                <><Spinner className="size-3" /> Saving to server…</>
              )}
              {audit.uploadState.status === "done" && (
                <span className="text-emerald-500">
                  Saved on server: data/{audit.uploadState.detail}
                </span>
              )}
              {audit.uploadState.status === "failed" && (
                <>
                  <span className="text-destructive">
                    Server save failed: {audit.uploadState.detail}
                  </span>
                  <Button size="xs" variant="ghost" onClick={() => void audit.retryUpload()}>
                    Retry
                  </Button>
                </>
              )}
            </p>
          )}

          {audit.error && <p className="text-[10px] text-destructive">{audit.error}</p>}

          {/* live progress */}
          {(running || progress.records.length > 0) && (
            <div className="space-y-1">
              <p className="text-[10px] text-muted-foreground">
                {running ? progress.phase
                  : `${progress.phase} — ${progress.records.filter((r) => r.status === "ok").length}/${progress.records.length} ok`}
              </p>
              <div className="max-h-32 space-y-0.5 overflow-y-auto">
                {progress.records.map((r) => (
                  <div key={r.index} className="flex items-center gap-2 text-[10px]">
                    <span className="w-8 font-mono text-muted-foreground">
                      {String(r.index).padStart(3, "0")}
                    </span>
                    <span className="truncate">{r.label}</span>
                    <span className={`ml-auto font-mono ${STATUS_TONE[r.status] ?? ""}`}>
                      {r.status}
                      {r.response_latency_ms != null && ` · ${Math.round(r.response_latency_ms)}ms`}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
