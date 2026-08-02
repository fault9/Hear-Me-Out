// ============================================================================
//  AUDIT RUNNER SECTION — automated conversations (soundboard)
// ----------------------------------------------------------------------------
//  UI for useSoundboardAudit. Two modes:
//   • Script (default): the eligible slots in soundboard order form one
//     multi-turn conversation, replayed N times in fresh conversations, with a
//     fixed quiet gap between turns. This is the multi-turn turn-taking eval.
//   • Matched (§3.3): shuffled single-turn presentations, matched
//     natural/converted pairs.
//  Rendered only inside the SoundboardPanel (soundboard mode, VC off), so the
//  runner is inherently gated to the direct-to-PP path.
// ============================================================================

import { useState } from "react"
import { Button } from "@shared/ui/button"
import { Badge } from "@shared/ui/badge"
import { Spinner } from "@shared/ui/spinner"
import { Bot, Download, Play, Square, Snowflake } from "lucide-react"
import {
  useSoundboardAudit,
  type AuditMode,
  type AuditRunRecord,
  type AuditSessionRecord,
} from "@/hooks/useSoundboardAudit"
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

const isSession = (r: AuditRunRecord | AuditSessionRecord): r is AuditSessionRecord =>
  "rep" in r

export function SoundboardAuditSection({ ws, playback, slots, onBeforeRun }: Props) {
  const [open, setOpen] = useState(false)
  const audit = useSoundboardAudit({ ws, playback, slots, onBeforeRun })
  const { config, setConfig, manifest, progress } = audit
  const running = progress.running
  const scriptMode = config.mode === "script"
  const set = (patch: Partial<typeof config>) => setConfig({ ...config, ...patch })

  return (
    <div className="mt-3 rounded-md border border-dashed p-2">
      <button
        className="flex w-full items-center gap-2 text-left text-xs font-semibold"
        onClick={() => setOpen((v) => !v)}
        disabled={running}
      >
        <Bot className="size-3.5" />
        Automated runs
        {running && <Spinner className="size-3" />}
        <span className="ml-auto text-[10px] font-normal text-muted-foreground">
          {open ? "hide" : "show"}
        </span>
      </button>

      {(open || running) && (
        <div className="mt-2 space-y-2">
          {/* mode toggle */}
          <div className="flex items-center gap-1 text-[10px]">
            <span className="text-muted-foreground">Mode:</span>
            {(["script", "matched"] as AuditMode[]).map((m) => (
              <button
                key={m}
                disabled={running}
                onClick={() => set({ mode: m })}
                className={`rounded px-2 py-0.5 ${config.mode === m
                  ? "bg-primary text-primary-foreground"
                  : "border text-muted-foreground"}`}
              >
                {m === "script" ? "Scripted conversation" : "Single-turn probes"}
              </button>
            ))}
          </div>

          <p className="text-[10px] leading-4 text-muted-foreground">
            {scriptMode
              ? `One conversation through all ${audit.eligibleSlots.length} lines `
                + `(PP finishes → ${config.interTurnGapMs} ms → next line), `
                + `replayed ${config.reps}× fresh.`
              : `Each slot alone in a fresh conversation, ${config.reps}× each, `
                + `seeded random order.`}
          </p>

          {/* config */}
          <div className="flex flex-wrap items-end gap-2">
            <label className="space-y-0.5 text-[10px]">
              <span className="block text-muted-foreground">
                {scriptMode ? "Replays" : "Reps / slot"}
              </span>
              <input
                type="number" min={1} max={100} value={config.reps} disabled={running}
                onChange={(e) => set({ reps: Math.max(1, Number(e.target.value) || 1) })}
                className="h-7 w-16 rounded-md border bg-background px-2 text-xs"
              />
            </label>
            {scriptMode ? (
              <>
              <label className="space-y-0.5 text-[10px]">
                <span className="block text-muted-foreground">Gap between lines (ms)</span>
                <input
                  type="number" min={0} max={5000} step={100} value={config.interTurnGapMs}
                  disabled={running}
                  onChange={(e) => set({ interTurnGapMs: Math.max(0, Number(e.target.value) || 0) })}
                  className="h-7 w-24 rounded-md border bg-background px-2 text-xs"
                />
              </label>
              <label className="space-y-0.5 text-[10px]"
                     title="PP's utterance counts as finished after this much audible silence (sentence pauses are shorter). The gap between lines starts AFTER this.">
                <span className="block text-muted-foreground">PP finished after quiet (ms)</span>
                <input
                  type="number" min={500} max={5000} step={100} value={config.responseLingerMs}
                  disabled={running}
                  onChange={(e) => set({ responseLingerMs: Math.max(500, Number(e.target.value) || 1500) })}
                  className="h-7 w-24 rounded-md border bg-background px-2 text-xs"
                />
              </label>
              <label className="flex items-center gap-1.5 pb-1 text-[10px]"
                     title="Builds one script per condition tag (exactly two tags) and alternates replays A,B,A,B… so condition isn't confounded with time. Replays = per condition.">
                <input
                  type="checkbox" checked={config.interleaveByCondition} disabled={running}
                  onChange={(e) => set({ interleaveByCondition: e.target.checked })}
                />
                Interleave by condition tag (A/B)
              </label>
              </>
            ) : (
              <label className="space-y-0.5 text-[10px]">
                <span className="block text-muted-foreground">Seed</span>
                <input
                  type="number" value={config.seed} disabled={running}
                  onChange={(e) => set({ seed: Number(e.target.value) || 0 })}
                  className="h-7 w-24 rounded-md border bg-background px-2 text-xs"
                />
              </label>
            )}
            <label className="min-w-40 flex-1 space-y-0.5 text-[10px]"
                   title="PersonaPlex's persona for these conversations. Frozen into the manifest.">
              <span className="block text-muted-foreground">PP persona prompt</span>
              <input
                value={config.prompt} disabled={running}
                onChange={(e) => set({ prompt: e.target.value })}
                className="h-7 w-full rounded-md border bg-background px-2 text-xs"
              />
            </label>
          </div>

          {/* script preview: the ordered turns */}
          {scriptMode && audit.eligibleSlots.length > 0 && (
            <div className="space-y-0.5">
              <p className="text-[10px] text-muted-foreground">
                Script ({audit.eligibleSlots.length} turns, soundboard order):
              </p>
              <ol className="max-h-24 space-y-0.5 overflow-y-auto text-[10px]">
                {audit.eligibleSlots.map((slot, i) => (
                  <li key={slot.id} className="flex items-center gap-2">
                    <span className="w-4 text-right font-mono text-muted-foreground">{i + 1}</span>
                    <span className="truncate">{slot.label}</span>
                    <span className="ml-auto font-mono text-muted-foreground">
                      {slot.manipulation === "vc" ? (slot.engine ?? "vc") : slot.manipulation}
                    </span>
                  </li>
                ))}
              </ol>
              <p className="text-[9px] text-muted-foreground">
                Reorder turns in Configure Soundboard (up/down).
              </p>
            </div>
          )}

          {/* matched-only: per-slot interrupt mode */}
          {!scriptMode && audit.eligibleSlots.length > 0 && (
            <div className="space-y-0.5">
              <p className="text-[10px] text-muted-foreground"
                 title="Checked slots play while PP is mid-speech instead of after silence; PP's yield time is measured.">
                Play during PP speech (interruption items):
              </p>
              <div className="flex flex-wrap gap-x-3 gap-y-0.5">
                {audit.eligibleSlots.map((slot) => (
                  <label key={slot.id} className="flex items-center gap-1 text-[10px]">
                    <input
                      type="checkbox" checked={audit.interruptSlotIds.has(slot.id)}
                      disabled={running}
                      onChange={() => audit.toggleInterruptSlot(slot.id)}
                    />
                    {slot.label}
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* production-engine discipline */}
          {audit.engineWarnings.length > 0 && (
            <div className="space-y-1 rounded-md border border-amber-500/40 bg-amber-500/10 p-2">
              {audit.engineWarnings.map((w) => (
                <p key={w} className="text-[10px] text-amber-600 dark:text-amber-400">{w}</p>
              ))}
              <label className="flex items-center gap-1.5 text-[10px]">
                <input
                  type="checkbox" checked={config.allowNonProductionEngines} disabled={running}
                  onChange={(e) => set({ allowNonProductionEngines: e.target.checked })}
                />
                Allow non-X-VC bakes (exploratory; recorded in the manifest)
              </label>
            </div>
          )}

          {/* freeze + run controls */}
          <div className="flex flex-wrap items-center gap-2">
            <Button size="xs" variant="secondary" disabled={running}
                    onClick={() => void audit.generateManifest()}>
              <Snowflake className="size-3" /> Freeze manifest
            </Button>
            {manifest && (
              <>
                <Badge variant="outline" className="text-[9px] font-mono">
                  {manifest.mode === "script"
                    ? (manifest.interleaved
                        ? `${manifest.scripts?.map((s) =>
                            `${s.condition}:${s.turns.length}t`).join(" / ")} × ${manifest.reps} ea, interleaved`
                        : `${manifest.script?.length ?? 0} turns × ${manifest.reps} replays`)
                    : `${manifest.presentations?.length ?? 0} runs`}
                  {" · "}{manifest.manifest_sha256?.slice(0, 8)}
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

          {/* matched-pair verification (matched mode) */}
          {!scriptMode && manifest && (manifest.pairing_warnings?.length ?? 0) > 0 && (
            <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-2">
              {manifest.pairing_warnings!.map((w) => (
                <p key={w} className="text-[10px] text-amber-600 dark:text-amber-400"
                   title="A pair = one source WAV in two slots (use Duplicate), one baked converted.">
                  {w}
                </p>
              ))}
            </div>
          )}

          {/* server persistence status */}
          {audit.uploadState.status !== "idle" && (
            <p className="flex items-center gap-1.5 text-[10px]">
              {audit.uploadState.status === "uploading" && (
                <><Spinner className="size-3" /> Saving to server…</>
              )}
              {audit.uploadState.status === "done" && (
                <span className="text-emerald-500">Saved on server: data/{audit.uploadState.detail}</span>
              )}
              {audit.uploadState.status === "failed" && (
                <>
                  <span className="text-destructive">Server save failed: {audit.uploadState.detail}</span>
                  <Button size="xs" variant="ghost" onClick={() => void audit.retryUpload()}>Retry</Button>
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
                {progress.records.map((r, i) => {
                  if (isSession(r)) {
                    const okTurns = r.turns.filter((t) => t.status === "ok").length
                    const lat = r.turns.filter((t) => t.response_latency_ms != null)
                      .map((t) => t.response_latency_ms as number)
                    const meanLat = lat.length
                      ? Math.round(lat.reduce((a, b) => a + b, 0) / lat.length) : null
                    return (
                      <div key={i} className="flex items-center gap-2 text-[10px]">
                        <span className="w-10 font-mono text-muted-foreground">rep {r.rep}</span>
                        <span className="truncate">
                          {r.condition ? `[${r.condition}] ` : ""}
                          {okTurns}/{r.turns.length} turns
                          {meanLat != null && ` · ~${meanLat}ms`}
                        </span>
                        <span className={`ml-auto font-mono ${STATUS_TONE[r.status] ?? ""}`}>
                          {r.status}
                        </span>
                      </div>
                    )
                  }
                  return (
                    <div key={i} className="flex items-center gap-2 text-[10px]">
                      <span className="w-8 font-mono text-muted-foreground">
                        {String(r.index).padStart(3, "0")}
                      </span>
                      <span className="truncate">{r.label}</span>
                      <span className={`ml-auto font-mono ${STATUS_TONE[r.status] ?? ""}`}>
                        {r.status}
                        {r.response_latency_ms != null && ` · ${Math.round(r.response_latency_ms)}ms`}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
