// ============================================================================
//  CONVERSATION-VIEW SOUNDBOARD PANEL — minimal runtime
// ----------------------------------------------------------------------------
//  Just the play buttons + condition filter + timing-log download. Recording,
//  baking, target uploads, engine settings: NONE of that lives here. Configure
//  them in the Configure Soundboard tab and they appear here automatically.
//
//  Architecture: clicks playSlot() in useSoundboardPlayback, which feeds the
//  baked WAV to PP via ws.sendAudio (direct, non-VC path). When the
//  soundboard is in use the conversation should be opened in non-VC mode so
//  baked clips reach PP exactly as baked (no double-conversion).
// ============================================================================

import { useEffect, useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Spinner } from "@/components/ui/spinner"
import { Play, Square, Download, Filter, ListMusic } from "lucide-react"
import { useSoundboard, makeSessionContext, type SessionContext } from "@/hooks/useSoundboard"
import { useSoundboardPlayback } from "@/hooks/useSoundboardPlayback"
import type { useWebSocket } from "@/hooks/useWebSocket"

interface Props {
  ws: ReturnType<typeof useWebSocket>
  // True when VC (MeanVC/X-VC) is enabled for this conversation. Soundboard
  // playback is direct-to-PP (Opus over the 0x01 tag) and CANNOT go through
  // the chat-proxy (which expects raw PCM). If VC is on, we disable the
  // panel and tell the researcher to turn VC off — otherwise PP would receive
  // garbled Opus packets on a raw-PCM channel and misbehave.
  vcEnabled?: boolean
}

export function SoundboardPanel({ ws, vcEnabled }: Props) {
  const sb = useSoundboard()
  const [conditionFilter, setConditionFilter] = useState<string>("__all__")
  const [conditionContext, setConditionContext] = useState("")

  // A new session is minted when the WS connects; lasts for the conversation.
  const [session, setSession] = useState<SessionContext>(() => makeSessionContext(""))
  useEffect(() => {
    if (ws.connected) {
      setSession(makeSessionContext(conditionContext))
    }
    // We deliberately don't mint a new session if conditionContext changes
    // mid-conversation — the user can rotate context label between conversations.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ws.connected])

  // The playback hook does the actual byte-fidelity work; we just give it
  // hooks for the timing log.
  const playback = useSoundboardPlayback({
    ws,
    onPlayEnd: (slot, rec) => {
      void sb.logPlayback(slot, session, rec.startMs, rec.endMs, rec.clipDurationMs)
    },
  })

  const conditions = useMemo(() => {
    const set = new Set<string>()
    for (const s of sb.slots) set.add(s.condition)
    return Array.from(set).sort()
  }, [sb.slots])

  const visible = useMemo(() => {
    const playable = sb.slots.filter((s) => !!(s.baked ?? s.raw))
    if (conditionFilter === "__all__") return playable
    return playable.filter((s) => s.condition === conditionFilter)
  }, [sb.slots, conditionFilter])

  if (sb.loading) {
    return (
      <Card>
        <CardContent className="p-3 text-xs text-muted-foreground flex items-center gap-2">
          <Spinner className="size-3" /> Loading soundboard…
        </CardContent>
      </Card>
    )
  }

  if (sb.slots.length === 0) return null

  return (
    <Card>
      <CardContent className="p-3 space-y-2">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2 text-xs font-medium">
            <ListMusic className="size-3.5" /> Soundboard
            <span className="text-[10px] font-normal text-muted-foreground">
              session: <span className="font-mono">{session.sessionId.slice(0, 16)}…</span>
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <Filter className="size-3 text-muted-foreground" />
              <select
                className="h-7 rounded-md border bg-background px-1.5 text-[11px]"
                value={conditionFilter}
                onChange={(e) => setConditionFilter(e.target.value)}
              >
                <option value="__all__">all conditions</option>
                {conditions.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <Button
              variant="ghost"
              size="xs"
              onClick={() => sb.downloadSessionLog(session.sessionId, "csv")}
              title="Download timing log for this session as CSV"
            >
              <Download className="size-3" /> log
            </Button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            className="flex-1 h-7 rounded-md border bg-background px-2 text-[11px]"
            placeholder="condition context for this session (e.g. exp_2_pilot)"
            value={conditionContext}
            onChange={(e) => setConditionContext(e.target.value)}
            onBlur={() => setSession((s) => ({ ...s, conditionContext }))}
            disabled={ws.connected}
          />
        </div>

        {vcEnabled && (
          <div className="rounded-md border border-amber-500/50 bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-500">
            Soundboard playback is disabled while live VC is enabled. Turn VC
            off in the control panel above — baked clips go direct to PP as
            Opus and cannot be routed through the MeanVC/X-VC chat-proxy.
          </div>
        )}

        <div className="flex flex-wrap gap-1.5">
          {visible.map((slot) => {
            const playing = playback.playingSlotId === slot.id
            return (
              <Button
                key={slot.id}
                variant={playing ? "default" : "outline"}
                size="xs"
                onClick={() => playing ? playback.stop() : playback.playSlot(slot)}
                disabled={
                  !ws.connected ||
                  vcEnabled ||
                  (playback.playingSlotId !== null && !playing)
                }
                className="max-w-[220px] truncate"
                title={`${slot.label} · ${slot.condition} · ${(((slot.bakedDurationMs || slot.rawDurationMs) / 1000)).toFixed(2)}s`}
              >
                {playing ? <Square className="size-3" /> : <Play className="size-3" />}
                <span className="truncate">{slot.label}</span>
                <Badge variant="secondary" className="text-[9px] h-3.5">
                  {slot.condition}
                </Badge>
              </Button>
            )
          })}
          {visible.length === 0 && (
            <p className="text-[10px] text-muted-foreground">
              No baked slots match this filter.
            </p>
          )}
        </div>

        {playback.error && (
          <p className="text-[10px] text-destructive">{playback.error}</p>
        )}
      </CardContent>
    </Card>
  )
}
