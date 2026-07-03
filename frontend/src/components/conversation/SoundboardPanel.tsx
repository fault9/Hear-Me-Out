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

import { useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Spinner } from "@/components/ui/spinner"
import { Play, Square, Download, Filter, ListMusic, Headphones, Pause, Volume2, VolumeX } from "lucide-react"
import { useSoundboard, makeSessionContext, type SessionContext } from "@/hooks/useSoundboard"
import { useSoundboardPlayback } from "@/hooks/useSoundboardPlayback"
import type { useWebSocket } from "@/hooks/useWebSocket"
import type { Slot } from "@/lib/soundboardDb"

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

  // Trail of slots played this session — greyed out so the researcher sees
  // what's already been triggered, but still fully clickable (a turn can be
  // replayed). Reset when a new conversation starts.
  const [playedIds, setPlayedIds] = useState<Set<string>>(new Set())

  // A new session is minted when the WS connects; lasts for the conversation.
  const [session, setSession] = useState<SessionContext>(() => makeSessionContext(""))
  useEffect(() => {
    if (ws.connected) {
      setSession(makeSessionContext(conditionContext))
      setPlayedIds(new Set())
    }
    // We deliberately don't mint a new session if conditionContext changes
    // mid-conversation — the user can rotate context label between conversations.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ws.connected])

  // Log PP speech-start/-end events into the same session, on the same
  // performance.now() clock as slot playback (RQ2b: latency/overlap/barge-in).
  // Register once against the stable ws callback; read the current session +
  // log fn via refs so we don't re-subscribe on every render.
  const sessionRef = useRef(session)
  sessionRef.current = session
  const logPpEventRef = useRef(sb.logPpEvent)
  logPpEventRef.current = sb.logPpEvent
  const { registerPpSpeechListener } = ws
  useEffect(() => {
    return registerPpSpeechListener((e) => {
      void logPpEventRef.current(sessionRef.current, e.type, e.timestampMs)
    })
  }, [registerPpSpeechListener])

  // The playback hook does the actual byte-fidelity work; we just give it
  // hooks for the timing log + the played-trail.
  const playback = useSoundboardPlayback({
    ws,
    onPlayStart: (slot) => {
      setPlayedIds((prev) => {
        const next = new Set(prev)
        next.add(slot.id)
        return next
      })
    },
    onPlayEnd: (slot, rec) => {
      void sb.logPlayback(slot, session, rec.startMs, rec.endMs, rec.clipDurationMs)
    },
  })

  // Local "hear this slot" preview — audio goes to the researcher's speakers
  // ONLY, never touches PP. Useful for auditing a clip before triggering it.
  // Shared previewRef ensures only one local preview at a time; clicking the
  // active one pauses.
  const previewRef = useRef<{ stop: () => void } | null>(null)
  const [previewingId, setPreviewingId] = useState<string | null>(null)
  const togglePreview = async (slot: Slot) => {
    if (previewingId === slot.id) {
      previewRef.current?.stop()
      return
    }
    previewRef.current?.stop()
    const blob = slot.baked ?? slot.raw
    if (!blob) return
    setPreviewingId(slot.id)
    const handle = await sb.previewBlob(blob, () => {
      setPreviewingId((cur) => (cur === slot.id ? null : cur))
      previewRef.current = null
    })
    previewRef.current = handle
  }

  const conditions = useMemo(() => {
    const set = new Set<string>()
    for (const s of sb.slots) set.add(s.condition)
    return Array.from(set).sort()
  }, [sb.slots])

  // Show ALL slots (not just playable ones) so the researcher can see what
  // exists and what still needs recording/baking. Playability is enforced
  // at the button level.
  const visible = useMemo(() => {
    if (conditionFilter === "__all__") return sb.slots
    return sb.slots.filter((s) => s.condition === conditionFilter)
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
  // Deliberately NOT returning null when slots.length === 0 anymore —
  // returning null makes the panel invisible when the toggle is on, which
  // looks identical to "the feature is broken". Show an empty-state card
  // instead so the researcher sees where to configure slots.

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
            <Button
              variant={playback.monitorMuted ? "outline" : "secondary"}
              size="xs"
              onClick={() => playback.setMonitorMuted(!playback.monitorMuted)}
              title={
                playback.monitorMuted
                  ? "Monitor is muted — you won't hear clips as they play to PP. Click to listen in."
                  : "You're hearing clips locally as they play to PP. Click to mute (does NOT affect what PP receives)."
              }
            >
              {playback.monitorMuted
                ? <VolumeX className="size-3" />
                : <Volume2 className="size-3" />}
              {playback.monitorMuted ? "monitor off" : "monitor on"}
            </Button>
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

        <div className="flex flex-col gap-1.5">
          {sb.slots.length === 0 && (
            <p className="text-[10px] text-muted-foreground italic">
              No slots yet. Open the Soundboard tab to create some.
            </p>
          )}
          {sb.slots.length > 0 && visible.length === 0 && (
            <p className="text-[10px] text-muted-foreground italic">
              No slots match this condition filter.
            </p>
          )}
          {visible.map((slot) => {
            const playing = playback.playingSlotId === slot.id
            const previewing = previewingId === slot.id
            const hasAudio = !!(slot.baked ?? slot.raw)
            // Played this session but not currently playing → greyed trail.
            // Still fully clickable so the turn can be replayed.
            const played = playedIds.has(slot.id) && !playing
            return (
              <div key={slot.id} className="flex items-center gap-1">
                <Button
                  variant={playing ? "default" : "outline"}
                  size="xs"
                  onClick={() => playing ? playback.stop() : playback.playSlot(slot)}
                  disabled={
                    !hasAudio ||
                    !ws.connected ||
                    vcEnabled ||
                    (playback.playingSlotId !== null && !playing)
                  }
                  className={`flex-1 justify-start max-w-[300px] truncate ${played ? "opacity-45" : ""}`}
                  title={
                    !hasAudio ? "Slot has no recording yet — record/bake in Soundboard tab" :
                    !ws.connected ? "Start a conversation first" :
                    vcEnabled ? "Turn off VC to enable soundboard playback" :
                    `Send to PP · ${slot.label} · ${slot.condition} · ${(((slot.bakedDurationMs || slot.rawDurationMs) / 1000)).toFixed(2)}s${played ? " · played — click to replay" : ""}`
                  }
                >
                  {playing ? <Square className="size-3" /> : <Play className="size-3" />}
                  <span className="truncate flex-1 text-left">{slot.label}</span>
                  {played && <span className="text-[9px] text-muted-foreground">played</span>}
                  <Badge variant="secondary" className="text-[9px] h-3.5">
                    {slot.condition}
                  </Badge>
                </Button>
                <Button
                  variant="ghost"
                  size="xs"
                  disabled={!hasAudio}
                  onClick={() => togglePreview(slot)}
                  title={hasAudio ? "Hear this slot locally (does NOT send to PP)" : "No audio yet"}
                >
                  {previewing
                    ? <Pause className="size-3" />
                    : <Headphones className="size-3" />}
                </Button>
              </div>
            )
          })}
        </div>

        {playback.error && (
          <p className="text-[10px] text-destructive">{playback.error}</p>
        )}
      </CardContent>
    </Card>
  )
}
