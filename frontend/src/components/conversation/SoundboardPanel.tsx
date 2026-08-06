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
import { Button } from "@shared/ui/button"
import { Badge } from "@shared/ui/badge"
import { Card, CardContent } from "@shared/ui/card"
import { Spinner } from "@shared/ui/spinner"
import { Play, Square, Download, Filter, ListMusic, Headphones, Pause, Volume2, VolumeX, RotateCcw } from "lucide-react"
import { useSoundboard, makeSessionContext, type SessionContext } from "@/hooks/useSoundboard"
import { nextSessionNumber } from "@/lib/sessionCounter"
import { useSoundboardPlayback } from "@/hooks/useSoundboardPlayback"
import { SoundboardAuditSection } from "@/components/conversation/SoundboardAuditSection"
import type { useWebSocket } from "@shared/hooks/useWebSocket"
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

  // P5 — protocol discipline. PP turn-gap indicator + enforced order + flagged
  // retry. All driven off P1's PP speech events.
  const [ppSpeaking, setPpSpeaking] = useState(false)
  const ppSpeakingRef = useRef(false)
  const ppLastEndRef = useRef(0)               // performance.now() of last speech_end
  const [ppSilentMs, setPpSilentMs] = useState(0)
  const [silenceThresholdMs, setSilenceThresholdMs] = useState(500)
  const [enforceOrder, setEnforceOrder] = useState(true)
  // Autoplay (VAD auto-advance): when on, the next slot fires automatically
  // once PP has responded and then been silent ≥ silenceThresholdMs.
  const [autoplay, setAutoplay] = useState(false)
  const ppSpokeSinceLastPlayRef = useRef(false)  // PP responded since our last send?
  const autoplayBusyRef = useRef(false)          // guards against double-fire
  // Set immediately before a rule-triggered replay so onPlayEnd logs retry=true.
  const pendingRetryRef = useRef(false)

  // A new session is minted when the WS connects; lasts for the conversation.
  const [session, setSession] = useState<SessionContext>(() => makeSessionContext(""))
  useEffect(() => {
    if (ws.connected) {
      // Counted session (P4): auto-increment a persistent number so session ids
      // in the timing log are stable + ordered. Short random suffix keeps ids
      // unique across machines when logs are merged.
      const n = nextSessionNumber()
      const suffix = Math.random().toString(36).slice(2, 6)
      setSession({ sessionId: `S${n}-${suffix}`, conditionContext, sessionNumber: n })
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
      if (e.type === "pp_speech_start") {
        ppSpeakingRef.current = true
        ppSpokeSinceLastPlayRef.current = true   // PP is responding to our turn
        setPpSpeaking(true)
      } else {
        ppSpeakingRef.current = false
        ppLastEndRef.current = e.timestampMs
        setPpSpeaking(false)
      }
    })
  }, [registerPpSpeechListener])

  // Live "PP silent for N ms" ticker for the turn-gap indicator. Reads speaking
  // state + last-end time via refs so the interval isn't re-created constantly.
  useEffect(() => {
    if (!ws.connected) {
      setPpSilentMs(0)
      setPpSpeaking(false)
      ppSpeakingRef.current = false
      ppLastEndRef.current = 0
      return
    }
    const id = window.setInterval(() => {
      setPpSilentMs(
        ppSpeakingRef.current || !ppLastEndRef.current
          ? 0
          : performance.now() - ppLastEndRef.current,
      )
    }, 100)
    return () => clearInterval(id)
  }, [ws.connected])

  // Start the PP-silence clock when PP becomes ready, so autoplay can fire the
  // first slot even if PP never greets (otherwise ppSilentMs sits at 0 until
  // PP's first utterance ends).
  useEffect(() => {
    if (ws.warmupComplete) ppLastEndRef.current = performance.now()
  }, [ws.warmupComplete])

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
      // Wait for PP to respond to THIS turn before autoplay advances.
      ppSpokeSinceLastPlayRef.current = false
    },
    onPlayEnd: (slot, rec) => {
      void sb.logPlayback(slot, session, rec.startMs, rec.endMs, rec.clipDurationMs, pendingRetryRef.current)
      pendingRetryRef.current = false
      autoplayBusyRef.current = false
    },
  })

  // Replay a slot as a rule-triggered retry (PP stayed silent / barged in).
  // Marks the next log row retry=true; the slot stays greyed (already played).
  const replaySlot = (slot: Slot) => {
    pendingRetryRef.current = true
    void playback.playSlot(slot)
  }

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

  // P5 enforced order: the next slot the operator should play is the first
  // un-played, playable slot in the (ordered) visible list.
  const nextExpectedId = useMemo(
    () => visible.find((s) => !playedIds.has(s.id) && !!(s.baked ?? s.raw))?.id,
    [visible, playedIds],
  )

  // Turn-gap gate: is PP silent long enough that it's OK to send the next turn?
  const gapReady = !ppSpeaking && ppSilentMs >= silenceThresholdMs

  // Autoplay / VAD auto-advance: fire the next slot (top-to-bottom order) once
  // PP has responded to the previous turn and then gone silent ≥ threshold.
  // Re-evaluated on each 100 ms tick (ppSilentMs). autoplayBusyRef guards the
  // async gap between playSlot() and playingSlotId updating so we don't double
  // fire; ppSpokeSinceLastPlay ensures we wait for PP's reply before advancing
  // (except the very first turn).
  useEffect(() => {
    if (!autoplay || vcEnabled) return
    if (!ws.connected || playback.playingSlotId !== null || autoplayBusyRef.current) return
    if (!gapReady || !nextExpectedId) return
    if (playedIds.size > 0 && !ppSpokeSinceLastPlayRef.current) return
    const slot = visible.find((s) => s.id === nextExpectedId)
    if (!slot) return
    autoplayBusyRef.current = true
    void playback.playSlot(slot)
  }, [ppSilentMs, autoplay, vcEnabled, ws.connected, gapReady, nextExpectedId, playedIds, visible, playback])

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
              {session.sessionNumber != null && (
                <span className="mr-1 rounded bg-primary/15 px-1 font-semibold text-primary">
                  Session #{session.sessionNumber}
                </span>
              )}
              <span className="font-mono">{session.sessionId}</span>
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

        {/* P5 — turn-gap indicator + protocol controls (silence threshold,
            enforced order). Driven by P1's PP speech events. */}
        <div className="flex items-center justify-between gap-2 flex-wrap rounded-md border bg-muted/30 px-2 py-1.5">
          <div className="flex items-center gap-1.5 text-[11px]">
            {!ws.connected ? (
              <span className="text-muted-foreground">turn gate idle — start a session</span>
            ) : ppSpeaking ? (
              <span className="inline-flex items-center gap-1 font-medium text-amber-500">
                <span className="size-1.5 rounded-full bg-amber-500 animate-pulse" /> PP speaking…
              </span>
            ) : gapReady ? (
              <span className="inline-flex items-center gap-1 font-medium text-emerald-500">
                <span className="size-1.5 rounded-full bg-emerald-500" /> PP silent {Math.round(ppSilentMs)} ms — OK to send
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-muted-foreground">
                <span className="size-1.5 rounded-full bg-muted-foreground/50" /> PP silent {Math.round(ppSilentMs)} ms — wait
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
            <label className="flex items-center gap-1" title="Silence threshold: how long PP must be quiet before the turn-gap indicator turns green and (if autoplay is on) the next slot fires.">
              silent ≥
              <input
                type="number"
                min={0}
                step={50}
                value={silenceThresholdMs}
                onChange={(e) => setSilenceThresholdMs(Math.max(0, Number(e.target.value) || 0))}
                className="w-14 h-6 rounded border bg-background px-1 text-[10px]"
              />
              ms
            </label>
            <button
              onClick={() => setAutoplay((v) => !v)}
              className={`rounded border px-1.5 py-0.5 ${autoplay ? "bg-emerald-500/15 text-emerald-600 border-emerald-500/40 dark:text-emerald-300" : "text-muted-foreground"}`}
              title="Autoplay: automatically send the next slot (top-to-bottom) once PP has replied and then been silent for the threshold above."
            >
              {autoplay ? "autoplay: on" : "autoplay: off"}
            </button>
            <button
              onClick={() => setEnforceOrder((v) => !v)}
              className={`rounded border px-1.5 py-0.5 ${enforceOrder ? "bg-primary/15 text-primary border-primary/40" : "text-muted-foreground"}`}
              title="When on, only the next slot in order can be sent (skip-ahead blocked). Replays of already-played slots stay available via the retry button and are logged as retries."
            >
              {enforceOrder ? "order: enforced" : "order: free"}
            </button>
          </div>
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
            const played = playedIds.has(slot.id) && !playing
            // P5 enforced order: this is the next slot expected in sequence.
            const isNext = slot.id === nextExpectedId
            // With order enforced, only the next-expected slot's MAIN button is
            // active; skip-ahead and re-firing played slots go through retry.
            const orderBlocked = enforceOrder && !isNext && !playing
            const otherPlaying = playback.playingSlotId !== null && !playing
            const mainDisabled = !hasAudio || !ws.connected || vcEnabled || otherPlaying || orderBlocked
            return (
              <div key={slot.id} className="flex items-center gap-1">
                <Button
                  variant={playing ? "default" : "outline"}
                  size="xs"
                  onClick={() => playing ? playback.stop() : playback.playSlot(slot)}
                  disabled={mainDisabled}
                  className={`flex-1 justify-start max-w-[300px] truncate ${played ? "opacity-45" : ""} ${isNext && ws.connected && !vcEnabled && !playing ? "ring-2 ring-emerald-500/60" : ""}`}
                  title={
                    !hasAudio ? "Slot has no recording yet — record/bake in Soundboard tab" :
                    !ws.connected ? "Start a conversation first" :
                    vcEnabled ? "Turn off VC to enable soundboard playback" :
                    orderBlocked ? "Order enforced — play the highlighted next slot (or use retry to replay a played one)" :
                    `Send to PP · ${slot.label} · ${slot.condition} · ${(((slot.bakedDurationMs || slot.rawDurationMs) / 1000)).toFixed(2)}s${isNext ? " · next" : ""}`
                  }
                >
                  {playing ? <Square className="size-3" /> : <Play className="size-3" />}
                  <span className="truncate flex-1 text-left">{slot.label}</span>
                  {played && <span className="text-[9px] text-muted-foreground">played</span>}
                  <Badge variant="secondary" className="text-[9px] h-3.5">
                    {slot.condition}
                  </Badge>
                </Button>
                {/* Rule-triggered retry: replay a played slot (PP silent / barge-in)
                    and tag the log row retry=true. Available once a slot has been
                    played, even under enforced order. */}
                {played && (
                  <Button
                    variant="ghost"
                    size="xs"
                    disabled={!hasAudio || !ws.connected || vcEnabled || otherPlaying}
                    onClick={() => replaySlot(slot)}
                    title="Replay this turn as a rule-triggered retry (logged as retry=1)"
                  >
                    <RotateCcw className="size-3" />
                  </Button>
                )}
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

        {/* Automated audit runs: only in soundboard mode (this panel), only
            with VC off (panel is disabled under VC anyway). Turns autoplay
            off before starting so the P5 auto-advance can't double-fire
            slots into the runner's conversations. */}
        {!vcEnabled && (
          <SoundboardAuditSection
            ws={ws}
            playback={playback}
            slots={visible}
            targets={sb.targets}
            onBeforeRun={() => setAutoplay(false)}
          />
        )}
      </CardContent>
    </Card>
  )
}
