import { useEffect, useMemo, useState } from "react"
import { X, AlertTriangle, ChevronRight, ChevronDown, User, Bot } from "lucide-react"
import type { VcQualityResult, VcQualitySegment, VcQualityAnomaly } from "@/services/api"
import type { DiarizedTurn } from "@/hooks/useConversation"

interface Props {
  data: VcQualityResult
  // Diarized turns from the parent conversation (user + PersonaPlex with
  // timestamps). When provided, the selected-segment detail card includes a
  // conversation snippet of every turn overlapping the chosen window.
  diarized?: DiarizedTurn[] | null
  onClose: () => void
}

const fmt = (v: number | null | undefined, digits = 2) =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits)

// Color thresholds match the per-segment anomaly floors in vc_quality.py.
function colorFor(metric: string, v: number | null | undefined): string {
  if (v == null) return "text-muted-foreground"
  if (metric === "wer" || metric === "cer") {
    if (v < 0.15) return "text-emerald-400"
    if (v < 0.35) return "text-amber-400"
    return "text-red-400"
  }
  if (metric === "secs") {
    if (v >= 0.7) return "text-emerald-400"
    if (v >= 0.5) return "text-amber-400"
    return "text-red-400"
  }
  if (metric === "utmos" || metric === "ovrl" || metric === "sig") {
    if (v >= 3.5) return "text-emerald-400"
    if (v >= 2.5) return "text-amber-400"
    return "text-red-400"
  }
  return "text-foreground"
}

function HeadlineCard({ label, value, metric, hint }:
                       { label: string; value: number | null | undefined; metric: string; hint?: string }) {
  return (
    <div className="rounded-lg border bg-muted/30 p-3">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className={`text-2xl font-semibold tabular-nums ${colorFor(metric, value)}`}>
        {fmt(value)}
      </div>
      {hint && <div className="mt-0.5 text-[10px] text-muted-foreground">{hint}</div>}
    </div>
  )
}

function SegmentTimeline({
  segments, anomalousIdxs, selectedIdx, onSelect,
}: {
  segments: VcQualitySegment[]
  anomalousIdxs: Set<number>
  selectedIdx: number | null
  onSelect: (i: number) => void
}) {
  if (segments.length === 0) return null
  const totalDur = segments[segments.length - 1].end
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-xs font-medium">Per-segment timeline</div>
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <span className="inline-block size-2 rounded-sm bg-emerald-500/40 border border-emerald-500/60" /> OK
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="inline-block size-2 rounded-sm bg-red-500/50 border border-red-400" /> Anomaly
          </span>
        </div>
      </div>
      <div className="relative h-9 w-full overflow-hidden rounded-md border bg-muted/20">
        {segments.map((seg, i) => {
          const left = (seg.start / totalDur) * 100
          const width = ((seg.end - seg.start) / totalDur) * 100
          const flagged = anomalousIdxs.has(i)
          const selected = selectedIdx === i
          let cls = flagged
            ? "bg-red-500/40 border-red-400 hover:bg-red-500/60"
            : "bg-emerald-500/15 border-emerald-500/30 hover:bg-emerald-500/35"
          if (selected) {
            cls = flagged
              ? "bg-red-500/80 border-red-300 ring-1 ring-red-200"
              : "bg-emerald-500/60 border-emerald-300 ring-1 ring-emerald-200"
          }
          return (
            <button
              key={i}
              type="button"
              onClick={() => onSelect(i)}
              className={`absolute top-0 bottom-0 border-l transition-colors cursor-pointer ${cls}`}
              style={{ left: `${left}%`, width: `${Math.max(width, 0.5)}%` }}
              title={`${seg.start.toFixed(2)}–${seg.end.toFixed(2)}s (click for details)`}
              aria-label={`Segment ${seg.start.toFixed(2)} to ${seg.end.toFixed(2)} seconds${flagged ? ", flagged" : ""}`}
            />
          )
        })}
      </div>
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>0s</span>
        <span>{totalDur.toFixed(1)}s</span>
      </div>
    </div>
  )
}

// Detail card for the currently selected segment.
// Turns whose time range overlaps [start, end). Two ranges overlap when
// turn.start < end && turn.end > start.
function turnsInWindow(turns: DiarizedTurn[], start: number, end: number) {
  return turns
    .filter(t => t.start < end && t.end > start)
    .sort((a, b) => a.start - b.start)
}

function fmtTimeRange(start: number, end: number) {
  const f = (s: number) => {
    const m = Math.floor(s / 60), sec = (s % 60).toFixed(1).padStart(4, "0")
    return `${m}:${sec}`
  }
  return `${f(start)}–${f(end)}`
}

function ConversationSnippet({
  turns, segStart, segEnd,
}: { turns: DiarizedTurn[]; segStart: number; segEnd: number }) {
  if (turns.length === 0) {
    return (
      <div className="rounded border border-dashed bg-background/30 p-2 text-[11px] text-muted-foreground italic">
        No transcript turns overlap this window.
      </div>
    )
  }
  return (
    <div className="space-y-1.5">
      {turns.map((t, i) => {
        const isUser = t.speaker === "user"
        const Icon = isUser ? User : Bot
        // Highlight the portion of each turn that falls inside the window.
        const inside = t.start < segEnd && t.end > segStart
        return (
          <div
            key={i}
            className={`rounded border p-1.5 text-[11px] ${inside ? "bg-background/60" : "bg-background/20 opacity-70"}`}
          >
            <div className="flex items-center justify-between gap-2 text-[10px] text-muted-foreground">
              <span className="inline-flex items-center gap-1">
                <Icon className={`size-3 ${isUser ? "text-blue-400" : "text-purple-400"}`} />
                <span className="font-medium">{isUser ? "You" : "PersonaPlex"}</span>
              </span>
              <span className="tabular-nums">{fmtTimeRange(t.start, t.end)}</span>
            </div>
            <div className="mt-0.5 text-foreground">{t.text || <span className="italic text-muted-foreground">(silent)</span>}</div>
          </div>
        )
      })}
    </div>
  )
}

function SegmentDetail({
  segment, anomalies, diarized, onClear,
}: {
  segment: VcQualitySegment
  anomalies: VcQualityAnomaly[]
  diarized?: DiarizedTurn[] | null
  onClear: () => void
}) {
  const offendingMetrics = new Set(anomalies.map(a => a.metric))
  const rows: Array<[string, number | null | undefined, string]> = [
    ["WER",        segment.wer,      "wer"],
    ["SECS",       segment.secs,     "secs"],
    ["UTMOS",      segment.utmos,    "utmos"],
    ["DNSMOS OVRL", segment.ovrl,    "ovrl"],
    ["DNSMOS SIG", segment.sig,      "sig"],
    ["F0 PCC",     segment.f0_pcc,   "f0_pcc"],
    ["F0 RMSE",    segment.f0_rmse,  "f0_rmse"],
  ]
  return (
    <div className="rounded-lg border bg-muted/40 p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold tabular-nums">
          Segment {segment.start.toFixed(2)}s – {segment.end.toFixed(2)}s
          <span className="ml-2 text-[10px] font-normal text-muted-foreground">
            ({(segment.end - segment.start).toFixed(2)}s window)
          </span>
        </div>
        <button
          onClick={onClear}
          className="text-[10px] text-muted-foreground hover:text-foreground"
        >
          clear selection
        </button>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
        {rows.map(([label, value, metric]) => {
          const flagged = offendingMetrics.has(metric)
          return (
            <div
              key={metric}
              className={`rounded border p-1.5 text-[11px] ${flagged ? "border-red-500/50 bg-red-500/10" : "bg-background/40"}`}
            >
              <div className="text-[10px] text-muted-foreground">{label}</div>
              <div className={`font-medium tabular-nums ${colorFor(metric, value)}`}>
                {fmt(value, metric === "f0_rmse" ? 1 : 2)}
                {metric === "f0_rmse" && value != null && <span className="ml-1 text-[9px] text-muted-foreground">Hz</span>}
                {flagged && <AlertTriangle className="ml-1 inline size-3 text-red-400" />}
              </div>
            </div>
          )
        })}
      </div>
      {diarized && diarized.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
            Conversation in this window
          </div>
          <ConversationSnippet
            turns={turnsInWindow(diarized, segment.start, segment.end)}
            segStart={segment.start}
            segEnd={segment.end}
          />
        </div>
      )}
      {segment.ref && (
        <div className="rounded border bg-background/30 p-1.5 text-[10px] text-muted-foreground">
          <span className="font-medium">WER reference: </span>{segment.ref}
        </div>
      )}
    </div>
  )
}

// Anomalies grouped by segment, with each group collapsible.
function AnomalyList({
  groups, expanded, onToggle, onSelectSegment, selectedSegmentIdx,
}: {
  groups: Array<{ segmentIdx: number; start: number; end: number; entries: VcQualityAnomaly[] }>
  expanded: Set<number>
  onToggle: (segmentIdx: number) => void
  onSelectSegment: (i: number) => void
  selectedSegmentIdx: number | null
}) {
  if (groups.length === 0) {
    return (
      <div className="rounded-lg border bg-emerald-500/5 border-emerald-500/20 p-3 text-xs text-emerald-300">
        No anomalous segments flagged.
      </div>
    )
  }
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5 text-xs font-medium text-amber-300">
        <AlertTriangle className="size-3.5" />
        Anomalous segments ({groups.length})
      </div>
      <div className="rounded-lg border overflow-hidden">
        {groups.map(g => {
          const open = expanded.has(g.segmentIdx)
          const isSelected = selectedSegmentIdx === g.segmentIdx
          return (
            <div key={g.segmentIdx} className={`border-b last:border-b-0 ${isSelected ? "bg-red-500/10" : ""}`}>
              <button
                type="button"
                onClick={() => { onToggle(g.segmentIdx); onSelectSegment(g.segmentIdx) }}
                className="flex w-full items-center justify-between gap-2 px-2.5 py-1.5 text-left text-[11px] hover:bg-muted/40"
                aria-expanded={open}
              >
                <span className="inline-flex items-center gap-1.5 tabular-nums">
                  {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
                  <span className="font-medium">{g.start.toFixed(2)}–{g.end.toFixed(2)}s</span>
                  <span className="text-muted-foreground">·</span>
                  <span className="text-muted-foreground">
                    {g.entries.length} {g.entries.length === 1 ? "metric" : "metrics"} flagged
                  </span>
                </span>
                <span className="flex gap-1">
                  {g.entries.map((a, i) => (
                    <span key={i} className="rounded bg-red-500/20 px-1.5 py-0.5 text-[9px] font-mono text-red-200">
                      {a.metric}
                    </span>
                  ))}
                </span>
              </button>
              {open && (
                <div className="border-t bg-background/40 px-2.5 py-1.5">
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="text-muted-foreground">
                        <th className="px-1 py-0.5 text-left font-medium">Metric</th>
                        <th className="px-1 py-0.5 text-right font-medium">Score</th>
                        <th className="px-1 py-0.5 text-right font-medium">z-score</th>
                        <th className="px-1 py-0.5 text-left font-medium">Why</th>
                      </tr>
                    </thead>
                    <tbody>
                      {g.entries.map((a, i) => (
                        <tr key={i} className="border-t border-muted/40">
                          <td className="px-1 py-0.5 font-mono">{a.metric}</td>
                          <td className="px-1 py-0.5 text-right tabular-nums">{a.score.toFixed(3)}</td>
                          <td className="px-1 py-0.5 text-right tabular-nums">{a.z.toFixed(2)}</td>
                          <td className="px-1 py-0.5 text-muted-foreground">
                            {Math.abs(a.z) >= 2 ? "outlier vs clip" : "below floor"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function VcQualityModal({ data, diarized, onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose() }
    document.addEventListener("keydown", onKey)
    return () => document.removeEventListener("keydown", onKey)
  }, [onClose])

  const segments = data.segments || []
  const anomalies = data.anomalies || []

  // Cross-link: which segments (by index) have at least one flagged metric.
  const { anomalousIdxs, groups } = useMemo(() => {
    const idxByRange = new Map<string, number>()
    segments.forEach((s, i) => idxByRange.set(`${s.start}-${s.end}`, i))
    const grouped = new Map<number, { start: number; end: number; entries: VcQualityAnomaly[] }>()
    for (const a of anomalies) {
      const idx = idxByRange.get(`${a.start}-${a.end}`)
      if (idx == null) continue
      if (!grouped.has(idx)) grouped.set(idx, { start: a.start, end: a.end, entries: [] })
      grouped.get(idx)!.entries.push(a)
    }
    const groupArr = Array.from(grouped.entries())
      .map(([segmentIdx, g]) => ({ segmentIdx, ...g }))
      .sort((a, b) => a.start - b.start)
    return {
      anomalousIdxs: new Set(grouped.keys()),
      groups: groupArr,
    }
  }, [segments, anomalies])

  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const [expanded, setExpanded] = useState<Set<number>>(
    () => new Set(groups.length === 1 ? [groups[0].segmentIdx] : [])
  )

  const toggleExpanded = (segmentIdx: number) => {
    setExpanded(prev => {
      const next = new Set(prev)
      if (next.has(segmentIdx)) next.delete(segmentIdx)
      else next.add(segmentIdx)
      return next
    })
  }

  const selectedSegment = selectedIdx != null ? segments[selectedIdx] : null
  const selectedSegmentAnomalies = selectedIdx != null
    ? anomalies.filter(a => {
        const s = segments[selectedIdx]
        return a.start === s.start && a.end === s.end
      })
    : []

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="relative flex max-h-[88vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border bg-card shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b px-5 py-3">
          <h3 className="text-sm font-semibold">VC Quality Analysis</h3>
          <button
            onClick={onClose}
            className="inline-flex size-7 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
            aria-label="Close"
          >
            <X className="size-4" />
          </button>
        </div>
        <p className="px-5 pt-3 text-xs text-muted-foreground">
          Post-hoc evaluation of the voice-conversion output: intelligibility (WER),
          target speaker similarity (SECS), naturalness (UTMOS / DNSMOS) and
          prosody fidelity (F0). Lower WER and higher SECS / UTMOS are better.
          Click a timeline bar or anomaly row to inspect a window.
        </p>
        <div className="overflow-y-auto p-5 pt-3 space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <HeadlineCard label="WER" value={data.wer} metric="wer" hint="word err rate" />
            <HeadlineCard label="SECS" value={data.secs} metric="secs" hint="vs target voice" />
            <HeadlineCard label="UTMOS" value={data.utmos} metric="utmos" hint="naturalness" />
            <HeadlineCard label="DNSMOS OVRL" value={data.ovrl} metric="ovrl" hint="overall MOS" />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div className="rounded border bg-muted/20 p-2">
              <span className="text-muted-foreground">CER </span>
              <span className="tabular-nums">{fmt(data.cer)}</span>
            </div>
            <div className="rounded border bg-muted/20 p-2">
              <span className="text-muted-foreground">SIG </span>
              <span className="tabular-nums">{fmt(data.sig)}</span>
            </div>
            <div className="rounded border bg-muted/20 p-2">
              <span className="text-muted-foreground">F0 PCC </span>
              <span className="tabular-nums">{fmt(data.f0_pcc)}</span>
            </div>
            <div className="rounded border bg-muted/20 p-2">
              <span className="text-muted-foreground">F0 RMSE </span>
              <span className="tabular-nums">{fmt(data.f0_rmse, 1)} Hz</span>
            </div>
          </div>

          <SegmentTimeline
            segments={segments}
            anomalousIdxs={anomalousIdxs}
            selectedIdx={selectedIdx}
            onSelect={setSelectedIdx}
          />

          {selectedSegment ? (
            <SegmentDetail
              segment={selectedSegment}
              anomalies={selectedSegmentAnomalies}
              diarized={diarized}
              onClear={() => setSelectedIdx(null)}
            />
          ) : (
            <div className="rounded-lg border border-dashed bg-muted/10 p-3 text-center text-[11px] text-muted-foreground">
              Click any timeline bar above to see per-segment metrics.
            </div>
          )}

          <AnomalyList
            groups={groups}
            expanded={expanded}
            onToggle={toggleExpanded}
            onSelectSegment={setSelectedIdx}
            selectedSegmentIdx={selectedIdx}
          />

          {data.vc_transcript && (
            <div className="space-y-1">
              <div className="text-xs font-medium">Whole-clip transcripts</div>
              <div className="rounded-lg border bg-muted/20 p-2 text-[11px] space-y-1">
                <div><span className="text-muted-foreground">VC: </span>{data.vc_transcript}</div>
                {data.ref_transcript && (
                  <div><span className="text-muted-foreground">Ref ({data.ref_kind}): </span>{data.ref_transcript}</div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
