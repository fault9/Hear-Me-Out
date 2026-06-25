import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { Download, Play, Pause, BarChart3, FileJson, Activity, ChevronDown } from "lucide-react"
import { formatTime } from "@/lib/utils"
import type { VcQualityMetric } from "@/services/api"

// Metric checkboxes in the "Analyze VC quality" dropdown. SECS is the
// heaviest (WavLM-large forward pass per segment); skipping it is the single
// biggest CPU-time win on long conversations, so we make the speed cost
// visible in the label.
const METRIC_CHOICES: { key: VcQualityMetric; label: string; hint: string }[] = [
  { key: "intelligibility", label: "Intelligibility (WER, CER)",            hint: "Whisper" },
  { key: "secs",            label: "Speaker similarity (SECS)",             hint: "WavLM-large — slow" },
  { key: "naturalness",     label: "Naturalness (UTMOS, DNSMOS)",           hint: "two MOS predictors" },
  { key: "prosody",         label: "Prosody (F0 PCC, F0 RMSE)",             hint: "librosa pyin" },
]

interface Props {
  userWavUrl: string | null
  personaplexWavUrl: string | null
  mergedWavUrl: string | null
  originalUserWavUrl?: string | null
  onDownloadTranscript: () => void
  onPlayTimeChange: (t: number) => void
  onPlayingChange: (p: boolean) => void
  vcMetricsLoading?: boolean
  vcMetricsReady?: boolean
  canTriggerVcMetrics?: boolean
  onTriggerVcMetrics?: () => void
  onShowVcMetrics?: () => void
  vcQualityLoading?: boolean
  vcQualityReady?: boolean
  canTriggerVcQuality?: boolean
  onTriggerVcQuality?: (skipMetrics: VcQualityMetric[]) => void
  onShowVcQuality?: () => void
  onDownloadVcQuality?: () => void
}

export function DownloadBar({
  userWavUrl, personaplexWavUrl, mergedWavUrl, originalUserWavUrl,
  onDownloadTranscript, onPlayTimeChange, onPlayingChange,
  vcMetricsLoading, vcMetricsReady, canTriggerVcMetrics, onTriggerVcMetrics, onShowVcMetrics,
  vcQualityLoading, vcQualityReady, canTriggerVcQuality, onTriggerVcQuality, onShowVcQuality, onDownloadVcQuality,
}: Props) {
  const [playing, setPlaying] = useState(false)
  const [playTime, setPlayTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  // Dropdown state for the "Analyze VC quality" button.
  // Default: SECS is OFF — it's the WavLM forward-pass-per-segment hog that
  // was timing out on long conversations. Opt back in when you want it.
  const [vcQualityMenuOpen, setVcQualityMenuOpen] = useState(false)
  const [enabledMetrics, setEnabledMetrics] = useState<Record<VcQualityMetric, boolean>>({
    intelligibility: true,
    secs: false,
    naturalness: true,
    prosody: true,
  })
  const vcQualityMenuRef = useRef<HTMLDivElement | null>(null)

  // Close the dropdown when clicking outside.
  useEffect(() => {
    if (!vcQualityMenuOpen) return
    const onDown = (e: MouseEvent) => {
      if (vcQualityMenuRef.current && !vcQualityMenuRef.current.contains(e.target as Node)) {
        setVcQualityMenuOpen(false)
      }
    }
    document.addEventListener("mousedown", onDown)
    return () => document.removeEventListener("mousedown", onDown)
  }, [vcQualityMenuOpen])

  const handleRunVcQuality = () => {
    const skip = (Object.entries(enabledMetrics) as [VcQualityMetric, boolean][])
      .filter(([, on]) => !on)
      .map(([k]) => k)
    setVcQualityMenuOpen(false)
    onTriggerVcQuality?.(skip)
  }

  const updatePlaying = (p: boolean) => { setPlaying(p); onPlayingChange(p) }
  const updatePlayTime = (t: number) => { setPlayTime(t); onPlayTimeChange(t) }

  const togglePlay = () => {
    const src = mergedWavUrl || userWavUrl
    if (!audioRef.current) {
      const a = new Audio(src!)
      a.ontimeupdate = () => updatePlayTime(a.currentTime)
      a.onloadedmetadata = () => setDuration(a.duration)
      a.onended = () => { updatePlaying(false); updatePlayTime(0) }
      a.onplay = () => updatePlaying(true)
      a.onpause = () => updatePlaying(false)
      audioRef.current = a
      a.play()
    } else if (playing) {
      audioRef.current.pause()
    } else {
      if (audioRef.current.currentTime >= (audioRef.current.duration || 0) - 0.5) {
        audioRef.current.currentTime = 0
      }
      audioRef.current.play()
    }
  }

  const seek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current) return
    const rect = e.currentTarget.getBoundingClientRect()
    const pct = (e.clientX - rect.left) / rect.width
    audioRef.current.currentTime = pct * (audioRef.current.duration || 0)
    updatePlayTime(audioRef.current.currentTime)
  }

  return (
    <div className="md:col-span-2 rounded-lg border bg-muted/50 px-4 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium">Conversation complete</span>
        <div className="flex flex-wrap items-center gap-2">
          {vcMetricsLoading && (
            <span className="inline-flex items-center gap-1.5 text-[10px] text-muted-foreground">
              <Spinner className="size-3" /> Analyzing voice…
            </span>
          )}
          {!vcMetricsLoading && !vcMetricsReady && canTriggerVcMetrics && onTriggerVcMetrics && (
            <Button variant="outline" size="xs" onClick={onTriggerVcMetrics} className="border-purple-500/50 text-purple-300 hover:bg-purple-500/10">
              <BarChart3 /> Analyze voice change
            </Button>
          )}
          {vcMetricsReady && !vcMetricsLoading && (
            <Button variant="outline" size="xs" onClick={onShowVcMetrics} className="border-purple-500/50 text-purple-300 hover:bg-purple-500/10">
              <BarChart3 /> Voice change metrics
            </Button>
          )}
          {vcQualityLoading && (
            <span className="inline-flex items-center gap-1.5 text-[10px] text-muted-foreground">
              <Spinner className="size-3" /> Scoring VC quality…
            </span>
          )}
          {!vcQualityLoading && !vcQualityReady && canTriggerVcQuality && onTriggerVcQuality && (
            <div ref={vcQualityMenuRef} className="relative">
              <Button
                variant="outline"
                size="xs"
                onClick={() => setVcQualityMenuOpen((v) => !v)}
                aria-haspopup="true"
                aria-expanded={vcQualityMenuOpen}
                className="border-emerald-500/50 text-emerald-300 hover:bg-emerald-500/10"
              >
                <Activity /> Analyze VC quality <ChevronDown className="ml-0.5 size-3" />
              </Button>
              {vcQualityMenuOpen && (
                <div
                  role="menu"
                  className="absolute right-0 z-30 mt-1 w-72 rounded-lg border bg-card shadow-xl p-3 space-y-2"
                >
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    Metrics to compute
                  </div>
                  <div className="space-y-1.5">
                    {METRIC_CHOICES.map((m) => (
                      <label
                        key={m.key}
                        className="flex items-start gap-2 rounded px-1.5 py-1 text-xs hover:bg-muted cursor-pointer"
                      >
                        <input
                          type="checkbox"
                          className="mt-0.5 size-3.5 accent-emerald-500"
                          checked={enabledMetrics[m.key]}
                          onChange={() =>
                            setEnabledMetrics((prev) => ({ ...prev, [m.key]: !prev[m.key] }))
                          }
                        />
                        <span className="flex-1 leading-tight">
                          <span className="font-medium">{m.label}</span>
                          <span className="block text-[10px] text-muted-foreground">{m.hint}</span>
                        </span>
                      </label>
                    ))}
                  </div>
                  <div className="flex items-center justify-between pt-1">
                    <button
                      className="text-[10px] text-muted-foreground hover:text-foreground"
                      onClick={() =>
                        setEnabledMetrics({
                          intelligibility: true, secs: true, naturalness: true, prosody: true,
                        })
                      }
                    >
                      Enable all
                    </button>
                    <Button
                      variant="default"
                      size="xs"
                      onClick={handleRunVcQuality}
                      className="bg-emerald-600 text-white hover:bg-emerald-500"
                    >
                      Run analysis
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}
          {vcQualityReady && !vcQualityLoading && onShowVcQuality && (
            <Button variant="outline" size="xs" onClick={onShowVcQuality} className="border-emerald-500/50 text-emerald-300 hover:bg-emerald-500/10">
              <Activity /> VC quality
            </Button>
          )}
          {vcQualityReady && !vcQualityLoading && onDownloadVcQuality && (
            <Button variant="outline" size="xs" onClick={onDownloadVcQuality} className="border-emerald-500/50 text-emerald-300 hover:bg-emerald-500/10">
              <FileJson /> VC quality (JSON)
            </Button>
          )}
          <Button variant="outline" size="xs" onClick={onDownloadTranscript}>
            <Download /> Transcript
          </Button>
          {originalUserWavUrl && (
            <a href={originalUserWavUrl} download="you-original.wav"
              className="inline-flex items-center gap-1 h-6 rounded-lg border px-2 text-[10px] font-medium hover:bg-muted">
              You (raw)
            </a>
          )}
          {userWavUrl && (
            <a href={userWavUrl} download="user-recording.wav"
              className="inline-flex items-center gap-1 h-6 rounded-lg border px-2 text-[10px] font-medium hover:bg-muted">
              You
            </a>
          )}
          {personaplexWavUrl && (
            <a href={personaplexWavUrl} download="personaplex-response.wav"
              className="inline-flex items-center gap-1 h-6 rounded-lg border px-2 text-[10px] font-medium hover:bg-muted">
              PP
            </a>
          )}
          {mergedWavUrl && (
            <a href={mergedWavUrl} download="conversation.wav"
              className="inline-flex items-center gap-1 h-6 rounded-lg bg-primary px-2 text-[10px] font-medium text-primary-foreground hover:bg-primary/90">
              All
            </a>
          )}
        </div>
      </div>

      {(mergedWavUrl || userWavUrl) && (
        <div className="mt-2 flex items-center gap-2">
          <Button size="icon" variant="ghost" className="size-7" onClick={togglePlay}>
            {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
          </Button>
          <div
            className="relative flex-1 h-1.5 rounded-full bg-muted-foreground/20 cursor-pointer"
            onClick={seek}
          >
            <div
              className="absolute inset-y-0 left-0 rounded-full bg-primary transition-[width] duration-100"
              style={{ width: `${duration > 0 ? (playTime / duration) * 100 : 0}%` }}
            />
          </div>
          <span className="text-[10px] text-muted-foreground tabular-nums min-w-[52px] text-right">
            {formatTime(playTime)} / {formatTime(duration)}
          </span>
        </div>
      )}
    </div>
  )
}
