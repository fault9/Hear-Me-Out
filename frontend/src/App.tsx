import { useState, useCallback } from "react"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@shared/ui/tabs"
import { ConversationView } from "@/components/ConversationView"
import { VoiceConversion } from "@/components/VoiceConversion"
import { MetricsComparison } from "@/components/MetricsComparison"
import { ConfigureSoundboard } from "@/components/ConfigureSoundboard"
import { useRecorder } from "@shared/hooks/useRecorder"
import { useWebSocket } from "@shared/hooks/useWebSocket"
import { ThemeToggle } from "@/components/ThemeToggle"
import { Mic, GitCompare, Wand2, ListMusic } from "lucide-react"

function App() {
  const ws = useWebSocket()
  const { isMicMuted, sendAudio } = ws
  // Live-mic bytes are gated by ws.isMicMuted() so the soundboard can suppress
  // them mid-clip. Without this the mic stream and the soundboard-clip stream
  // are both sent as Opus over the same 0x01 channel, doubling PP's input rate
  // and confusing its turn-taking logic (short "okay" barge-ins). useCallback
  // keeps its identity stable so the recorder-start effect doesn't re-fire.
  const handleMicData = useCallback((data: ArrayBuffer) => {
    if (isMicMuted()) return
    sendAudio(data)
  }, [isMicMuted, sendAudio])
  const recorder = useRecorder(handleMicData)
  const [activeTab, setActiveTab] = useState("conversation")
  // Lifted here (not inside ConversationView) so it survives tab switches.
  // Radix Tabs unmounts inactive TabsContent by default, so any state inside
  // ConversationView is destroyed when you leave the Chat tab and re-created
  // fresh when you come back. State that must persist across tabs lives here.
  const [soundboardEnabled, setSoundboardEnabled] = useState(false)


  return (
    <div className="mx-auto flex max-w-6xl flex-col px-4 py-4 sm:px-8 sm:py-6 h-screen overflow-hidden">
      <header className="mb-4 flex items-center gap-4 sm:mb-5">
        {/* Public assets resolve against Vite's configured deployment base. */}
        <img src={`${import.meta.env.BASE_URL}KTH_Logo.jpg`} alt="KTH Logo" className="h-16 sm:h-20 flex-shrink-0" />
        <div className="flex flex-col gap-0.5 min-w-0 flex-1">
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-foreground">Hear Me Out</h1>
          <p className="text-xs sm:text-sm text-foreground/80 leading-snug">
            Interactive evaluation and bias discovery platform for speech-to-speech conversational AI
          </p>
          <p className="text-[11px] sm:text-xs text-foreground/60">
            KTH Royal Institute of Technology, Stockholm, Sweden
          </p>
        </div>
        <ThemeToggle />
      </header>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex flex-1 flex-col min-h-0">
        <TabsList className="mb-4 w-full !flex !flex-row">
          <TabsTrigger
            value="conversation"
            className={`flex-1 gap-1.5 rounded-md ${activeTab === "conversation" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground"}`}
          >
            <Mic />Chat
          </TabsTrigger>
          <TabsTrigger
            value="voice-conversion"
            className={`flex-1 gap-1.5 rounded-md ${activeTab === "voice-conversion" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground"}`}
          >
            <Wand2 />Convert
          </TabsTrigger>
          <TabsTrigger
            value="metrics"
            className={`flex-1 gap-1.5 rounded-md ${activeTab === "metrics" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground"}`}
          >
            <GitCompare />Metrics
          </TabsTrigger>
          <TabsTrigger
            value="soundboard"
            className={`flex-1 gap-1.5 rounded-md ${activeTab === "soundboard" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground"}`}
          >
            <ListMusic />Soundboard
          </TabsTrigger>
        </TabsList>

        <TabsContent value="conversation" className="flex-1 min-h-0">
          <ConversationView
            ws={ws}
            recorder={recorder}
            soundboardEnabled={soundboardEnabled}
            onSoundboardEnabledChange={setSoundboardEnabled}
          />
        </TabsContent>
        <TabsContent value="voice-conversion" className="flex-1 min-h-0 overflow-y-auto">
          <div className="mx-auto max-w-lg pb-6">
            <VoiceConversion />
          </div>
        </TabsContent>
        <TabsContent value="metrics" className="flex-1 min-h-0 overflow-y-auto">
          <div className="mx-auto max-w-lg pb-6">
            <MetricsComparison />
          </div>
        </TabsContent>
        <TabsContent value="soundboard" className="flex-1 min-h-0 overflow-y-auto">
          <div className="mx-auto max-w-4xl pb-6">
            <ConfigureSoundboard />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default App
