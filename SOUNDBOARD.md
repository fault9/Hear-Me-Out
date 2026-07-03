# Soundboard

Pre-record short scripted utterances, bake each one offline into one of several
voice/pitch conditions, then trigger them during a live PersonaPlex (PP)
conversation by clicking a button. The point is reproducible stimulus delivery:
same WAV in → same bytes to PP, run after run.

Two surfaces:

- **Configure Soundboard tab** — slow setup. Slots, targets, record, bake,
  preview, export/import. Use this when you're not in a conversation.
- **Soundboard panel** inside the Chat tab — fast runtime. Just buttons,
  filterable by condition, with a per-session timing log download.

## Setup order

1. **Install backend deps**. The pitch/formant bake uses `pyworld`; loudness
   normalization uses `pyloudnorm`. Both are listed in
   `services/app_api/pyproject.toml`. Re-sync (and **restart app_api** so new
   endpoints load):
   ```bash
   cd services/app_api && uv sync
   ```
   First sync may need `gcc`/`build-essential` (pyworld has a small Cython
   extension). `pyloudnorm` is pure numpy/scipy (no build step).

2. **Rebuild the frontend**:
   ```bash
   cd frontend && npm install && npm run build
   ```

3. **Verify config matches PP**. Open
   `frontend/src/lib/soundboardConfig.ts` and check:
   - `PP_SAMPLE_RATE` — must equal what PP expects on its WebSocket. Default
     `24000`, matching the live-mic opus-recorder config in
     `frontend/src/hooks/useRecorder.ts`. **If you change it here, change it
     there too.**
   - `PP_CHANNELS = 1` and `PP_BIT_DEPTH = 16`. WAV writer only does mono
     16-bit; raise both together if PP ever needs more.
   - `OPUS_ENCODER_CONFIG` — the opus-recorder options used for both live mic
     and soundboard playback. **Must match useRecorder.ts exactly**; if not,
     bakes produce a different Opus framing than live input and the
     reproducibility claim breaks.

4. **(First use)** Open the **Soundboard** tab. The default target voice
   (NATF2) is seeded automatically. Upload one or more additional target WAVs
   if you plan to use VC mode bakes — built-in target VC is not wired yet.

## Bake modes

- **Unconverted** — slot plays the raw recording as-is. Use this as the
  control condition (your unaltered voice into PP).
- **VC** — slot is converted offline to a chosen target voice via
  `/api/voice-conversion` (seed-vc). Target must be an uploaded WAV; built-in
  target VC not yet supported.
- **Pitch + Formant** — slot is run through WORLD-vocoder pitch and formant
  shift via `/api/pitch-formant`. Pitch in semitones (±12), formant as a
  multiplicative ratio (0.5–2.0). Independent axes — useful when the
  experimental contrast is gender cues.

### Loudness normalization (bake defaults)

The Configure tab's **Bake defaults** card applies EBU R128 loudness
normalization to the final clip of *every* bake (VC, pitch/formant, and
unconverted), so conditions don't differ in playback level — a live confound
otherwise. Default: **on, −23 LUFS** (`LOUDNESS_TARGET_LUFS`). Implementation:
`/api/loudness-normalize` (`pyloudnorm`, gain-only + an anti-clip peak guard;
never resamples). For unconverted mode with normalization on, the level-matched
audio is stored as `baked` while `raw` stays the untouched take. Each baked
slot shows a `… LUFS` badge; `slot.measuredLufs` records the achieved loudness.
Re-bake existing slots to apply a changed target. If you'd rather normalize
outside the app, turn it off and run ffmpeg `loudnorm` pre-upload.

## Format integrity — how to verify end-to-end

The whole point of the soundboard is that what goes to PP is deterministic
and inspectable. To verify:

1. Bake any slot in any mode. The Configure UI shows a "drift X.X ms" badge
   per slot:
   - ✓ green if `|baked − raw| ≤ DURATION_DRIFT_TOLERANCE_MS` (default 5 ms)
   - ⚠ red otherwise — means something silently re-timed the audio and the
     experiment is invalid until you fix it
2. Re-bake the same slot. The output WAV should be byte-identical to the
   first bake (WORLD synthesis is deterministic; seed-vc is deterministic
   given the same diffusion seed).
3. Export the soundboard (the **Export** button). The resulting `.zip`
   contains every slot's raw + baked WAV plus a JSON manifest. Hand it to a
   collaborator → `Import` → they should get bit-identical slots.
4. During a conversation: each click on a slot button appends a row to the
   per-session timing log (`Download log`). Check `playEndMs − playStartMs`
   vs `clipDurationMs` — should be within a few tens of ms. Larger drift
   means PP barged in (started responding before the clip finished) or the
   Opus encoder fell behind.

## Running a counted session (protocol discipline)

The runtime panel is built for reproducible, independent sessions:

- **Session bootstrap** — the **Start counted session** button (in the control
  panel when the soundboard is enabled) does the four session-independence
  steps in one click: resets PP context (fresh socket), applies the persona
  prompt, forces **VC OFF**, and mints the next session number. Sessions are
  numbered by a persistent counter (`S<n>-<suffix>`, shown as *Session #n*).
  When VC is armed instead, the button becomes **Start counted VC session** and
  keeps VC on — same reset, for live-VC conditions.
- **Turn-gap indicator** — a live *PP speaking / PP silent N ms* readout, green
  once PP has been silent ≥ the configurable threshold (`silent ≥ [ ] ms`,
  default 500). Tells the operator when it's safe to send the next turn.
- **Enforced order** (toggle, default on) — only the next un-played slot
  (top-to-bottom) is directly sendable; it gets a highlight ring. Skip-ahead is
  blocked. Turn it off for free play.
- **Flagged retry** — played slots show a ↺ button that replays the turn (e.g.
  PP stayed silent or barged in) and logs the row with `retry=1`, so
  protocol-driven repeats are separable in analysis.
- **Autoplay** (toggle, default off) — auto-advances through the slots
  top-to-bottom, firing the next one once PP has *replied* to the previous turn
  and then been silent ≥ the threshold. It waits for PP's reply before
  advancing (except the first turn), so it won't blast the whole script.
- **Monitor on/off** — hear clips locally as they play to PP (does not change
  what PP receives). **Played trail** — triggered slots grey out (still
  replayable).

## What "byte fidelity to PP" actually means

PP's WebSocket expects **Ogg-Opus** packets, not raw PCM (`useRecorder.ts`
encodes Opus at 24kHz / 80 ms frames / 1 frame per Ogg page). So the strict
"byte baked == byte to PP" claim is impossible: Opus sits between us and PP.

The practical, useful invariant the soundboard provides:

> The baked WAV is the canonical, byte-stable artifact. It is fed to the
> same opus-recorder library, with the same encoder config, as the live mic.
> Opus encoding is deterministic given fixed input + config, so the wire
> bytes PP receives are reproducible across plays and across machines.

For diff / hash / version-control purposes, work with the baked **WAV**.
The Opus stream is an implementation detail of how that WAV reaches PP.

`useWebSocket.sendAudio` (the only function that sends user audio to PP)
performs no resampling, no gain, no transcoding — it prepends one tag byte
and forwards the rest untouched. That function is the byte-fidelity contract
on the wire side; it has a comment block explaining the invariant.

## Architecture decisions worth knowing

- **Soundboard playback uses the direct-to-PP non-VC connection.** Baked
  clips are pre-converted; routing them through the live MeanVC/X-VC chat
  proxy would re-VC them and destroy reproducibility. **Open the conversation
  with VC OFF** when using the soundboard.
- **Recording uses a short-lived `getUserMedia` stream**, independent from
  the conversation mic. You can re-record a slot mid-conversation without
  disturbing the live mic (though you usually wouldn't).
- **AudioContext is created at `PP_SAMPLE_RATE`** everywhere the soundboard
  touches PCM, so no implicit browser resampling crosses the bake boundary.
  If the OS refuses that rate, playback aborts with a clear error rather
  than silently resampling.
- **Storage is IndexedDB**, not the server. Slots and targets live in your
  browser; the export/import zip is how you move them.
- **Session timing log** is structured (CSV or JSON) with one row per event —
  slot playbacks *and* PP speech-start/-end — all on the same `performance.now()`
  clock. See [Session timing log](#session-timing-log-rq2b) below for the schema
  and how to compute latency / overlap / barge-in.

## Session timing log (RQ2b)

Every conversation writes a per-session log (IndexedDB `sessions` store,
downloadable as CSV/JSON from the panel's **log** button). It's the basis for
response latency, overlap duration, and barge-in length. Two kinds of row,
distinguished by `eventType`, share one `performance.now()` clock so they're
directly comparable:

- `slot` — a soundboard clip was sent to PP.
- `pp_speech_start` / `pp_speech_end` — PersonaPlex began/ended a speech run
  (detected from its audio-packet stream; a gap > 400 ms ends a run, logged at
  the last packet's time).

CSV columns:

| Column | Meaning |
|---|---|
| `sessionId`, `conditionContext` | session id (`S<n>-<suffix>`) + operator tag |
| `eventType` | `slot` \| `pp_speech_start` \| `pp_speech_end` |
| `timestampMs` | event time on the `performance.now()` clock (all rows) |
| `slotId`, `slotLabel`, `slotCondition` | slot fields (blank for PP events) |
| `playStartMs`, `playEndMs`, `clipDurationMs` | slot playback timing (ms) |
| `retry` | `1` if a rule-triggered replay, else `0` (slot rows only) |
| `timestamp` | wall-clock unix ms (for chronological sort across kinds) |

Computing the metrics from one session's rows:

- **Response latency** = `pp_speech_start.timestampMs − slot.playEndMs` for the
  slot that preceded it.
- **Overlap / barge-in** = intersect a slot's `[playStartMs, playEndMs]` with PP
  intervals `[pp_speech_start, pp_speech_end]`; a PP start before `playEndMs` is
  a barge-in, and its length is the overlap.
- Filter `retry=1` rows out (or analyze separately) to keep planned turns clean.

Legacy rows written before this schema have `eventType`/`timestampMs` blank; the
exporter treats them as `slot` with `timestampMs = playStartMs`.

## Files

| File | What it does |
|------|--------------|
| `services/app_api/app.py` (`/api/pitch-formant`, `/api/loudness-normalize`) | Server-side pitch+formant bake + EBU R128 normalize |
| `services/app_api/pitch_formant.py` | WORLD-vocoder shift; duration-preserving; WAV helpers reused by normalize |
| `frontend/src/lib/soundboardConfig.ts` | One source of truth for SR/channels/bit-depth/loudness/defaults |
| `frontend/src/lib/audioFormat.ts` | Decode / resample / encode WAV at PP rate, duration checks |
| `frontend/src/lib/soundboardDb.ts` | IndexedDB store for slots, targets, sessions (+ event/retry schema) |
| `frontend/src/lib/soundboardZip.ts` | Pure-JS store-mode zip writer + reader |
| `frontend/src/lib/soundboardCapture.ts` | Conversation-scoped capture of sent clips → transcript + You/All WAVs + VC-quality |
| `frontend/src/lib/sessionCounter.ts` | Persistent counted-session number |
| `frontend/src/hooks/useSoundboard.ts` | Orchestration: record, bake (incl. normalize), log, export |
| `frontend/src/hooks/useSoundboardPlayback.ts` | Drives opus-recorder over baked WAVs; local monitor |
| `frontend/src/hooks/useWebSocket.ts` | PP speech-run detection + `registerPpSpeechListener` |
| `frontend/src/components/ConfigureSoundboard.tsx` | Slow-setup tab UI (bake defaults, reorder) |
| `frontend/src/components/conversation/SoundboardPanel.tsx` | Runtime panel (turn-gap, autoplay, retry, log) |

## Importing local audio + per-slot downloads

- **Upload** button per slot — pull in a pre-recorded WAV (or any
  browser-decodable audio file) instead of recording with the mic. The file
  is conformed to PP's sample rate + downmixed to mono via the same path
  target uploads use, so format integrity holds.
- **Download** icons next to the raw and baked playback buttons — grabs the
  individual WAV. Use bulk **Export** (top-right) for moving a whole
  stimulus set; per-slot download is for spot-checks outside the app.

## GPU / CPU costs (relative to PP)

PP is the heavy resident on the GPU (~19.6 GB on an RTX 3090). What the
soundboard adds beyond that:

| Operation | GPU | CPU | Conflicts with PP? |
|---|---|---|---|
| Recording into a slot | none | tiny | no |
| Upload local file | none | tiny | no |
| Bake: **Unconverted** | none | none | no |
| Bake: **Pitch + Formant** | **none** | modest (~1 s per 3 s clip via pyworld) | **no** |
| Bake: **VC** | **yes** (seed-vc subprocess) | modest | **yes — same GPU as PP** |
| Soundboard playback during a conversation | none | tiny (Opus in a WebWorker) | no |

Practical guidance:

- **Runtime playback is essentially free.** Click slot buttons as much as
  you want during a conversation; the Opus encoder runs in a worker thread
  on the browser side, and the server just relays bytes to PP.
- **Bake VC slots upfront** when PP is *not* loaded — VC shares the GPU with
  PP and may OOM otherwise. Once baked, playback never touches seed-vc
  again. Pitch+Formant bakes are CPU-only and safe to run with PP loaded.
- **Recording, upload, IndexedDB storage** are all browser-side. No server
  cost at all.

## Post-conversation analysis

When soundboard clips drove a conversation, the results bar rebuilds the user
side from what was actually sent (not the muted mic): the **You** / **You raw**
downloads are the sent/pre-bake clips placed on the conversation timeline, and
**All** is that merged with PP's audio — a synced recording of the interaction.
The final transcript includes the soundboard turns (Whisper text, or a
`[soundboard: label]` placeholder). **Analyze VC quality** lights up for a clean
single-target VC session (it scores the gapless concat of the sent vs raw clips
against the target). See `frontend/src/lib/soundboardCapture.ts`.

## What's not built (yet)

- Per-slot VC-quality badge at bake time (session-level VC-quality *is* wired
  post-conversation; per-slot-at-bake is still nice-to-have)
- VCTK corpus browser for target picker (a built-in default + uploads is
  enough for now)
- Built-in default target VC bake (needs server-side target-id resolution
  on `/api/voice-conversion`; for now upload a target WAV to use VC mode)
- Full VAD auto-advance is present (**Autoplay**); a stricter timing state
  machine (hard-gating sends on the turn-gap) is intentionally left as operator
  discretion

Auto-transcribe on bake (Whisper) is **done** — every record/bake caches a
transcript used for the live + final transcript.
