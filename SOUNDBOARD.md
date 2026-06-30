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

1. **Install backend dep**. The pitch/formant bake uses `pyworld`. Already
   listed in `services/app_api/pyproject.toml`. Re-sync:
   ```bash
   cd services/app_api && uv sync
   ```
   First sync may need `gcc`/`build-essential` (pyworld has a small Cython
   extension).

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
- **Session timing log** is structured (CSV or JSON) with one row per
  playback. Columns: `sessionId, conditionContext, slotId, slotLabel,
  slotCondition, playStartMs, playEndMs, clipDurationMs, timestamp`. Times
  in ms via `performance.now()` (monotonic, sub-ms). Use it to detect PP
  barge-ins post-hoc.

## Files

| File | What it does |
|------|--------------|
| `services/app_api/app.py` (`/api/pitch-formant`) | Server-side pitch+formant bake |
| `services/app_api/pitch_formant.py` | WORLD-vocoder shift; duration-preserving |
| `frontend/src/lib/soundboardConfig.ts` | One source of truth for SR/channels/bit-depth/defaults |
| `frontend/src/lib/audioFormat.ts` | Decode / resample / encode WAV at PP rate, duration checks |
| `frontend/src/lib/soundboardDb.ts` | IndexedDB store for slots, targets, sessions |
| `frontend/src/lib/soundboardZip.ts` | Pure-JS store-mode zip writer + reader |
| `frontend/src/hooks/useSoundboard.ts` | Orchestration: record, bake, log, export |
| `frontend/src/hooks/useSoundboardPlayback.ts` | Drives opus-recorder over baked WAVs |
| `frontend/src/components/ConfigureSoundboard.tsx` | Slow-setup tab UI |
| `frontend/src/components/conversation/SoundboardPanel.tsx` | Runtime panel |

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

## What's not built (yet)

Marked nice-to-have in the spec, skipped for the first usable version:

- Auto-transcribe raw takes via Whisper for label suggestions
- Per-slot VC-quality badge (call `/api/vc-quality` after bake)
- VCTK corpus browser for target picker (a built-in default + uploads is
  enough for now)
- Built-in default target VC bake (needs server-side target-id resolution
  on `/api/voice-conversion`; for now upload a target WAV to use VC mode)
