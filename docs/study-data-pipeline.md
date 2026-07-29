# Study data and post-hoc analysis

## Collection contract

Each run enforces `eligibility -> consent -> background -> audio check -> scenarios`.
Consent answers are persisted before the audio-check endpoint permits microphone
access; answers from an earlier restarted run do not satisfy this guard.

Each scenario attempt has a unique session ID and directory:

`sessions/study_<id>/<participant>/run_<n>/scenario_<n>/attempt_<n>_<session-id>/`

The directory contains frozen study/session configuration, a copied target WAV,
raw microphone audio, transmitted participant audio, assistant/merged audio,
model transcript, an append-only event timeline, hashes, WAV metadata, and final
capture status. Restarting a run or scenario creates a new attempt; it never
reuses or overwrites a previous recording.

The VC proxy records monotonic event sequence numbers and sample offsets for
route requests and activations, input chunks, transmitted windows, model-bound
packets, output packets, estimated speech boundaries, and inference failures.
The browser separately reports ScriptProcessor callback gaps as an estimate;
these are labelled `reported_by: browser` and must not be described as
server-observed packet loss.

## Timing timelines

Study calls keep two linked timelines without running model inference in the
browser or adding a synchronous network request to the audio loop:

- The participant-experienced timeline maps raw microphone callback samples and
  scheduled assistant playback packets to the PersonaPlex-handshake
  `performance.now()` epoch. `client_timeline.json` records callback sample
  offsets, scheduled playback spans, and browser output-latency estimates.
- The proxy/model timeline records X-VC input, route activation, transmitted
  windows, exact PersonaPlex-bound Opus bytes, PersonaPlex output packets, and
  monotonic server timestamps in `events.jsonl`, `proxy_timeline.json`, and the
  proxy audio artifacts.

The browser prefixes study-only PCM frames with a sequence number and capture
sample offset. The X-VC proxy copies those identifiers into `input_chunk` events,
providing the crosswalk between timelines. The header is removed before audio is
processed and regular non-study calls retain the legacy raw-float frame format.

Admin-triggered preprocessing writes versioned
`analysis/timing/<analysis-id>/timing.json` outputs with overlap, barge-in,
stop-latency, route-switch projection, and crosswalk integrity diagnostics. RMS
participant boundaries and decoded-playback RMS assistant boundaries remain
labelled `estimated_pending_validation`; use `validate_intervals` with manually
annotated pilot intervals before treating these measures as validated outcomes.
New captures store packet RMS on the browser AudioContext schedule. Existing
captures fall back to RMS over the silence-preserving `model.wav` artifact.

## Post-hoc processing

Use the admin Data tab after collection. The primary `Analysis pipeline` action
first transcribes recordings and derives interaction timing, then automatically
runs VC quality. It is backend-owned and continues if the admin page is closed.
The general transcript includes raw participant text, browser-clock participant
segments with route labels, the transmitted transcript, and PersonaPlex turns.
Whole-recording metrics must not be used as a route-specific VC comparison for
switching sessions.

The optional `VC-quality rerun` control runs the repository's real
`vc_quality.py` for one session, one participant, or the full study. It:

- reads the frozen raw, transmitted, target, and event artifacts;
- derives stable route clips using actual input/transmitted sample boundaries;
- excludes a guard interval around switches from stable VC scoring;
- runs the X-VC objective profile (WER, WavLM SIM, and UTMOS) on stable VC-only clips;
- records WER reference provenance as source-ASR because interactions are unscripted;
- saves transition windows and activation/discontinuity diagnostics separately;
- writes a new `analysis/vc_quality/<analysis-id>/` snapshot on every forced run.

AudioBox aesthetics is optional (`uv sync --extra aesthetics` in `services/app_api`).
Missing AudioBox produces null aesthetic fields, never mock scores. AudioBox is
not used for latency.

Before preregistration/data collection, freeze the transition guard duration,
transition-window duration, RMS speech threshold/hangover, vc_quality model
versions, and rules for failed/incomplete captures. Validate participant and
assistant RMS speech estimates against a manually annotated subset; neither
should be described as diarization ground truth.

## Gender-conditional target assignment and counterbalancing

When the protocol requires an opposite-gender-presenting target, participant
codes are generated provisionally. Submitting the configured background item
fixes the target and assigns the next least-filled design variant within that
answer category. A scenario cannot start before this immutable assignment.

```yaml
counterbalancing:
  target_assignment:
    questionnaire_kind: background
    answer_id: gender_identity
    target_by_answer:
      Woman: masculine_presenting
      Man: feminine_presenting
    fallback_targets: [masculine_presenting, feminine_presenting]
```

The mapped values are target `ref` values uploaded for the study. The example
uses gender identity because that is the stated protocol. An unmapped response,
including non-binary, self-described, or undisclosed, is assigned randomly among
the currently least-used `fallback_targets`; those participants share one
fallback allocation stratum so scenario variants remain balanced. If the
intended rule is instead based on perceived vocal presentation, use a dedicated
required categorical item and preregister its mapping and fallback rule.

Scenario counterbalancing is also prespecified in YAML. Variants may omit
`target_ref` when `target_assignment` is configured; the assigned target is
inserted into every VC segment after the background response. For two scenario
definitions and two conditions, the shape is:

```yaml
counterbalancing:
  conditions:
    natural_then_vc:
      voice_schedule:
        - {mode: natural, start_s: 0, end_s: 120}
        - {mode: vc, engine: xvc, start_s: 120, end_s: null}
    vc_then_natural:
      voice_schedule:
        - {mode: vc, engine: xvc, start_s: 0, end_s: 120}
        - {mode: natural, start_s: 120, end_s: null}
  variants:
    - id: A1
      scenario_order: [1, 2]
      condition_assignment: {1: natural_then_vc, 2: vc_then_natural}
    - id: B1
      scenario_order: [2, 1]
      condition_assignment: {1: vc_then_natural, 2: natural_then_vc}
```

`scenario_order` and `condition_assignment` use the 1-based positions of the
scenario definitions in the YAML. Define the complete variant table manually.
The platform validates it, balances variants separately within each configured
gender category, and exports each participant's immutable category, target,
variant, and current balance counts. Without `variants`, target assignment still
works and the YAML scenario order/schedules are used as the single `default`
variant.
