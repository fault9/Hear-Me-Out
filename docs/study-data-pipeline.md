# Study data and post-hoc analysis

## Collection contract

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

## Post-hoc processing

Use the admin Data tab after collection. `Session preprocessing and diagnostics`
transcribes and summarizes whole recordings, and must not be used as a
route-specific VC comparison for switching sessions.

`Voice-conversion quality` runs the repository's real `vc_quality.py` for one
session, one participant, or the full study. It:

- reads the frozen raw, transmitted, target, and event artifacts;
- derives stable route clips using actual input/transmitted sample boundaries;
- excludes a guard interval around switches from stable VC scoring;
- runs WER/CER, speaker similarity, prosody, and naturalness on VC-only clips;
- saves transition windows and activation/discontinuity diagnostics separately;
- writes a new `analysis/vc_quality/<analysis-id>/` snapshot on every forced run.

AudioBox aesthetics is optional (`uv sync --extra aesthetics` in `services/app_api`).
Missing AudioBox produces null aesthetic fields, never mock scores. AudioBox is
not used for latency.

Before preregistration/data collection, freeze the transition guard duration,
transition-window duration, RMS speech threshold/hangover, vc_quality model
versions, and rules for failed/incomplete captures. Validate the RMS speech
estimate against a manually annotated subset; packet-gap assistant speech is an
estimate, not diarization ground truth.

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
```

The mapped values are target `ref` values uploaded for the study. The example
uses gender identity because that is the stated protocol. If the intended rule
is instead based on perceived vocal presentation, use a dedicated required
categorical item and preregister how ambiguous or undisclosed answers are
handled.

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
