# Observability (study mode): traces + logs

End-to-end OpenTelemetry across every process of a participant's session — **traces**
(spans) and **logs**, correlated by `trace_id` — viewed in a bundled OTel-native UI
(OpenObserve) served **through app-api at `/logs` on :5001**. No separate container, no
extra exposed port, no custom viewer of ours.

```
browser ──▶ VC proxy :5002 (WebSocket) ──▶ app-api :5001 /condition ──▶ PersonaPlex :8000 (WS)
        └─▶ app-api :5001  /session/start, /save, /questionnaire …
analysis worker (offline batch)
```

**Off by default.** Nothing is installed or exported unless you opt in; when disabled
the tracing/logging code is a no-op.

## 1. Install the backend (once, via setup)

```bash
bash infra/setup.sh --observability      # or answer "Y" to the observability prompt
```

This downloads the **OpenObserve** single binary (traces + logs + UI, OTLP-native) to
`$WORKSPACE/observability/`. Nothing else to run — it's a static binary, not a
container. (Standalone: `bash infra/observability.sh install`; pin a release with
`O2_VERSION=vX.Y.Z` if the default tag is unavailable for your platform.)

## 2. Start the stack with observability on

```bash
STUDY_OBSERVABILITY=1 APP_MODE=study bash infra/run_all.sh
```

run_all then:
- starts OpenObserve (data under `$STUDY_DATA_ROOT/observability`, served under `/logs`),
- points every service's OTLP exporter at it (traces `/v1/traces` + logs `/v1/logs`),
- sets `STUDY_OBSERVABILITY_URL` so **app-api reverse-proxies the UI under `/logs`**.

Each process self-names — `study-app-api`, `xvc`, `meanvc`, `study-analysis` — so don't
set `OTEL_SERVICE_NAME`. The OTLP vars are inherited by app-api and (via `engine.py`)
the on-demand VC engine.

### Open the UI

`https://<host>:5001/logs` — OpenObserve, proxied by app-api on the same port the study
runs on. Default login `admin@example.com` / `ChangeMe123` (override with
`O2_ROOT_USER_EMAIL` / `O2_ROOT_USER_PASSWORD`).

- **Traces**: search by service or tag **`study.session_id = P01002_S02`**. A scenario
  is one trace: `POST /session/start` → `GET /chat-proxy` (`study.chunks=914`,
  `personaplex.connect`, `GET /condition`) → `POST /save`, plus `vc.ensure_engine`
  (prepare) and `analysis.session` (batch).
- **Logs**: every line carries `trace_id`, `span_id`, and (where set) `session_id` —
  filter by any of them, or pivot from a span to its logs by `trace_id`.

### External collector instead

If you'd rather use your own OTLP backend (Grafana Tempo/Loki, SigNoz, …) and not the
bundled UI, skip `--observability` and use `STUDY_TRACING=1` with
`OTEL_EXPORTER_OTLP_ENDPOINT=http://your-collector:4318`.

## Latency metrics

Alongside traces/logs, the services emit OTel **metric histograms** (graphable in
OpenObserve → Metrics; unit ms) so you can watch p50/p95 model + network latency:

| Metric | Where measured | Meaning |
|---|---|---|
| `vc.inference_ms` `{engine}` | VC proxy, per window/chunk | X-VC / MeanVC GPU conversion time |
| `personaplex.first_response_ms` `{engine}` | VC proxy | time from first converted audio sent → first PersonaPlex audio back |
| `client.network_rtt_ms` | browser → app-api `/ping` | browser↔server network round-trip |
| `client.connect_ms` | browser | call start → PersonaPlex handshake (via proxy) |
| `client.first_audio_ms` | browser | call start → first model audio heard (end-to-end) |

The same numbers also ride on the trace: the `GET /chat-proxy` span carries
`study.first_response_ms` and `study.vc_inference_avg_ms`, and the browser posts its
marks to `POST /api/study/telemetry` (recorded + logged, tagged with the session). So in
a single trace you can see the network hop, the VC compute, and PersonaPlex's response
time for that scenario.

### GPU metrics

OTel does **not** collect GPU stats on its own — we add them as the source. app-api polls
**NVML** (needs the NVIDIA driver + `nvidia-ml-py`) and emits observable gauges every ~10 s,
per device (`{gpu}`): `gpu.utilization` (%), `gpu.memory.used_mib`, `gpu.memory.total_mib`,
`gpu.temperature_c`, `gpu.power_w`. Since the GPU is shared by PersonaPlex + X-VC, graph
these against `vc.inference_ms` / `personaplex.first_response_ms` to see how GPU load tracks
latency (e.g. contention when the analysis batch runs). No-op on CPU-only boxes.

## How it's wired

- **Traces**: `services/common/otel.py` — FastAPI + requests + aiohttp-client
  instrumentation, a WS-aware aiohttp middleware (reads `traceparent` from the WS query,
  since browsers can't set WS headers), and manual spans.
- **Logs**: `services/common/logging_setup.py` attaches an OTel `LoggingHandler` to the
  root logger, so stdlib `logging` exports over OTLP with the active span's trace/span
  id attached; `set_log_session()` adds `session_id`. stdout / `/tmp/hmo_vc_engine.log`
  stay human-readable.
- **UI proxy**: `app.py` streams `/logs*` to `STUDY_OBSERVABILITY_URL` (only when set).
- **Backend lifecycle**: `infra/observability.sh` (install/start/stop/status).

## Dependencies

OTel packages (+ `httpx` for the proxy) are declared in the service `pyproject.toml`s.
Re-sync after pulling: `uv sync` in `services/app_api`, `services/xvc`, `services/meanvc`.
If a resolver conflict appears (e.g. `protobuf` vs `tensorboard` in the xvc venv), pin
`protobuf` or drop `opentelemetry-exporter-otlp-proto-http` there — observability is
optional and its absence degrades to a no-op.
