# Distributed tracing (study mode)

End-to-end OpenTelemetry tracing of a participant's session across all processes:

```
browser ──▶ VC proxy :5002 (WebSocket) ──▶ app-api :5001 /condition ──▶ PersonaPlex :8000 (WS)
        └─▶ app-api :5001  /session/start, /save, /questionnaire …
analysis worker (offline batch)
```

**Off by default.** Nothing is exported until you point the services at a collector,
so prod is unaffected. When disabled the tracing code is a no-op.

## 1. Run a collector (Jaeger all-in-one accepts OTLP directly)

```bash
docker run -d --name jaeger \
  -p 16686:16686 -p 4318:4318 \
  -e COLLECTOR_OTLP_ENABLED=true \
  jaegertracing/all-in-one:latest
```

- `16686` — Jaeger UI
- `4318` — OTLP/HTTP ingest (what our services export to)

## 2. Enable tracing when starting the study stack

```bash
STUDY_TRACING=1 APP_MODE=study bash infra/run_all.sh
```

`STUDY_TRACING=1` sets `OTEL_TRACES_EXPORTER=otlp` and
`OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318` (override either if Jaeger is
elsewhere). These are inherited by app-api and, via `engine.py`, the on-demand VC
engine. Each process names itself — `study-app-api`, `xvc`, `meanvc`,
`study-analysis` — so **don't** set `OTEL_SERVICE_NAME` globally.

For local debugging without a collector, use `OTEL_TRACES_EXPORTER=console` to print
spans to stdout.

## 3. Read a session in the UI (http://<host>:16686)

- Filter by **service** (`study-app-api`, `xvc`, …), or
- Search by tag **`study.session_id=P01002_S02`** to pull every span for one scenario.

A scenario session shows as one trace (the frontend generates a `traceparent` per
`sessionStart` and reuses it for the REST calls + the chat-proxy WS):

```
POST /session/start        (study-app-api)
GET  /chat-proxy           (xvc)            study.session_id=P01002_S02  study.chunks=914
 ├─ personaplex.connect    (xvc, client)
 └─ GET /condition         (study-app-api)  study.session_id=P01002_S02
POST /session/{id}/save    (study-app-api)
```

Plus `vc.ensure_engine` → `vc.restart_engine` / `vc.load_targets` during prepare, and
`analysis.session` spans from the admin-triggered batch.

## How propagation works

- **REST**: the browser sends a W3C `traceparent` header (`lib/trace.ts`); FastAPI
  auto-instrumentation continues the trace.
- **WebSocket**: browsers can't set WS headers, so `traceparent` is passed as a query
  param and the aiohttp tracing middleware (`services/common/otel.py`) extracts it.
- **Service→service**: the proxy's outbound `/condition` call (aiohttp client
  instrumentation) and app-api's outbound `requests` (load-target) inject the context
  automatically.
- **PersonaPlex** is a third-party fork and doesn't emit spans; the `personaplex.connect`
  client span on the proxy side marks that hop.

## Dependencies

OTel packages are declared in each service's `pyproject.toml` (optional). Re-sync the
venvs after pulling: `uv sync` in `services/app_api`, `services/xvc`, `services/meanvc`.
If a resolver conflict appears (e.g. `protobuf` vs `tensorboard` in the xvc venv), pin
`protobuf` or drop `opentelemetry-exporter-otlp-proto-http` from that service — tracing
is optional and its absence degrades to a no-op.
