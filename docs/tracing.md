# Observability (study mode): traces + logs

End-to-end OpenTelemetry across all processes of a participant's session — **traces**
(spans) and **logs**, correlated by `trace_id`, viewed in an off-the-shelf OTel-native
UI (no custom viewer of ours):

```
browser ──▶ VC proxy :5002 (WebSocket) ──▶ app-api :5001 /condition ──▶ PersonaPlex :8000 (WS)
        └─▶ app-api :5001  /session/start, /save, /questionnaire …
analysis worker (offline batch)
```

**Off by default.** Nothing is exported until you point the services at an OTLP
collector, so prod is unaffected. When disabled, the tracing/logging code is a no-op.

## 1. Run a backend (one container: Grafana + Tempo + Loki)

Grafana's `otel-lgtm` bundles Grafana (UI), Tempo (traces) and Loki (logs) and accepts
OTLP directly — the lightest way to see traces **and** logs together:

```bash
docker run -d --name lgtm \
  -p 3000:3000 -p 4318:4318 -p 4317:4317 \
  grafana/otel-lgtm:latest
```

- `3000` — Grafana UI
- `4318` / `4317` — OTLP HTTP / gRPC ingest (our services export to 4318)

(Traces-only alternative: Jaeger all-in-one with `COLLECTOR_OTLP_ENABLED=true` on
`16686`/`4318` — but it doesn't store logs. Prefer LGTM for both. SigNoz/Uptrace are
heavier all-in-one OTel platforms if you want metrics + long retention too.)

## 2. Enable when starting the study stack

```bash
STUDY_TRACING=1 APP_MODE=study bash infra/run_all.sh
```

Sets `OTEL_TRACES_EXPORTER=otlp` and `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318`
(override if the backend is elsewhere). Both **traces** (`/v1/traces`) and **logs**
(`/v1/logs`) export to that endpoint. Inherited by app-api and, via `engine.py`, the
on-demand VC engine. Each process self-names — `study-app-api`, `xvc`, `meanvc`,
`study-analysis` — so **don't** set `OTEL_SERVICE_NAME` globally.

For local span debugging without a backend: `OTEL_TRACES_EXPORTER=console` prints spans
to stdout (logs export stays off in that mode).

## 3. Read a session in Grafana (http://<host>:3000)

- **Traces** — Explore → Tempo. Search by service or by tag
  **`study.session_id = P01002_S02`**. A scenario shows as one trace:

  ```
  POST /session/start        (study-app-api)
  GET  /chat-proxy           (xvc)        study.session_id=P01002_S02  study.chunks=914
   ├─ personaplex.connect    (xvc, client)
   └─ GET /condition         (study-app-api)
  POST /session/{id}/save    (study-app-api)
  ```
  Plus `vc.ensure_engine → restart_engine / load_targets` (prepare) and
  `analysis.session` spans (batch).

- **Logs** — Explore → Loki. Every log line carries `trace_id`, `span_id`, and (where
  set) `session_id`. From a trace span, use Grafana's trace→logs correlation to jump to
  the exact log lines for that request; or filter Loki by `service_name` / `trace_id`.

## How it's wired

- **Traces**: `services/common/otel.py` (FastAPI + requests + aiohttp-client
  instrumentation, a WS-aware aiohttp middleware, manual spans).
- **Logs**: `services/common/logging_setup.py` attaches an OTel `LoggingHandler` to the
  root logger, so stdlib `logging` calls export over OTLP with the active span's
  trace/span id attached automatically; `set_log_session()` adds `session_id`.
- **Propagation**: browser sends W3C `traceparent` (header on REST, query param on the
  WS since browsers can't set WS headers); services continue the trace and inject it
  onward. Console logs on stdout / `/tmp/hmo_vc_engine.log` are unchanged.

## Dependencies

Declared (optional) in each service's `pyproject.toml`. Re-sync after pulling:
`uv sync` in `services/app_api`, `services/xvc`, `services/meanvc`. If a resolver
conflict appears (e.g. `protobuf` vs `tensorboard` in the xvc venv), pin `protobuf` or
drop `opentelemetry-exporter-otlp-proto-http` there — observability is optional and its
absence degrades to a no-op.
