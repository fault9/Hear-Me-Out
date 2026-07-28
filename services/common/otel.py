"""Shared OpenTelemetry bootstrap for the study services.

One helper used by all three Python processes (app-api FastAPI, X-VC/MeanVC aiohttp
proxies, the analysis worker) so a participant's session can be traced end-to-end:

    browser ──▶ VC proxy :5002 (WS) ──▶ app-api :5001 /condition ──▶ PersonaPlex :8000 (WS)

Design goals:
- **Zero-cost and safe when unconfigured.** If the OTel packages aren't installed,
  or no exporter is configured, everything degrades to a no-op — the services run
  exactly as before. Tracing turns on only when `OTEL_TRACES_EXPORTER` (or
  `OTEL_EXPORTER_OTLP_ENDPOINT`) is set, so prod isn't affected until you point it
  at a collector (Jaeger).
- **One correlation key.** Every span carries `study.session_id` (e.g. P01002_S02)
  as an attribute, so you can search Jaeger by session regardless of trace linkage.
- **WebSocket propagation.** Browsers can't set request headers on a WebSocket, so
  the frontend passes W3C `traceparent` as a query param; `extract_context` reads it.

Env:
  OTEL_TRACES_EXPORTER            "otlp" (default when endpoint set) | "console" | "none"
  OTEL_EXPORTER_OTLP_ENDPOINT     collector base, default http://127.0.0.1:4318 (OTLP/HTTP)
  OTEL_SERVICE_NAME               overrides the service name passed to init_tracing
  OTEL_SDK_DISABLED               "true" hard-disables even if other vars are set
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:  # OTel is optional — absence must not break the services.
    from opentelemetry import trace
    from opentelemetry.propagate import extract, inject
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import SpanKind, Status, StatusCode
    _HAVE_OTEL = True
except Exception:  # noqa: BLE001
    _HAVE_OTEL = False

_ENABLED = False


def _want_tracing() -> bool:
    if os.environ.get("OTEL_SDK_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    exporter = os.environ.get("OTEL_TRACES_EXPORTER", "").lower()
    if exporter in ("none", "false"):
        return False
    return bool(exporter) or bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


def init_tracing(service_name: str) -> bool:
    """Configure the global tracer provider for this process. Idempotent. Returns
    True if tracing is active (so callers can decide whether to add instrumentation)."""
    global _ENABLED
    if _ENABLED:
        return True
    if not _HAVE_OTEL or not _want_tracing():
        return False

    name = os.environ.get("OTEL_SERVICE_NAME", service_name)
    provider = TracerProvider(resource=Resource.create({"service.name": name}))

    exporter_kind = os.environ.get("OTEL_TRACES_EXPORTER", "otlp").lower()
    if exporter_kind == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:  # otlp/http -> collector (Jaeger all-in-one accepts OTLP directly)
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")))
        except Exception:  # noqa: BLE001 - bad/missing exporter shouldn't crash the app
            return False

    trace.set_tracer_provider(provider)
    _ENABLED = True
    return True


def is_enabled() -> bool:
    return _ENABLED


def get_tracer(name: str):
    """A real tracer when OTel is present, else a no-op with the same surface."""
    if _HAVE_OTEL:
        return trace.get_tracer(name)
    return _NoopTracer()


def set_session_attributes(**attrs: Any) -> None:
    """Stamp attributes (session_id, engine, scenario…) on the current span."""
    if not _HAVE_OTEL:
        return
    span = trace.get_current_span()
    if span is None or not span.is_recording():
        return
    for k, v in attrs.items():
        if v is not None:
            span.set_attribute(k if k.startswith("study.") else f"study.{k}", v)


# ---- metrics (latency histograms, graphable in the observability UI) ----
_METRICS_ENABLED = False
_hist_cache: dict = {}

# The currently-active study session (single live run), used to tag device-level GPU
# metrics so they can be sliced per session/study in dashboards.
_active_session: dict = {"session_id": None, "study_id": None}


def set_active_session(session_id, study_id=None) -> None:
    _active_session["session_id"] = session_id or None
    _active_session["study_id"] = str(study_id) if study_id is not None else None


def init_metrics(service_name: str) -> bool:
    """Set up the OTLP metrics pipeline (same enable gate + endpoint as tracing).
    Idempotent; returns True if metrics are active."""
    global _METRICS_ENABLED
    if _METRICS_ENABLED:
        return True
    if not _HAVE_OTEL or not _want_tracing():
        return False
    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        if os.environ.get("OTEL_TRACES_EXPORTER", "otlp").lower() == "console":
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
            reader = PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=10000)
        else:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=f"{endpoint.rstrip('/')}/v1/metrics"),
                export_interval_millis=10000)
        metrics.set_meter_provider(MeterProvider(
            resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", service_name)}),
            metric_readers=[reader]))
        _METRICS_ENABLED = True
    except Exception:  # noqa: BLE001
        return False
    return True


def init_gpu_metrics(service_name: str) -> bool:
    """Emit NVIDIA GPU metrics (utilization / memory / temp / power) as OTel observable
    gauges, polled from NVML at export time. OTel does NOT collect GPU stats itself —
    this is the source. No-op unless metrics are enabled and NVML is available (needs
    the NVIDIA driver + `nvidia-ml-py`); safe on CPU-only / dev boxes.

    Device-level metrics describe the whole GPU regardless of which process reads them,
    so call this from ONE always-on process (app-api) to avoid double-counting."""
    if not init_metrics(service_name):
        return False
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    except Exception:  # noqa: BLE001 - no driver / not installed
        return False

    from opentelemetry import metrics
    from opentelemetry.metrics import Observation

    def _obs(fn):
        def _cb(_options):
            out = []
            # Tag with the active session/study (only one runs at a time), so GPU load
            # is attributable per session in dashboards.
            sid = _active_session.get("session_id")
            stid = _active_session.get("study_id")
            for i, h in enumerate(handles):
                attrs = {"gpu": i}
                if sid:
                    attrs["study.session_id"] = sid
                if stid:
                    attrs["study.study_id"] = stid
                try:
                    out.append(Observation(fn(h), attrs))
                except Exception:  # noqa: BLE001 - one bad read shouldn't drop the rest
                    pass
            return out
        return _cb

    m = metrics.get_meter("study.gpu")
    m.create_observable_gauge("gpu.utilization", unit="%",
        callbacks=[_obs(lambda h: pynvml.nvmlDeviceGetUtilizationRates(h).gpu)])
    m.create_observable_gauge("gpu.memory.used_mib", unit="MiB",
        callbacks=[_obs(lambda h: pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024))])
    m.create_observable_gauge("gpu.memory.total_mib", unit="MiB",
        callbacks=[_obs(lambda h: pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 * 1024))])
    m.create_observable_gauge("gpu.temperature_c", unit="Cel",
        callbacks=[_obs(lambda h: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))])
    m.create_observable_gauge("gpu.power_w", unit="W",
        callbacks=[_obs(lambda h: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)])
    return True


def record_latency(name: str, value_ms: float, **attrs) -> None:
    """Record a latency (ms) into a histogram. No-op unless metrics are active.
    Attribute keys are namespaced under `study.` unless already dotted."""
    if not _METRICS_ENABLED:
        return
    try:
        from opentelemetry import metrics
        h = _hist_cache.get(name)
        if h is None:
            h = metrics.get_meter("study").create_histogram(name, unit="ms", description=name)
            _hist_cache[name] = h
        clean = {(k if "." in k else f"study.{k}"): v for k, v in attrs.items() if v is not None}
        h.record(value_ms, clean)
    except Exception:  # noqa: BLE001
        pass


def extract_context(headers: Optional[dict] = None, traceparent: Optional[str] = None,
                    tracestate: Optional[str] = None):
    """Build a parent context from inbound headers and/or an explicit traceparent
    (used for WebSockets, where the browser passes it as a query param)."""
    if not _HAVE_OTEL:
        return None
    carrier = dict(headers or {})
    if traceparent and "traceparent" not in {k.lower(): k for k in carrier}:
        carrier["traceparent"] = traceparent
        if tracestate:
            carrier["tracestate"] = tracestate
    return extract(carrier)


def inject_headers(headers: dict) -> dict:
    """Inject the current trace context into an outbound header dict (in place)."""
    if _HAVE_OTEL:
        inject(headers)
    return headers


@contextmanager
def start_span(tracer, name: str, *, context=None, kind: str = "internal",
               attributes: Optional[dict] = None) -> Iterator[Any]:
    """Uniform span helper that records exceptions and sets ERROR status on raise."""
    if not _HAVE_OTEL:
        yield _NoopSpan()
        return
    span_kind = {"server": SpanKind.SERVER, "client": SpanKind.CLIENT,
                 "producer": SpanKind.PRODUCER, "consumer": SpanKind.CONSUMER}.get(kind, SpanKind.INTERNAL)
    with tracer.start_as_current_span(name, context=context, kind=span_kind) as span:
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(k, v)
        try:
            yield span
        except Exception as e:  # noqa: BLE001
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


# ---- aiohttp server middleware (extracts trace context incl. WS query param) ----
def aiohttp_middleware(service_name: str):
    """A @web.middleware that opens a SERVER span per request, reading trace context
    from headers or the `traceparent` query param (WebSocket handshakes)."""
    from aiohttp import web

    tracer = get_tracer(service_name)

    @web.middleware
    async def _mw(request: "web.Request", handler):
        ctx = extract_context(dict(request.headers),
                              traceparent=request.query.get("traceparent"),
                              tracestate=request.query.get("tracestate"))
        attrs = {"http.method": request.method, "http.route": request.path}
        sid = request.query.get("session_id")
        if sid:
            attrs["study.session_id"] = sid
        with start_span(tracer, f"{request.method} {request.path}", context=ctx,
                        kind="server", attributes=attrs) as span:
            resp = await handler(request)
            if hasattr(span, "set_attribute") and getattr(resp, "status", None) is not None:
                span.set_attribute("http.status_code", resp.status)
            return resp

    return _mw


# ---- instrumentation shims (guarded; only call when init_tracing returned True) ----
def instrument_fastapi(app) -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except Exception:  # noqa: BLE001
        pass


def instrument_requests() -> None:
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument()
    except Exception:  # noqa: BLE001
        pass


def instrument_aiohttp_client() -> None:
    try:
        from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
        AioHttpClientInstrumentor().instrument()
    except Exception:  # noqa: BLE001
        pass


# ---- no-op fallbacks so call sites need no guards when OTel is absent ----
class _NoopSpan:
    def set_attribute(self, *_a, **_k): ...
    def set_status(self, *_a, **_k): ...
    def record_exception(self, *_a, **_k): ...
    def is_recording(self): return False


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, *_a, **_k):
        yield _NoopSpan()
