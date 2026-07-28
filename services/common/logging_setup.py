"""Bridge stdlib logging → OpenTelemetry logs (OTLP), shared by all study services.

`init_logging(service)` attaches an OTel `LoggingHandler` to the root logger, so every
`logging` call is exported over OTLP alongside the traces — with the active span's
`trace_id`/`span_id` stamped on each record automatically. Point the services at an
OTel-native backend (e.g. Grafana's `otel-lgtm`: Grafana + Tempo + Loki in one
container) and you get traces and logs in one UI, correlated by trace id. No custom log
viewer of our own.

It is **additive and gated**: the existing stdout handlers stay (console logs unchanged),
and nothing is exported unless the same `OTEL_*` config that enables tracing is present.
When OTel isn't installed or isn't configured, this is a no-op.

`set_log_session(session_id)` tags subsequent records with `session_id`, which rides
along as a log attribute (searchable in Loki), complementing the `study.session_id`
span attribute.
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar

try:
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    _HAVE_OTEL_LOGS = True
except Exception:  # noqa: BLE001
    _HAVE_OTEL_LOGS = False

_session_var: ContextVar = ContextVar("study_log_session", default=None)
_configured: set = set()


def set_log_session(session_id) -> None:
    """Associate subsequent log records on this task/thread with a session id."""
    _session_var.set(session_id or None)


# Loggers to DROP from OTLP export (they don't reach the observability backend, but
# still print to stdout). httpx/httpcore/urllib3 are the HTTP clients used by the OTLP
# exporter and the app-api /logs reverse-proxy — every proxied UI request logs an
# "HTTP Request: …" line that would otherwise flood the logs stream and feed back on
# itself. Override with STUDY_LOG_SUPPRESS (comma-separated logger-name prefixes; empty
# to disable).
_SUPPRESS_PREFIXES = tuple(
    p.strip() for p in os.environ.get(
        "STUDY_LOG_SUPPRESS", "httpx,httpcore,urllib3").split(",") if p.strip())


class _SessionFilter(logging.Filter):
    """Drop noisy HTTP-client logs from export; stamp the session id on the rest."""
    def filter(self, record: logging.LogRecord) -> bool:
        if _SUPPRESS_PREFIXES and record.name.startswith(_SUPPRESS_PREFIXES):
            return False
        sid = _session_var.get()
        if sid:
            record.session_id = sid
        return True


def _want_export() -> bool:
    # Same enable signal as tracing (see common.otel), and OTLP logs need an endpoint.
    if os.environ.get("OTEL_SDK_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    exporter = os.environ.get("OTEL_TRACES_EXPORTER", "").lower()
    if exporter in ("none", "false", "console"):  # console exporter has no logs pipeline
        return False
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")) or exporter == "otlp"


def init_logging(service_name: str, level: int = logging.INFO) -> None:
    """Route stdlib logging to OTLP for this service. Idempotent; no-op if unconfigured."""
    if service_name in _configured:
        return
    _configured.add(service_name)
    if not _HAVE_OTEL_LOGS or not _want_export():
        return
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    try:
        provider = LoggerProvider(resource=Resource.create(
            {"service.name": os.environ.get("OTEL_SERVICE_NAME", service_name)}))
        provider.add_log_record_processor(BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=f"{endpoint.rstrip('/')}/v1/logs")))
        set_logger_provider(provider)
        handler = LoggingHandler(level=level, logger_provider=provider)
        handler.addFilter(_SessionFilter())
        root = logging.getLogger()
        root.addHandler(handler)
        if root.level > level or root.level == logging.NOTSET:
            root.setLevel(level)
    except Exception:  # noqa: BLE001 - logging export must never break the service
        pass
