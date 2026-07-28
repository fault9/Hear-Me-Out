"""Minimal MCP server exposing the study's OpenObserve data (traces / logs / metrics)
so an assistant can query it directly.

Runs locally (stdio transport), talks to OpenObserve over HTTP using its search API.
Point it at the app-api-proxied base URL (…:5001/logs) so it goes through the same
port you use in the browser. Self-signed TLS is accepted (verify=False).

Env:
  O2_URL       base incl. the /logs sub-path, e.g. https://130.237.3.103:5001/logs
  O2_USER      OpenObserve login email (default admin@example.com)
  O2_PASSWORD  OpenObserve password    (default ChangeMe123)
  O2_ORG       organization            (default "default")

Run:  uv run --with "mcp[cli]" --with httpx python infra/o2-mcp/server.py
"""

from __future__ import annotations

import json
import os
import time

import httpx
from fastmcp import FastMCP

O2_URL = os.environ.get("O2_URL", "https://127.0.0.1:5001/logs").rstrip("/")
O2_ORG = os.environ.get("O2_ORG", "default")
O2_USER = os.environ.get("O2_USER", "admin@example.com")
O2_PASSWORD = os.environ.get("O2_PASSWORD", "ChangeMe123")

_client = httpx.Client(verify=False, timeout=30.0, auth=(O2_USER, O2_PASSWORD))
mcp = FastMCP("openobserve")


def _micros(seconds_ago: float) -> int:
    return int((time.time() - seconds_ago) * 1_000_000)


def _err(where: str, e: Exception) -> str:
    return f"ERROR {where}: {type(e).__name__}: {e}"


@mcp.tool()
def list_streams(stream_type: str = "logs") -> str:
    """List OpenObserve streams. stream_type: logs | metrics | traces."""
    try:
        r = _client.get(f"{O2_URL}/api/{O2_ORG}/streams", params={"type": stream_type})
        r.raise_for_status()
        data = r.json()
        names = [s.get("name") for s in data.get("list", data if isinstance(data, list) else [])]
        return json.dumps({"stream_type": stream_type, "streams": names}, indent=2)
    except Exception as e:  # noqa: BLE001
        return _err("list_streams", e)


@mcp.tool()
def search(sql: str, stream_type: str = "logs", minutes: int = 30, size: int = 50) -> str:
    """Run a SQL query against OpenObserve over the last `minutes`.

    stream_type: logs | metrics | traces. Use double-quoted stream names, e.g.
    `SELECT * FROM "study-app-api" WHERE session_id = 'P01002_S02' ORDER BY _timestamp DESC`
    or `SELECT _timestamp, value, gpu FROM "gpu_utilization"`.
    """
    try:
        body = {"query": {"sql": sql, "start_time": _micros(minutes * 60),
                          "end_time": _micros(0), "from": 0, "size": size}}
        r = _client.post(f"{O2_URL}/api/{O2_ORG}/_search", params={"type": stream_type}, json=body)
        if r.status_code >= 400:
            return f"HTTP {r.status_code}: {r.text[:1000]}"
        hits = r.json().get("hits", [])
        return json.dumps({"count": len(hits), "hits": hits}, indent=2, default=str)[:60000]
    except Exception as e:  # noqa: BLE001
        return _err("search", e)


@mcp.tool()
def recent_logs(minutes: int = 30, size: int = 50, contains: str = "",
                service: str = "", session_id: str = "") -> str:
    """Recent structured log lines (all services share the `default` logs stream;
    fields: severity, service_name, session_id, body). Optionally filter by `service`
    (study-app-api | xvc | meanvc | study-analysis), a `session_id`, and/or a `contains`
    substring in the body."""
    conds = []
    if service:
        conds.append(f"service_name = '{service}'")
    if session_id:
        conds.append(f"session_id = '{session_id}'")
    if contains:
        conds.append(f"body LIKE '%{contains}%'")
    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    sql = f'SELECT _timestamp, severity, service_name, session_id, body FROM "default"{where} ORDER BY _timestamp DESC'
    return search(sql, stream_type="logs", minutes=minutes, size=size)


@mcp.tool()
def latest_metric(metric: str, minutes: int = 15, size: int = 50) -> str:
    """Latest samples of a metric stream (e.g. gpu_utilization, vc_inference_ms,
    personaplex_first_response_ms, client_network_rtt_ms). Returns recent (ts, value)."""
    sql = f'SELECT _timestamp, value, service_name FROM "{metric}" ORDER BY _timestamp DESC'
    return search(sql, stream_type="metrics", minutes=minutes, size=size)


if __name__ == "__main__":
    mcp.run()
