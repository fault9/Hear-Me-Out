# OpenObserve MCP server

Lets an assistant (Claude Code) query the study's OpenObserve data — traces, logs, and
metrics — directly, instead of you shuttling logs by hand.

It runs **locally** (on your Mac, where Claude Code runs) over stdio, and talks to
OpenObserve through the app-api proxy at `…:5001/logs` (the same URL you open in the
browser). Self-signed TLS is accepted.

## 1. Add it to Claude Code

`claude mcp add` (run from the repo):

```bash
claude mcp add openobserve \
  --env O2_URL=https://130.237.3.103:5001/logs \
  --env O2_USER=admin@example.com \
  --env O2_PASSWORD=ChangeMe123 \
  --env O2_ORG=default \
  -- uv run --with "mcp[cli]" --with httpx python "$(pwd)/infra/o2-mcp/server.py"
```

…or add it to `.mcp.json` at the repo root:

```json
{
  "mcpServers": {
    "openobserve": {
      "command": "uv",
      "args": ["run", "--with", "mcp[cli]", "--with", "httpx",
               "python", "infra/o2-mcp/server.py"],
      "env": {
        "O2_URL": "https://130.237.3.103:5001/logs",
        "O2_USER": "admin@example.com",
        "O2_PASSWORD": "ChangeMe123",
        "O2_ORG": "default"
      }
    }
  }
}
```

Then restart Claude Code (or reload MCP servers). The assistant will have
`mcp__openobserve__*` tools.

## 2. Tools

- `list_streams(stream_type)` — logs | metrics | traces.
- `search(sql, stream_type, minutes, size)` — raw SQL over a time window. Stream names
  are double-quoted, e.g.
  `SELECT * FROM "study-app-api" WHERE session_id='P01002_S02' ORDER BY _timestamp DESC`.
- `recent_logs(minutes, size, contains, service)` — recent structured log lines.
- `latest_metric(metric, minutes, size)` — recent samples of a metric stream
  (`gpu_utilization`, `vc_inference_ms`, `personaplex_first_response_ms`, …).

## Notes / caveats

- **Reachability**: your Mac must be able to reach `130.237.3.103:5001` (it can — that's
  the browser URL). If you tunnel instead, point `O2_URL` at the tunnel.
- **Secret**: this puts the OpenObserve password in your MCP config. Keep `.mcp.json`
  out of git if you commit real creds (the repo copy uses the placeholder default).
- **Untested against your build**: OpenObserve's search API path/shape can vary by
  version (`v0.14.4` here). If a tool returns an HTTP error, the raw response is included
  so it's easy to adjust the SQL/endpoint in `server.py`.
- Metric names are sanitized (dots→underscores); histograms may appear as
  `<name>_bucket` / `_sum` / `_count`. Use `list_streams("metrics")` to see the truth.
