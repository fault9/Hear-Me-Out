# Observability dashboards

Three ready-made OpenObserve dashboards for the study stack. Traces and logs are also
explorable directly (Traces / Logs menus) — these dashboards are the at-a-glance views.

They are already created in the running OpenObserve (via the API). The JSON here in
`dashboards/` is the version-controlled copy so they can be re-imported / recreated.

| File | Dashboard | Panels |
|---|---|---|
| `dashboards/latency.json` | **HMO — Latency** | PersonaPlex first-response (ms), VC inference avg (ms), session duration by session, span latency by operation, GPU utilization %, GPU memory (MiB) |
| `dashboards/tracing.json` | **HMO — Distributed Tracing** | spans over time by service, span count by service, avg + max span duration by operation, slowest spans table |
| `dashboards/session-logs.json` | **HMO — Session Logs** | log volume by severity, logs by service, logs by study/user session, warnings/errors by service, errors table, recent logs table |

## View them

Open `https://<host>:5001/logs` → **Dashboards** → the three `HMO — …` dashboards.
Set the time range (top-right) to cover your data (they default to **Last 24 hours**).

## Recreate / import

The dashboards live in OpenObserve's data dir (`$STUDY_DATA_ROOT/observability`, on the
mounted volume) so they survive restarts. To recreate them elsewhere, POST each file to
the API (UI *Import* also works):

```bash
O2=https://<host>:5001/logs; A=admin@example.com:ChangeMe123
for f in infra/observability/dashboards/*.json; do
  curl -sk -u "$A" -H 'Content-Type: application/json' \
    -X POST "$O2/api/default/dashboards?folder=default" --data @"$f" -o /dev/null -w "$f -> %{http_code}\n"
done
```

## Notes

- **Panels use SQL, not PromQL** — OpenObserve's PromQL engine 500s on these OTLP
  metrics on this build (`v0.14.4`), and SQL is more flexible anyway.
- **Metric histograms are cumulative** (`_sum`/`_count`/…), so latency is read from the
  **traces** instead: `study_first_response_ms` (PersonaPlex) and
  `study_vc_inference_avg_ms` (VC) are per-span attributes — cast to DOUBLE in SQL since
  span attributes are stored as strings. Trace `duration` is in **microseconds**.
- If you change metric/attribute names in the code, update the panel SQL here to match.
