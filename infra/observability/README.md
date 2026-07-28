# Observability dashboard

`dashboard.json` is a ready-to-import OpenObserve dashboard with latency + GPU panels
for the study stack. Traces and logs need **no** dashboard — explore them directly
(Traces / Logs menus). This is only for the **metrics** graphs.

## Import

1. Start the stack with observability on (see `docs/tracing.md`) and open the UI at
   `https://<host>:5001/logs`.
2. **Dashboards → Import** → upload `infra/observability/dashboard.json`
   (or paste its contents). Save.

That's it — no Grafana, no building an app. OpenObserve has the dashboard builder
built in; this file just pre-wires the panels.

## If a panel is empty

Metric/label names are sanitized by the OTLP→store path and can vary slightly by
OpenObserve version (dots→underscores; histograms get `_bucket`/`_sum`/`_count`). Open
**Metrics → Explore**, find the real stream name, and fix the PromQL in the panel.
The queries the dashboard uses:

| Panel | PromQL |
|---|---|
| VC inference p95 | `histogram_quantile(0.95, sum by (le, study_engine) (rate(vc_inference_ms_bucket[5m])))` |
| VC inference p50 | `histogram_quantile(0.50, sum by (le, study_engine) (rate(vc_inference_ms_bucket[5m])))` |
| PersonaPlex first-response p95 | `histogram_quantile(0.95, sum by (le, study_engine) (rate(personaplex_first_response_ms_bucket[5m])))` |
| Client network RTT p95 | `histogram_quantile(0.95, sum by (le) (rate(client_network_rtt_ms_bucket[5m])))` |
| Client connect p95 | `histogram_quantile(0.95, sum by (le) (rate(client_connect_ms_bucket[5m])))` |
| Client first-audio p95 (e2e) | `histogram_quantile(0.95, sum by (le) (rate(client_first_audio_ms_bucket[5m])))` |
| GPU utilization | `gpu_utilization` |
| GPU memory used | `gpu_memory_used_mib` |

Other metrics emitted (add panels as needed): `gpu_memory_total_mib`,
`gpu_temperature_c`, `gpu_power_w`. Latency histograms also expose `_sum` / `_count`
(e.g. average = `rate(vc_inference_ms_sum[5m]) / rate(vc_inference_ms_count[5m])`).

Tip: put a GPU panel next to the VC/PersonaPlex latency panels to see contention —
latency rising as `gpu_utilization` / `gpu_memory_used_mib` climb (e.g. when the
analysis batch runs on the shared GPU).
