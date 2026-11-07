---
icon: lucide/chart-no-axes-column
---

# Telemetry

`qseek search` and `qseek continue` send anonymous usage statistics to [PostHog](https://posthog.com) (EU cloud). We use them to learn how large the searches are, on which hardware Qseek runs, and to report the use of Qseek to our funders. Using Qseek as a Python library sends nothing.

Qseek shows a notice before the first search sends anything.

## Turn it off

```bash title="Turn telemetry off"
qseek telemetry off
```

Setting `QSEEK_TELEMETRY=0` or `DO_NOT_TRACK=1` turns it off as well. Qseek sends nothing in CI (`CI` is set) and under pytest. `qseek telemetry` shows the current setting and your install ID.

## What Qseek sends

Each search sends two events. Failed or interrupted searches send no `run_finished` event and no error reports.

`run_started`, after the search is prepared:

| Field | Example |
|---|---|
| `run_id` | Random ID of the run, the same when you continue it |
| `continued` | `true` for `qseek continue` from a saved state |
| `qseek_version`, `python_version`, `torch_version`, `seisbench_version` | `2026.11.0`, `3.13.5`, ... |
| `os`, `arch`, `cpu_count` | `Linux`, `x86_64`, `64` |
| `gpu` | `cuda`, `mps` or `none` |
| `n_stations`, `n_networks` | `48`, `3` |
| `n_nodes` | Number of nodes in the octree, `12000` |
| `ray_tracers` | Types of the ray tracers, `["CakeTracer", "FastMarchingTracer"]` |
| `station_corrections` | Type of the station corrections, `SimpleCorrections`, or `null` without corrections |
| `octree_lat`, `octree_lon` | Center of the search volume, rounded to 0.001°. Left out when the center is at 0°, 0°. |

`run_finished`, after the search completed:

| Field | Example |
|---|---|
| `run_id` | Same as in `run_started` |
| `wall_time_seconds` | Duration of the search, `86400` |
| `processed_seconds` | Duration of the processed waveforms, `31536000` |
| `station_days` | `processed_seconds` × `n_stations` in days, `17520` |
| `n_detections` | `1520` |

Every event carries a random install ID, stored with the setting in `~/.config/qseek/telemetry.json`. PostHog derives a coarse location (country and city) from your IP address and does not store the address. Qseek sends no station codes, waveforms, detections, file paths, host or user names.
