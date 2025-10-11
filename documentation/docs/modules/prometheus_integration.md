# Prometheus Integration

File: `app/prometheus.py`

### Functions
- `fetch_and_merge_all_metrics()` → Merges CPU, Memory, and Latency metrics into a single DataFrame.
- `prometheus_to_dataframe()` → Converts API JSON response to pandas DataFrame.

### Queries
Defined in `app/config.py` and `user_config.yml` under:
```yaml
metrics:
  cpu:
    query:
      Windows: "windows_cpu_time_total"
      Linux: "node_cpu_seconds_total{mode!='idle'}"
      Darwin: "node_cpu_seconds_total{mode!='idle'}"

  memory:
    query:
      Windows: "windows_memory_available_bytes"
      Linux: "node_memory_MemAvailable_bytes"
      Darwin: "node_memory_MemAvailable_bytes"

  latency:
    query:
      Windows: "windows_physical_disk_write_latency_seconds_total"
      Linux: "node_disk_io_time_seconds_total"
      Darwin: "node_disk_io_time_seconds_total"
```