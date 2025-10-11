# **Configuration**

All runtime behavior for PulseWatch is controlled via the `user_config.yml` file located at the project root.  
This file defines endpoints, model parameters, metric queries, and other operational settings used across both the backend and dashboard.

---

## ⚙️ Core Sections

| Section | Description |
|----------|-------------|
| `app` | General application metadata like name and version |
| `prometheus` | Base endpoint for Prometheus queries (usually `/api/v1/query_range`) |
| `metrics` | PromQL queries for CPU, Memory, and Latency metrics |
| `lstm_model` | Configuration for the LSTM Autoencoder used in anomaly detection |
| `prophet_model` | Forecasting parameters for the Prophet model |
| `dashboard` | Dashboard behavior, refresh intervals, and display settings |

---

## 🧩 Detailed Breakdown

### **1. App Section**
Defines global identifiers for the project.  
Used primarily for logging and metadata display.

```yaml
app:
  name: "PulseWatch"
  version: "1.0.0"
  mode: "production"
```

### **2. Prometheus**
Specifies the endpoint for querying system metrics.
This must point to your active Prometheus server.
```yaml
prometheus:
  url: "http://localhost:9090/api/v1/query_range"
  step: "30s"        # Query step size (granularity)
  window: "1h"       # Time range for each data pull
```

### **3. Metrics**
Each metric is defined as a PromQL query.
You can extend or modify these queries based on your system exporters.
```yaml
metrics:
  cpu: "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode='idle'}[1m])) * 100)"
  memory: "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100"
  latency: "rate(node_network_transmit_errs_total[1m])"
```

### **4. LSTM Model**
Controls how the LSTM Autoencoder behaves during anomaly detection and retraining.
Tuning these parameters directly affects sensitivity and performance.
```yaml
lstm_model:
  enabled: true
  sequence_length: 50        # Number of past data points per training sample
  hidden_size: 32            # Size of hidden layer in the LSTM
  epochs: 10                 # Number of training iterations
  retrain_interval: 24       # Retrain model every 24 hours
  reconstruction_error_limit: 0.025  # Threshold for marking anomalies
  save_path: "models/lstm_model.pt"
```

### **5. Prophet Model**
Defines settings for the forecasting module.
Used for trend prediction and anomaly correlation with future patterns.
```yaml
prophet_model:
  enabled: true
  forecast_horizon: 60       # Minutes into the future to forecast
  retrain_interval: 12       # Retrain every 12 hours
  changepoint_prior_scale: 0.1
  seasonality_mode: "additive"
```


### **6. Dashboard**
Defines parameters controlling refresh intervals, display limits, and user-triggered actions.
```yaml
dashboard:
  refresh_interval: 5        # Seconds between metric updates
  show_predictions: true
  max_points: 500
  retrain_button_enabled: true
```
---

### **Notes**
- Threshold tuning (reconstruction_error_limit) directly impacts false positives.
- Retraining intervals should be based on the data drift rate and metric volatility.
- Use shorter Prometheus query steps (15s–30s) for more granular anomaly detection.
- Always keep save_path within a mounted or persistent volume when running in containers.