# **System Architecture**

PulseWatch follows a modular **service-oriented architecture** designed for scalability and maintainability.

---

## 🗂️ High-Level Architecture

``` bash
┌────────────────────────────────────┐
│ Prometheus Exporters               |
│ (Linux, Windows, macOS metrics)    |
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│ Prometheus Server                  │
│ Collects and exposes metrics via   │
│ /api/v1/query_range endpoint       │
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│ FastAPI Backend (app/)             │
│ - /metrics/live-short              │
│ - /metrics/live-long               │
│ - /metrics/test                    │
│ - /train                           │
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│ Model Handler (LSTM, Prophet)      │
│ - Anomaly Detection (LSTM AE)      │
│ - Forecasting (Prophet)            │
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│ Streamlit Dashboard (Frontend).    │
│ - Live view of metrics & anomalies │
│ - Trigger model retraining         │
└────────────────────────────────────┘

```

---

## 🔁 Data Flow

1. **Prometheus Integration** → Metrics fetched through API queries  
2. **Data Aggregation** → Merged via Pandas and converted to DataFrame  
3. **Inference Layer** → LSTM Autoencoder detects anomalies  
4. **Forecast Layer** → Prophet forecasts future trends  
5. **Visualization Layer** → Real-time charts in Streamlit  

---

## ⚙️ Scalability

- Modular microservice structure  
- Configurable retrain and prediction intervals  
- Supports distributed deployment with Docker/Kubernetes  