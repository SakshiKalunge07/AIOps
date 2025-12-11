# PulseWatch: AIOps Anomaly Detection System

**PulseWatch** is a next-generation **AIOps (Artificial Intelligence for IT Operations)** platform that automates the detection of anomalies, forecasts system behavior, and visualizes real-time metrics through intelligent dashboards. By combining **machine learning**, **Prometheus-based metric ingestion**, and **FastAPI services**, it enables **proactive IT operations** and helps teams maintain reliability, stability, and performance at scale.

PulseWatch bridges the gap between **data monitoring** and **intelligent automation**, turning raw infrastructure data into **actionable, predictive insights**. It’s designed to make IT systems smarter, more autonomous, and self-healing.

## 🚀 Key Highlights
- Real-time metric monitoring from **Prometheus**
- Anomaly detection using **LSTM Autoencoders**
- Predictive forecasting with **Facebook Prophet**
- Modular backend built on **FastAPI**
- Live visualization via **Streamlit Dashboards**
- Automated model retraining and configurable thresholds

## 🎯 Objectives
- Automate system health monitoring through intelligent models
- Predict anomalies and performance issues before they occur
- Continuously retrain models to adapt to evolving workloads
- Provide unified visibility through live dashboards
- Enable proactive incident prevention and capacity planning

## 🧩 Core Technologies
| Component | Technology | Purpose |
|------------|-------------|----------|
| Backend | **FastAPI** | REST API service layer |
| Dashboard | **Streamlit** | Visualization and interaction |
| Metric Collection | **Prometheus** | Real-time metric scraping |
| ML Models | **PyTorch LSTM Autoencoder**, **Facebook Prophet** | Anomaly detection & forecasting |
| Data Layer | **Pandas, NumPy** | Processing and feature engineering |

## 🏗️ System Overview
PulseWatch operates as a closed-loop AIOps pipeline:
1. **Collect metrics** from Prometheus exporters
2. **Aggregate & preprocess** data into unified time-series
3. **Detect anomalies** using LSTM Autoencoders
4. **Forecast future states** using Prophet models
5. **Visualize & alert** through Streamlit dashboards
6. **Retrain models** periodically for continual learning

<img width="486" height="350" alt="image" src="https://github.com/user-attachments/assets/f6a06581-dc17-4a46-a93d-e5fc2f1a6b99" />


## 📂 Repository Overview
Directory structure:
```
└── sakshikalunge07-aiops/
    ├── README.md
    └── PulseWatch-backend/
        ├── dashboard.py
        ├── model_handler.py
        ├── prometheus.yml
        ├── requirements.txt
        ├── run.py
        ├── test.py
        ├── user_config.yml
        ├── app/
        │ ├── __init__.py
        │ ├── config.py
        │ ├── main.py
        │ └── prometheus.py
        ├── data/
        │ └── merged_metrics.csv
        ├── model/
        │ ├── anamoly_detection.py
        │ ├── prophet_model.py
        │ └── scaler.gz
        └── sample_data/
            ├── multivariant_data.csv
            └── univariant_data.csv
```

## Setup Guide
### Prerequisites
- Python 3.9+
- Prometheus installed locally or remotely
- Node Exporter / Windows Exporter running
- (Optional) Docker & Kubernetes for deployment

### 1️⃣ Clone Repository
```bash
git clone https://github.com/SakshiKalunge07/AIOps.git
cd sakshikalunge07-aiops/PulseWatch-backend
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Configure Prometheus
Edit `prometheus.yml` as per your system:
```yaml
scrape_configs:
  - job_name: "linux_exporter"
    static_configs:
      - targets: ["localhost:9100"]
```

### 4️⃣ Run FastAPI Backend
```bash
python -m app.main
```
Your API runs at: http://127.0.0.1:8000

### 5️⃣ Run Dashboard
```bash
streamlit run dashboard.py
```
Access at: http://localhost:8501

## Usage Guide
Once running, PulseWatch provides three operational modes through the Streamlit dashboard:

| Mode | Description |
|------|--------------|
| **Live Metrics** | Displays real-time metrics fetched from Prometheus |
| **Long-Term Prediction** | Shows forecasted behavior and historical anomalies |
| **Test Mode** | Runs local anomaly detection using sample data with synthetic spikes |

### Dashboard Controls
- **Metrics Type Selector:** Choose between live, long-term, or test datasets
- **Refresh Interval:** Adjust real-time polling frequency
- **Train Model Button:** Trigger backend model retraining
- **Anomaly Count Display:** View total anomalies detected per session

### Example Workflow
1. Launch backend and Prometheus
2. Start Streamlit dashboard
3. Select “Live Metrics”
4. Observe anomaly detection results in real time
5. Switch to “Test on Sample Data” to simulate spikes
6. Trigger retraining when performance drifts

## Configuration
All runtime behavior for PulseWatch is controlled via the `user_config.yml` file located at the project root. This file defines endpoints, model parameters, metric queries, and other operational settings used across both the backend and dashboard.

### ⚙️ Core Sections
| Section | Description |
|----------|-------------|
| `app` | General application metadata like name and version |
| `prometheus` | Base endpoint for Prometheus queries (usually `/api/v1/query_range`) |
| `metrics` | PromQL queries for CPU, Memory, and Latency metrics |
| `lstm_model` | Configuration for the LSTM Autoencoder used in anomaly detection |
| `prophet_model` | Forecasting parameters for the Prophet model |
| `dashboard` | Dashboard behavior, refresh intervals, and display settings |

### Detailed Breakdown
#### 1. App Section
Defines global identifiers for the project. Used primarily for logging and metadata display.
```yaml
app:
  name: "PulseWatch"
  version: "1.0.0"
  mode: "production"
```

#### 2. Prometheus
Specifies the endpoint for querying system metrics. This must point to your active Prometheus server.
```yaml
prometheus:
  url: "http://localhost:9090/api/v1/query_range"
  step: "30s" # Query step size (granularity)
  window: "1h" # Time range for each data pull
```

#### 3. Metrics
Each metric is defined as a PromQL query. You can extend or modify these queries based on your system exporters.
```yaml
metrics:
  cpu: "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode='idle'}[1m])) * 100)"
  memory: "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100"
  latency: "rate(node_network_transmit_errs_total[1m])"
```

#### 4. LSTM Model
Controls how the LSTM Autoencoder behaves during anomaly detection and retraining. Tuning these parameters directly affects sensitivity and performance.
```yaml
lstm_model:
  enabled: true
  sequence_length: 50 # Number of past data points per training sample
  hidden_size: 32 # Size of hidden layer in the LSTM
  epochs: 10 # Number of training iterations
  retrain_interval: 24 # Retrain model every 24 hours
  reconstruction_error_limit: 0.025 # Threshold for marking anomalies
  save_path: "models/lstm_model.pt"
```

#### 5. Prophet Model
Defines settings for the forecasting module. Used for trend prediction and anomaly correlation with future patterns.
```yaml
prophet_model:
  enabled: true
  forecast_horizon: 60 # Minutes into the future to forecast
  retrain_interval: 12 # Retrain every 12 hours
  changepoint_prior_scale: 0.1
  seasonality_mode: "additive"
```

#### 6. Dashboard
Defines parameters controlling refresh intervals, display limits, and user-triggered actions.
```yaml
dashboard:
  refresh_interval: 5 # Seconds between metric updates
  show_predictions: true
  max_points: 500
  retrain_button_enabled: true
```

### Notes
- Threshold tuning (reconstruction_error_limit) directly impacts false positives.
- Retraining intervals should be based on the data drift rate and metric volatility.
- Use shorter Prometheus query steps (15s–30s) for more granular anomaly detection.
- Always keep save_path within a mounted or persistent volume when running in containers.

## Modules
### Dashboard (Streamlit)
The `dashboard.py` visualizes live metrics and anomalies in real-time.
- Features: Choose between **Live**, **Long-Term**, and **Test** modes. Start or stop **Model Training** directly from UI. Auto-refresh chart every N seconds.
- Endpoints used: `/metrics/live-short`, `/metrics/live-long`, `/metrics/test`

### Backend API (FastAPI)
Core Routes:
| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/` | GET | Welcome message |
| `/metrics/live-short` | GET | Fetch short-term live metrics |
| `/metrics/live-long` | GET | Long-term predictions |
| `/metrics/test` | GET | Test data with artificial spikes |
| `/train` | POST | Retrain LSTM model |

Implementation in `app/main.py`:
- `get_short_live_data()` → Fetch 20 most recent data points
- `get_long_live_data()` → Full window analysis
- `get_test_data()` → Inject anomalies and evaluate model

### Model Handler
**File:** `model_handler.py`  
The core inference and orchestration layer. Connects data ingestion, ML models (LSTM + Prophet), and backend API.
- Functions: `predict_lstm_from_df()`, `train_lstm_from_df()`, `predict_prophet_from_df()`
- Handles model retraining, persistence, and failure recovery.

### Model Architecture
PulseWatch uses:
1. **LSTM Autoencoder (LSTMAE)** for unsupervised anomaly detection: Captures temporal dependencies in multivariate metrics (CPU, memory, latency).
   - Why LSTMAE? Superior to RNN (gradient instability) or Isolation Forest (no temporal awareness).
   - Hyperparameters: Sequence length (30), hidden size (256), epochs (20), etc.
     <img width="500" height="700" alt="ChatGPT Image Oct 11, 2025 at 10_15_54 PM" src="https://github.com/user-attachments/assets/735fb40c-adc2-4423-aade-6471dc655644" />

2. **Facebook Prophet** for forecasting: Handles seasonality and trends for predictive anomaly detection.
   - Why Prophet? Robust, explainable, and efficient compared to deep forecasting models.

Model Interaction Flow:
```
Prometheus Metrics → Preprocessing → LSTMAE (Anomalies) → Prophet (Forecasts) → Dashboard
```

### Prometheus Integration
**File:** `app/prometheus.py`  
- Functions: `fetch_and_merge_all_metrics()` merges CPU, Memory, Latency into a DataFrame; `prometheus_to_dataframe()` converts JSON to pandas.
- Queries defined in `user_config.yml` with OS-specific variations (Windows/Linux/Darwin).

### Testing
**File:** `test.py`  
- Key Function: `inject_spike_anomalies()` adds random spikes to simulate anomalies.
- Evaluates LSTM and Prophet on modified data; visualizes with Matplotlib.

## Results and Insights
- **Detected anomalies:** CPU and memory spikes
- **Forecasts:** Predictive trends for latency and utilization
- **Visualization:** Real-time charts with red markers for anomalies

## Future Work
- Integrate database (PostgreSQL / InfluxDB) for persistent storage
- Containerize services (Docker + Kubernetes)
- Add auto-scaling ML model retraining pipeline
- Enhance dashboard with Grafana integration

## Contributors
- [**Sakshi Kalunge**](https://github.com/SakshiKalunge07)
- [**Anushree Upasham**](https://github.com/annuuxoxo)


For more details, refer to the full documentation in the `documentation/docs/` folder, built with MkDocs (config in `documentation/mkdocs.yml`).
