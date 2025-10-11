# **Model Handler**
**File:** `model_handler.py`

The `model_handler` module is the **core inference and orchestration layer** of PulseWatch.  
It connects the data ingestion pipeline, machine learning models (LSTM Autoencoder + Prophet), and the backend API endpoints served via **FastAPI**.

This module is responsible for:
- Running inference using pre-trained models  
- Managing model retraining cycles  
- Coordinating between Prometheus data fetchers and visualization endpoints  
- Handling failure recovery and model persistence  

---

## 🧩 Overview

The `model_handler.py` script acts as the **bridge** between live metric data and intelligent insights.

Internally, it wraps around three key functions:
1. `predict_lstm_from_df()` → Detects anomalies using LSTM Autoencoder  
2. `train_lstm_from_df()` → Retrains LSTM model on new metric data  
3. `predict_prophet_from_df()` → Forecasts future trends with Prophet  

These are invoked by **FastAPI routes** (`/metrics/live-short`, `/metrics/test`, `/train`) defined in the backend layer.

---

## ⚙️ File Responsibilities

| Function | Description |
|-----------|-------------|
| `predict_lstm_from_df(df)` | Runs LSTMAE model inference and returns anomaly-labeled data |
| `train_lstm_from_df(df)` | Trains and saves a new LSTM Autoencoder model using recent Prometheus metrics |
| `predict_prophet_from_df(df)` | Fits a Prophet model to the latest dataset and produces forecasts |
| `_calculate_reconstruction_error()` | Internal helper to compute model loss and identify spikes |
| `_save_model()` / `_load_model()` | Persist and reload PyTorch models across runs |

---

## 🧠 1. LSTM Autoencoder Integration

### **Prediction Flow**

```python
def predict_lstm_from_df(df):
    model = load_trained_lstm()
    reconstructed = model.predict(df)
    error = mean_squared_error(df, reconstructed)
    anomalies = error > threshold
    return anomalies, error
```

- Takes preprocessed metric DataFrame (cpu, memory, latency) as input.
- Loads the most recent trained model from models/lstm_model.pt.
- Performs reconstruction for each sequence of size sequence_length.
- Computes reconstruction error per timestep.
- Flags time points exceeding reconstruction_error_limit as anomalies.
- Returns anomaly timestamps and summary statistics to the API layer.

## **Retraining Flow**
```python
def train_lstm_from_df(df):
    model = LSTMAutoencoder(input_dim=df.shape[1])
    model.fit(df, epochs=EPOCHS)
    save_model(model)
    return "Model retrained successfully"
```

Retraining is triggered by:
- Manual dashboard request (Train Model button), or
- Scheduled retrain interval (retrain_interval hours, via background scheduler).
- The updated model is saved and automatically loaded on the next inference request.

## **Persistence & Recovery**
* Model is stored at models/lstm_model.pt
* On startup, the handler checks for the saved checkpoint
* If missing, it triggers a one-time training on available metric data
* During runtime, if inference fails, the handler falls back to the last valid model snapshot

---

### **📈 2. Prophet Forecast Integration**

**Prediction Flow**
```python
def predict_prophet_from_df(df):
    m = Prophet(seasonality_mode="additive")
    m.fit(df[['ds', 'y']])
    future = m.make_future_dataframe(periods=forecast_horizon, freq='min')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
```
1. Converts timestamp → Prophet-compatible ds, metric value → y.
2. Fits a new Prophet model on each metric column (CPU, memory, latency).
3. Generates forecast for the next N minutes (forecast_horizon).
4. Returns both predicted values and confidence intervals.
5. These forecasts are displayed on the dashboard’s “Prediction Mode” view, overlayed on the live metrics.

**Retraining and Scheduling**
Prophet models are lightweight — they retrain automatically on each data pull.
Retraining frequency is controlled by prophet_model.retrain_interval (hours) in user_config.yml.
This allows continuous adaptation to seasonal or cyclical trends (e.g., CPU load peaks at specific times of day).

## **🔁 3. Model Handler Workflow**
```python
            ┌───────────────────────┐
            │  Prometheus (metrics) │
            └───────────┬───────────┘
                        │
                        ▼
                ┌────────────────--┐
                │model_handler.py  │ 
                ├────────────────--┤
                │ - preprocess()   │
                │ - predict_lstm   │
                │ - predict_prophet│
                │ - train_lstm     │
                └────────┬─────────┘
                        │
     ┌──────────────────┴──────────────────┐
     ▼                                     ▼
 Streamlit Dashboard               FastAPI Endpoints
 (Visualizes anomalies)            (Serves /metrics/, /train/)
```

## **🧠 Model Interaction Details**

Shared Data Interface
All models use a common Pandas DataFrame interface:
``` bash
timestamp | cpu | memory | latency
```
1. This structure ensures consistency across inference and forecasting steps.
2. Anomaly Marking
3. LSTMAE returns an array of boolean anomaly flags per timestamp.
4. Prophet returns upper/lower prediction bounds; values exceeding them are marked anomalous.
5. These results are merged before being pushed to the dashboard layer.

## **Error Handling**
- Model Load Failures:
Fallback to train_lstm_from_df() for retraining.

- DataFrame Shape Mismatch:
Automatically trims columns or adjusts sequence length.

- Empty Data:
Gracefully exits with a “No metrics available” response.

## **Notes**
- The model handler does not hardcode thresholds — all are read dynamically from user_config.yml.
- For performance, Prophet runs asynchronously to avoid blocking the main loop.
- The handler’s modular design makes it easy to plug in alternate models (e.g., Transformers, TCNs) without touching the API layer.
- During testing, test.py calls predict_lstm_from_df() with synthetic spike data to validate end-to-end anomaly logic.

## **Inference Sequence**
# Dashboard triggers /metrics/live-short
```yaml
→ FastAPI calls predict_lstm_from_df()
→ Model handler fetches and normalizes data
→ LSTMAE detects anomalies
→ Prophet forecasts upcoming trends
→ Results merged and returned as JSON
→ Dashboard visualizes live anomalies
```

## **Future Enhancements**

1. Implement a Model Manager class to handle concurrent model lifecycles.
2. Add model performance metrics (precision, recall, F1) for periodic evaluation.
3. Support asynchronous training and job queuing for long-running retrains.
4. Introduce checkpoint versioning for rollback to previous model states.

__In short:__
__`model_handler.py` is the intelligent glue between your metric streams, your models, and your monitoring dashboard — ensuring that every prediction, anomaly, and trend stays in sync with reality.__
