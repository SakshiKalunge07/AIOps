# **Model Architecture and Selection**

This document provides an in-depth look at the models powering PulseWatch — focusing on the design choices, reasoning behind model selection, and implementation details.  
PulseWatch uses two key models:
1. **LSTM Autoencoder (LSTMAE)** for unsupervised anomaly detection  
2. **Facebook Prophet** for univariate time-series forecasting and trend analysis  

---

## 🧠 1. LSTM Autoencoder (LSTMAE)

### **Why LSTM Autoencoder?**

The **LSTM Autoencoder** was chosen as the core anomaly detection model for several technical reasons:

- **Temporal Context Awareness:**  
  LSTM networks are designed to capture time dependencies and sequential correlations — critical for system metrics like CPU or latency that evolve over time.

- **Unsupervised Learning Capability:**  
  PulseWatch does not rely on labeled anomaly data. The autoencoder learns to reconstruct “normal” metric patterns and flags deviations as anomalies automatically.

- **Multivariate Support:**  
  Handles multiple correlated metrics (CPU, memory, latency) simultaneously, learning joint temporal dependencies across them.

- **Adaptive Thresholding:**  
  The model’s reconstruction error can be used as a dynamic anomaly score rather than relying on static thresholds.

---

### **Why Not a Simple RNN?**

We initially experimented with a **vanilla RNN** architecture for sequence modeling. However:

- **Gradient Instability:** RNNs suffer from vanishing gradients for long sequences (50+ timesteps).  
- **Poor Long-Term Memory:** They fail to capture long-term dependencies between spikes and subsequent recoveries.  
- **Lower Reconstruction Accuracy:** In reconstruction tasks, RNNs tend to overfit on recent samples, leading to unstable anomaly boundaries.

LSTMs (Long Short-Term Memory units) solved these issues with **gating mechanisms** (input, forget, and output gates), improving both reconstruction accuracy and temporal learning stability.

---

### **Why Not Isolation Forest?**

We also tested **Isolation Forests**, a popular unsupervised anomaly detection method for tabular data.  
While it performed decently on static metrics, it was **not well-suited for streaming time-series data** for the following reasons:

- **No Temporal Awareness:**  
  Isolation Forests treat each observation independently. They cannot capture temporal trends or patterns leading up to anomalies.  

- **No Forecasting Capability:**  
  They cannot predict upcoming anomalies or detect seasonality-related deviations.

- **Model Drift Issues:**  
  Needs frequent retraining as data distribution changes — not practical for real-time pipelines.

For PulseWatch’s goal of **continuous time-series anomaly detection**, LSTMAE provided a far more robust and adaptive approach.

---

### **LSTM Autoencoder Architecture**

```bash

Input Sequence (Timestamps × Metrics)
                │
                ▼
LSTM Encoder (Hidden Size = 32)
                │
                ▼
Latent Representation (Bottleneck)
                │
                ▼
LSTM Decoder (Reconstructs Input)
                │
                ▼
Reconstructed Sequence → Reconstruction Error → Anomaly Score

```

---

### **Training Workflow**

1. Collect clean metric data from Prometheus (CPU, memory, latency).  
2. Normalize each feature using MinMax scaling.  
3. Segment into overlapping sequences of `sequence_length = 50`.  
4. Train the LSTM Autoencoder for 10 epochs with MSE loss.  
5. Save the model to `models/lstm_model.pt`.  
6. Use reconstruction error to flag anomalies during inference.  

---

### **Hyperparameters (Configurable in `user_config.yml`)**

| Parameter | Description | Default |
|------------|--------------|----------|
| `sequence_length` | Length of input window | 50 |
| `hidden_size` | Number of hidden units | 32 |
| `epochs` | Number of training epochs | 10 |
| `reconstruction_error_limit` | Threshold for anomaly detection | 0.025 |
| `retrain_interval` | Hours between automatic retraining | 24 |

---

## 📈 2. Prophet Model (Forecasting Layer)

### **Why Prophet?**

The **Prophet** model (developed by Facebook/Meta) is used for **forecasting future metric behavior** and **detecting trend-based anomalies**.  
It was chosen because:

- **Robust to Seasonality and Trend Shifts:**  
  System metrics often follow daily or weekly cycles — Prophet handles that automatically.

- **Quick Training:**  
  Lightweight and retrains efficiently on new metric data streams.

- **Easy Interpretability:**  
  Produces intuitive trend and seasonal components that can be visualized on the dashboard.

- **Flexible Forecast Horizon:**  
  Can project metric behavior minutes or hours ahead, useful for capacity planning and early risk detection.

---

### **Why Not Deep Learning for Forecasting?**

We deliberately chose Prophet over deep forecasting models (LSTM Forecasting or Transformer-based models) because:

- **Operational Simplicity:**  
  Prophet runs fast and retrains without GPUs — ideal for edge and production systems.

- **Explainability:**  
  Deep models offer better accuracy but limited interpretability — a key drawback for DevOps environments where clarity matters.

- **Stable Predictions:**  
  Prophet avoids overfitting on short or noisy data windows, ensuring more reliable predictions in real-world conditions.

---

### **Prophet Pipeline**

1. Ingest recent metric data as a Pandas DataFrame (`timestamp`, `value`).  
2. Fit a Prophet model on each metric (CPU, memory, latency).  
3. Generate forecasts for the next `forecast_horizon` minutes.  
4. Compare actual vs predicted values — deviations beyond `±3σ` flagged as anomalies.  

---

### **Configurable Parameters**

| Parameter | Description | Default |
|------------|--------------|----------|
| `forecast_horizon` | Minutes to forecast ahead | 60 |
| `retrain_interval` | Frequency (in hours) to refresh the Prophet model | 12 |
| `changepoint_prior_scale` | Controls trend flexibility | 0.1 |
| `seasonality_mode` | Additive or multiplicative | additive |

---

## 🧪 Model Comparison Summary

| Model | Temporal Awareness | Forecasting | Unsupervised | Explainable | Reason for Rejection |
|--------|---------------------|-------------|--------------|--------------|----------------------|
| **RNN** | ✅ Partial | ⚠️ Limited | ✅ | ❌ | Poor long-term memory |
| **Isolation Forest** | ❌ None | ❌ None | ✅ | ⚠️ Moderate | No time dependency support |
| **LSTM Autoencoder** | ✅✅ Excellent | ⚠️ Indirect | ✅ | ⚠️ Limited | Selected for sequence learning |
| **Prophet** | ⚠️ Moderate | ✅✅ Excellent | ⚠️ Needs retraining | ✅✅ Very High | Selected for trend forecasting |

---

## ⚙️ Model Interaction Flow

``` bash
Prometheus Metrics
        ↓
Preprocessing (DataFrame)
        ↓
[LSTM Autoencoder] → Detect anomalies (real-time)
        ↓
[Prophet Model] → Forecast future values and validate anomalies
        ↓
Streamlit Dashboard → Display Live + Forecast Views

```
## Notes

- LSTM Autoencoder is **stateful** — retraining frequency depends on system drift.  
- Prophet can be **run in parallel** with the LSTMAE model for faster feedback loops.  
- For production, LSTMAE models are persisted locally under `models/` and loaded at API startup.  
- When running on long time windows, ensure **sequence padding** and **data normalization consistency**.

---

## Future Model Improvements

- Replace LSTMAE with **Temporal Convolutional Networks (TCN)** for faster inference.  
- Experiment with **Transformer-based architectures** (Informer or TimesNet).  
- Introduce **AutoML-based threshold tuning** for dynamic anomaly boundary calibration.  
- Support **multivariate Prophet** via additive regressors for correlated metric prediction.  