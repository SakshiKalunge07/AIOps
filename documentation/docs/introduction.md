# **Introduction to PulseWatch - Real time AIOps anamoly detection**

Modern IT systems generate massive volumes of operational data. Traditional monitoring tools can detect threshold breaches but struggle to understand *context* and *predict future failures*.  
**PulseWatch AIOps** solves this gap by applying artificial intelligence to operations — collecting metrics, detecting anomalies, and forecasting performance degradation before it impacts users.

---

## 🎯 Objectives

- Automate system health monitoring through intelligent models  
- Predict anomalies and performance issues before they occur  
- Continuously retrain models to adapt to evolving workloads  
- Provide unified visibility through live dashboards  
- Enable proactive incident prevention and capacity planning  

---

## 🧩 Core Technologies

| Component | Technology | Purpose |
|------------|-------------|----------|
| Backend | **FastAPI** | REST API service layer |
| Dashboard | **Streamlit** | Visualization and interaction |
| Metric Collection | **Prometheus** | Real-time metric scraping |
| ML Models | **PyTorch LSTM Autoencoder**, **Facebook Prophet** | Anomaly detection & forecasting |
| Data Layer | **Pandas, NumPy** | Processing and feature engineering |

---

## 🏗️ System Overview

PulseWatch operates as a closed-loop AIOps pipeline:

1. **Collect metrics** from Prometheus exporters  
2. **Aggregate & preprocess** data into unified time-series  
3. **Detect anomalies** using LSTM Autoencoders  
4. **Forecast future states** using Prophet models  
5. **Visualize & alert** through Streamlit dashboards  
6. **Retrain models** periodically for continual learning  

