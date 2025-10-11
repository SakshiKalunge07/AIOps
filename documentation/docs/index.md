# 🧠 PulseWatch – AIOps Platform

**PulseWatch** is a next-generation **AIOps (Artificial Intelligence for IT Operations)** platform that automates the detection of anomalies, forecasts system behavior, and visualizes real-time metrics through intelligent dashboards.  

By combining **machine learning**, **Prometheus-based metric ingestion**, and **FastAPI services**, it enables **proactive IT operations** and helps teams maintain reliability, stability, and performance at scale.  

---

## 🚀 Key Highlights

- Real-time metric monitoring from **Prometheus**
- Anomaly detection using **LSTM Autoencoders**
- Predictive forecasting with **Facebook Prophet**
- Modular backend built on **FastAPI**
- Live visualization via **Streamlit Dashboards**
- Automated model retraining and configurable thresholds

---

## 📂 Repository Overview

Directory structure:
``` bash

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
        │   ├── __init__.py
        │   ├── config.py
        │   ├── main.py
        │   └── prometheus.py
        ├── data/
        │   └── merged_metrics.csv
        ├── model/
        │   ├── anamoly_detection.py
        │   ├── prophet_model.py
        │   └── scaler.gz
        └── sample_data/
            ├── multivariant_data.csv
            └── univariant_data.csv
    
   
```
---

## 💡 Vision

PulseWatch bridges the gap between **data monitoring** and **intelligent automation**, turning raw infrastructure data into **actionable, predictive insights**.  
It’s designed to make IT systems smarter, more autonomous, and self-healing.



