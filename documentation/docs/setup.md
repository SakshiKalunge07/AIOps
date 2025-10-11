# Setup Guide

## Prerequisites

- Python 3.9+
- Prometheus installed locally or remotely
- Node Exporter / Windows Exporter running
- (Optional) Docker & Kubernetes for deployment

---

## 1️⃣ Clone Repository
```bash
git clone https://github.com/SakshiKalunge07/AIOps.git
cd sakshikalunge07-aiops/PulseWatch-backend
```
## 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 3️⃣ Configure Prometheus
Edit prometheus.yml as per your system:
```yaml
scrape_configs:
  - job_name: "linux_exporter"
    static_configs:
      - targets: ["localhost:9100"]
```

## 4️⃣ Run FastAPI Backend
```bash
python -m app.main
```
Your API runs at: http://127.0.0.1:8000

## 5️⃣ Run Dashboard
```bash
streamlit run dashboard.py
```
Access at: http://localhost:8501

---
