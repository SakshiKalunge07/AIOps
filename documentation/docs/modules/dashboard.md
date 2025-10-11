# Streamlit Dashboard

The `dashboard.py` visualizes live metrics and anomalies in real-time.

### Features
- Choose between **Live**, **Long-Term**, and **Test** modes.
- Start or stop **Model Training** directly from UI.
- Auto-refresh chart every N seconds.

### Code Reference
- **File:** `PulseWatch-backend/dashboard.py`
- **Endpoints used:**
  - `/metrics/live-short`
  - `/metrics/live-long`
  - `/metrics/test`