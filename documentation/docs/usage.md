# **Usage Guide**

Once running, PulseWatch provides three operational modes through the Streamlit dashboard:

| Mode | Description |
|------|--------------|
| **Live Metrics** | Displays real-time metrics fetched from Prometheus |
| **Long-Term Prediction** | Shows forecasted behavior and historical anomalies |
| **Test Mode** | Runs local anomaly detection using sample data with synthetic spikes |

---

## 🧩 Dashboard Controls

- **Metrics Type Selector:** Choose between live, long-term, or test datasets  
- **Refresh Interval:** Adjust real-time polling frequency  
- **Train Model Button:** Trigger backend model retraining  
- **Anomaly Count Display:** View total anomalies detected per session  

---

## 📈 Example Workflow

1. Launch backend and Prometheus  
2. Start Streamlit dashboard  
3. Select “Live Metrics”  
4. Observe anomaly detection results in real time  
5. Switch to “Test on Sample Data” to simulate spikes  
6. Trigger retraining when performance drifts