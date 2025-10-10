import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import time

LIVE_URL = "http://localhost:8000/metrics/live-short"
LONG_TERM_URL = "http://localhost:8000/metrics/live-long"
TEST_URL = "http://localhost:8000/metrics/test"
TRAIN_URL = "http://localhost:8000/train" 
DEFAULT_REFRESH = 5

st.set_page_config(page_title="PulseWatch Dashboard", layout="wide")
st.title("PulseWatch AIOps Dashboard")

st.sidebar.header("Settings")
metric_type = st.sidebar.selectbox(
    "Select Metrics Type",
    ["Live Metrics", "Long-Term Prediction", "Test on Sample Data"]
)
refresh_sec = st.sidebar.number_input(
    "Refresh Interval (sec)", min_value=1, max_value=60, value=DEFAULT_REFRESH
)

train_triggered = st.sidebar.button("Train Model")
train_status_placeholder = st.sidebar.empty()

def fetch_data(url):
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if "error" in data:
            return None, [], 0
        df = pd.DataFrame(data["metrics"])
        anomalies = data.get("anomalies", [])
        anomaly_count = data.get("anomaly_count", 0)
        return df, anomalies, anomaly_count
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, [], 0

def trigger_training():
    try:
        resp = requests.post(TRAIN_URL, timeout=25)
        if resp.status_code == 200:
            return "Training completed successfully!"
        else:
            return f"Training failed: {resp.status_code}"
    except Exception as e:
        return f"Error during training: {e}"

def format_timestamps(ts_list):
    formatted = []
    for t in ts_list:
        try:
            formatted.append(datetime.fromisoformat(t).strftime("%H:%M:%S"))
        except Exception:
            formatted.append(str(t))
    return formatted

placeholder_graph = st.empty()
placeholder_info = st.empty()

if train_triggered:
    status_msg = trigger_training()
    train_status_placeholder.success(status_msg)

if metric_type == "Live Metrics":
    url = LIVE_URL
elif metric_type == "Test on Sample Data":
    url = TEST_URL
else:
    url = LONG_TERM_URL

df, anomalies, count = fetch_data(url)

if metric_type == "Test on Sample Data":
    st.markdown("### Test Endpoint Anomalies")
    if anomalies:
        st.write(pd.DataFrame({"Anomaly Timestamps": format_timestamps(anomalies)}))
    else:
        st.write("No anomalies detected.")

placeholder_info.markdown(
    f"**Anomalies Detected:** {count}  \n"
    f"**Timestamps:** {', '.join(format_timestamps(anomalies)) if anomalies else 'None'}"
)

if df is not None and not df.empty:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    chart_df = df[["cpu", "memory", "latency"]].copy()

    for ts in anomalies:
        if ts in chart_df.index:
            chart_df[["cpu", "memory", "latency"]] = chart_df[["cpu", "memory", "latency"]].astype(float)
            chart_df.loc[ts, ["cpu", "memory", "latency"]] *= 1.1

    placeholder_graph.line_chart(chart_df)

time.sleep(refresh_sec)
st.rerun()
