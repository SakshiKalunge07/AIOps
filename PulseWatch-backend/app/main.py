import sys
import os
import time
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.prometheus import fetch_and_merge_all_metrics
from model_handler import predict_lstm_from_df, train_lstm_from_df 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

PREDICT_INTERVAL = 5

app = FastAPI(title="PulseWatch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def get_short_live_data():
    try:
        df = fetch_and_merge_all_metrics()
        if df is None or df.empty:
            return {"error": "No data fetched from Prometheus"}

        if "timestamp" not in df.columns:
            df = df.reset_index()

        df_live = df.tail(20).copy()

        try:
            result = predict_lstm_from_df(df_live)
            if isinstance(result, tuple) and len(result) == 3:
                lstm_result, _, _ = result
            else:
                lstm_result = result
        except Exception as e:
            return {"error": f"LSTM prediction failed: {e}"}

        if "anomaly" not in lstm_result.columns:
            lstm_result["anomaly"] = 0 

        lstm_anomalies = lstm_result[lstm_result["anomaly"] == 1]

        return {
            "metrics": df_live.to_dict(orient="records"),
            "anomalies": lstm_anomalies["timestamp"].astype(str).tolist() if "timestamp" in lstm_anomalies.columns else [],
            "anomaly_count": len(lstm_anomalies)
        }

    except Exception as e:
        return {"error": str(e)}

def get_long_live_data():
    try:
        df = fetch_and_merge_all_metrics()
        if df is None or df.empty:
            return {"error": "No data fetched from Prometheus"}

        if "timestamp" not in df.columns:
            df = df.reset_index()

        df_spike = df.copy()

        try:
            result = predict_lstm_from_df(df_spike)
            if isinstance(result, tuple) and len(result) == 3:
                lstm_result, _, _ = result
            else:
                lstm_result = result
        except Exception as e:
            return {"error": f"LSTM prediction failed: {e}"}

        if "anomaly" not in lstm_result.columns:
            lstm_result["anomaly"] = 0

        lstm_anomalies = lstm_result[lstm_result["anomaly"] == 1]

        return {
            "metrics": df_spike.to_dict(orient="records"),
            "anomalies": lstm_anomalies["timestamp"].astype(str).tolist() if "timestamp" in lstm_anomalies.columns else [],
            "anomaly_count": len(lstm_anomalies)
        }

    except Exception as e:
        return {"error": str(e)}

def inject_spike_anomalies(df, columns=None, n_anomalies=6, magnitude=10):
    """Inject spikes for testing only"""
    df_anom = df.copy()
    if columns is None:
        columns = [c for c in df.columns if c != "timestamp"]
    
    for _ in range(n_anomalies):
        col = np.random.choice(columns)
        idx = np.random.randint(0, len(df))
        df_anom.loc[df_anom.index[idx], col] += magnitude * df[col].std()
    return df_anom

def get_test_data():
    try:
        file_path = os.path.join(BASE_DIR, "data", "merged_metrics.csv")
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.tail(50).reset_index(drop=True)
        df_test = inject_spike_anomalies(df, n_anomalies=2, magnitude=10)
        result = predict_lstm_from_df(df_test)
        if isinstance(result, tuple) and len(result) == 3:
            lstm_result, _, _ = result
        else:
            lstm_result = result
        if "anomaly" not in lstm_result.columns:
            lstm_result["anomaly"] = 0
        detected_anomalies = lstm_result[lstm_result["anomaly"] == 1]
        return {
            "metrics": df_test.to_dict(orient="records"),
            "anomalies": detected_anomalies["timestamp"].astype(str).tolist() 
                         if "timestamp" in detected_anomalies.columns else [],
            "anomaly_count": len(detected_anomalies)
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/train")
async def train_model():
    try:
        df = fetch_and_merge_all_metrics()
        if df is None or df.empty:
            return {"error": "No data fetched for training"}
        message = train_lstm_from_df(df)
        return {"status": "success", "message": message}
    except Exception as e:
        return {"error": f"Training failed: {e}"}

@app.get("/")
async def welcome():
    return {'msg': 'Welcome To PulseWatch anomaly detection!'}

@app.get("/metrics/live-short")
async def live_metrics():
    return get_short_live_data()

@app.get("/metrics/live-long")
async def spike_metrics():
    return get_long_live_data()

@app.get("/metrics/test")
async def test_metrics():
    return get_test_data()

if __name__ == "__main__":
    import uvicorn
    print("Starting PulseWatch API on http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)