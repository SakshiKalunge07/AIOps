import time
from datetime import datetime
import yaml
from model_handler import train_lstm_from_df, predict_prophet_from_df, predict_lstm_from_df
import pandas as pd
import os
import numpy as np
# from test import inject_spike_anomalies
# from app.prometheus import fetch_and_merge_all_metrics


def load_config(path="user_config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

PREDICT_INTERVAL = CONFIG["lstm_model"]["prediction_horizon"] * 60
RETRAIN_INTERVAL = CONFIG["lstm_model"]["retrain_interval"] * 3600
PROPHET_INTERVAL = CONFIG["prophet_model"]["retrain_interval"] * 3600

last_retrain_time = 0
last_prophet_time = 0

print("Starting PulseWatch AIOps Service...")
print(f"Prometheus URL : {CONFIG['prometheus']['url']}")
print(f"LSTM Prediction Interval (minutes): {PREDICT_INTERVAL / 60}")
print(f"LSTM Retrain Interval (hours): {RETRAIN_INTERVAL / 3600}")
print(f"Prophet Retrain Interval (hours): {PROPHET_INTERVAL / 3600}")

# df = fetch_and_merge_all_metrics()
df = pd.read_csv("data/merged_metrics.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

train_lstm_from_df(df)
print("Initial LSTM model training completed.")

while True:
    try:

        df = pd.read_csv("data/merged_metrics.csv")
        if df is None or df.empty:
            print("No data fetched from Prometheus, retrying in 30 seconds...")
            time.sleep(30)
            continue
        else:
            print(f"Fetched {len(df)} records from Prometheus at {datetime.now()}")

            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

            lstm_result = predict_lstm_from_df(df)
            if isinstance(lstm_result, tuple):
                lstm_result = lstm_result[0]

            lstm_anomalies = lstm_result[lstm_result["anomaly"]]

            print(f"LSTM Anomalies detected: {len(lstm_anomalies)}")

            current_time = time.time()

            if current_time - last_prophet_time > PROPHET_INTERVAL:
                prophet_result = predict_prophet_from_df(df)
                if isinstance(prophet_result, tuple):
                    prophet_result = prophet_result[0]

                total_prophet_anomalies = 0
                if isinstance(prophet_result, dict):
                    for res in prophet_result.values():
                        if isinstance(res, pd.DataFrame) and "anomaly" in res.columns:
                            total_prophet_anomalies += len(res[res["anomaly"]])
                else:
                    print(f"[WARN] Unexpected Prophet result type: {type(prophet_result)}")

                print(f"Prophet Anomalies detected: {total_prophet_anomalies}")
                last_prophet_time = current_time

            if current_time - last_retrain_time > RETRAIN_INTERVAL:
                train_lstm_from_df(df)
                print("LSTM model retrained.")
                last_retrain_time = current_time

            if not lstm_anomalies.empty:
                cols = ['timestamp'] + [col for col in df.columns if col not in ['timestamp', 'ds']]
                print(f"[ALERT] LSTM Anomalies detected at:\n{lstm_anomalies[cols].tail(5)}")

            print(f"Sleeping for {PREDICT_INTERVAL / 60} minutes...\n")
            time.sleep(PREDICT_INTERVAL)

    except Exception as e:
        print(f"Error occurred: {e}. Retrying in 60 seconds...")
        time.sleep(60)

    except KeyboardInterrupt:
        print("PulseWatch service interrupted by user. Exiting...")
        break
