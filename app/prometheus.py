import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional
from app.config import prometheus_url, metric_queries  # imported from config.py

rolling_window_min = 180  
step_second = "15s"       

def prometheus_to_dataframe(prometheus_data, column_name: str) -> Optional[pd.DataFrame]:    #helper function for converting individual metrics to dataframe
    if not prometheus_data:
        return None
    raw_data = prometheus_data[0].get("values", [])
    if not raw_data:
        return None
    df = pd.DataFrame(raw_data, columns=['timestamp_unix', column_name])
    df['timestamp'] = pd.to_datetime(df['timestamp_unix'], unit='s',utc=True)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df = df.drop(columns=['timestamp_unix']).set_index('timestamp')
    return df

def fetch_and_merge_all_metrics() -> Optional[pd.DataFrame]:
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=rolling_window_min)
    final_df = None

    for col_name, promql_query in metric_queries.items():
        if not promql_query:
            continue

        params = {
            "query": promql_query,
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": step_second
        }

        try:
            response = requests.get(prometheus_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success" or not data["data"]["result"]:
                continue

            current_df = prometheus_to_dataframe(data["data"]["result"], col_name)
            if current_df is not None:
                final_df = (
                    current_df if final_df is None
                    else final_df.merge(current_df, left_index=True, right_index=True, how="outer")
                )

        except Exception as e:
            print(f"Error fetching {col_name}: {e}")
            continue

    if final_df is not None:
        final_df.sort_index(inplace=True)
    return final_df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = fetch_and_merge_all_metrics()

    if df is not None and not df.empty:
        df.to_csv("data/merged_metrics.csv", index=True, index_label="timestamp")
        print("Metrics fetched successfully:\n", df.head())
    else:
        print("No data fetched from Prometheus.")
