import pandas as pd
import numpy as np
from prophet import Prophet

def train_and_detect_prophet_df(df, metric_column="value", forecast_horizon=60):
    temp = df[["timestamp", metric_column]].dropna().rename(columns={"timestamp": "ds", metric_column: "y"})
    temp["ds"] = pd.to_datetime(temp["ds"])

    model = Prophet(
        interval_width=0.90,
        changepoint_prior_scale=0.3,
        seasonality_prior_scale=15.0,
        changepoint_range=0.9,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
    )
    model.add_seasonality(name='minute', period=60, fourier_order=5)
    model.add_seasonality(name='hour', period=3600, fourier_order=10)
    model.fit(temp[["ds", "y"]])

    future = model.make_future_dataframe(periods=forecast_horizon, freq="min", include_history=True)
    forecast = model.predict(future)

    merged = pd.merge(temp, forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left")
    merged["residual"] = merged["y"] - merged["yhat"]
    resid_std = np.std(merged["residual"])
    resid_mean = np.mean(merged["residual"])
    lower_bound = resid_mean - 3 * resid_std
    upper_bound = resid_mean + 3 * resid_std
    merged["anomaly"] = (merged["residual"] < lower_bound) | (merged["residual"] > upper_bound)
    merged.rename(columns={"y": "actual"}, inplace=True)
    return merged

def detect_multimetric_anomalies_df(df, forecast_horizon=60):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    results = {}
    metric_cols = [col for col in df.columns if col not in ["timestamp", "ds"]]

    for col in metric_cols:
        print(f"Running Prophet anomaly detection for: {col}")
        if df[col].isna().sum() > 0 or len(df[col]) < 30:
            print(f"Skipping {col}, insufficient data.")
            continue
        results[col] = train_and_detect_prophet_df(df, metric_column=col, forecast_horizon=forecast_horizon)

    return results