import pandas as pd
from prophet import Prophet


def train_and_detect_prophet_df(df, metric_column="value", forecast_horizon=60):
    temp = df[["ds", metric_column]].dropna().rename(columns={"ds": "ds", metric_column: "y"})
    temp["ds"] = pd.to_datetime(temp["ds"])

    model = Prophet(interval_width=0.95, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(temp[["ds", "y"]])

    future = model.make_future_dataframe(periods=forecast_horizon, freq="min")
    forecast = model.predict(future)

    merged = pd.merge(temp, forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left")
    merged["anomaly"] = (merged["y"] > merged["yhat_upper"]) | (merged["y"] < merged["yhat_lower"])
    merged.rename(columns={"y": "actual"}, inplace=True)
    return merged


def detect_multimetric_anomalies_df(df, forecast_horizon=60):
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    results = {}
    metric_cols = [col for col in df.columns if col not in ["timestamp", "ds"]]

    for col in metric_cols:
        print(f"Running Prophet anomaly detection for: {col}")
        if df[col].isna().sum() > 0 or len(df[col]) < 10:
            print(f"Skipping {col}, insufficient data.")
            continue
        results[col] = train_and_detect_prophet_df(df, metric_column=col, forecast_horizon=forecast_horizon)

    return results
