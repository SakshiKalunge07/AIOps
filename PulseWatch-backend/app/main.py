from fastapi import FastAPI
from app.prometheus import fetch_and_merge_all_metrics
from model_handler import predict_lstm_from_df, predict_prophet_from_df

app = FastAPI(title="PulseWatch Anomaly Detection API")

@app.get("/predict")
def predict_anomaly():
    try:

        df = fetch_and_merge_all_metrics()

        if df is None or df.empty:
            return {"message": "No metrics found for the configured window."}

        lstm_result = predict_lstm_from_df(df)
        prophet_result = predict_prophet_from_df(df)


        return {
            "lstm_anomaly": lstm_result,
            "prophet_anomaly": prophet_result
        }

    except Exception as e:
        return {"error":str(e)}