import numpy as np
import pandas as pd
from model.anamoly_detection import MetricPredictor,run_inference_df 
from model.prophet_model import detect_multimetric_anomalies_df
import os 

MODEL_PATH = "model/lstmae.pth"
SCALER_PATH = "model/scaler.gz"

def predict_lstm_from_df(df):
    """Run anomaly detection using LSTM Autoencoder on in-memory metrics DataFrame."""
    return run_inference_df(df, model_path=MODEL_PATH, scaler_path=SCALER_PATH)

def train_lstm_from_df(df):
    """Train or retrain LSTM Autoencoder on in-memory metrics DataFrame."""
    feature_cols = [col for col in df.columns if col not in ["timestamp", "ds"]]
    
    # --- FIX APPLIED HERE ---
    # Reduced seq_len from 20 to 10. 
    predictor = MetricPredictor(
        feature_cols=feature_cols,
        seq_len=10, 
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
    )
    
    predictor.train(df, epochs=20)
    return "LSTM model retrained successfully."

def predict_prophet_from_df(df):
    """Run Prophet-based anomaly detection on in-memory metrics DataFrame."""
    results = detect_multimetric_anomalies_df(df)
    print("Prophet anomaly detection complete.")
    if isinstance(results, tuple):
        results = results[0]
    return results
