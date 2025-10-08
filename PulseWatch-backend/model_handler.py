import numpy as np
import pandas as pd
# Assuming the import path is correct after previous fixes
from model.anamoly_detection import MetricPredictor,run_inference_df 
from model.prophet_model import detect_multimetric_anomalies_df
import os # <-- Added os for potential path handling, though not strictly needed here if constants are absolute

# Assuming these paths are correct relative to the PulseWatch-backend directory
MODEL_PATH = "model/lstmae.pth"
SCALER_PATH = "model/scaler.gz"

def predict_lstm_from_df(df):
    """Run anomaly detection using LSTM Autoencoder on in-memory metrics DataFrame."""
    return run_inference_df(df, model_path=MODEL_PATH, scaler_path=SCALER_PATH)

def train_lstm_from_df(df):
    """Train or retrain LSTM Autoencoder on in-memory metrics DataFrame."""
    feature_cols = [col for col in df.columns if col not in ["timestamp", "ds"]]
    
    # --- FIX APPLIED HERE ---
    # Reduced seq_len from 20 to 10. If your CSV has less than 11 rows, training will still fail.
    # If the training still fails, try seq_len=5, or add more data to merged_metrics.csv.
    predictor = MetricPredictor(
        feature_cols=feature_cols,
        seq_len=10, 
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
    )
    # ------------------------
    
    predictor.train(df, epochs=20)
    return "LSTM model retrained successfully."

def predict_prophet_from_df(df):
    """Run Prophet-based anomaly detection on in-memory metrics DataFrame."""
    results = detect_multimetric_anomalies_df(df)
    print("Prophet anomaly detection complete.")
    return results