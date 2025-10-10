from model_handler import train_lstm_from_df,predict_prophet_from_df,predict_lstm_from_df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CORRECTED PATH HANDLING ---
# Get the absolute path of the script's directory (PulseWatch-backend)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Construct the absolute path to the CSV file
CSV_PATH = os.path.join(SCRIPT_DIR, "data", "merged_metrics.csv")
# -------------------------------

def inject_spike_anomalies(df, columns=None, n_anomalies=5, magnitude=10):
    df_anom = df.copy()
    if columns is None:
        columns = df.columns.tolist()
    
    for _ in range(n_anomalies):
        col = np.random.choice(columns)
        idx = np.random.randint(0, len(df))
        print(idx)
        # Use .loc to target the specific cell for injection
        df_anom.loc[df_anom.index[idx], col] += magnitude * df[col].std()

    return df_anom

# --- CORRECTED FILE READING ---
# Use the robust path variable
df = pd.read_csv(CSV_PATH) 

# --- CORRECTED COLUMN NAMES ---
# Changed 'cpu_usage', 'memory_available', 'latency' to the correct 'cpu', 'memory', 'latency'
df_anamolies = inject_spike_anomalies(df,columns=['cpu','memory','latency'],n_anomalies=10)

train_lstm_from_df(df)
lstm_result,reconstructed,errors = predict_lstm_from_df(df_anamolies)
prophet_result = predict_prophet_from_df(df_anamolies)

for i, l_res in lstm_result.iterrows():
    if l_res["anomaly"]: 
        print("Anomaly detected at index:", i)
        print("LSTM Reconstruction Error:", l_res["reconstruction_error"])

print("=====================================")
# Feature selection for plotting
feature_index = 1 
feature_name = df_anamolies.columns[feature_index]
# Ensure the plotting uses the correct column name for indexing
original_plot = df_anamolies.loc[lstm_result.index, feature_name] 
reconstructed_plot = reconstructed[:, feature_index]

plt.figure(figsize=(12, 6))
plt.plot(original_plot.values, label='Actual', color='blue', linewidth=2)
plt.plot(reconstructed_plot, label='Reconstructed', color='orange', linestyle='solid', linewidth=2)
# X-axis index adjustment: The anomaly indices start after the sequence length (e.g., 30 steps), 
# so we plot them relative to the start of the 'lstm_result' series.
plt.scatter(lstm_result.index[lstm_result["anomaly"] == 1] - lstm_result.index[0],
            original_plot[lstm_result["anomaly"] == 1],
            color='red', label='Anomaly', zorder=5)

plt.title(f"Actual vs Reconstructed ({feature_name})", fontsize=14)
plt.xlabel("Time Steps")
plt.ylabel(feature_name)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()