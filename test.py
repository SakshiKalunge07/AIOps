from model_handler import train_lstm_from_df,predict_prophet_from_df,predict_lstm_from_df
import numpy as np
import pandas as pd

def inject_spike_anomalies(df, columns=None, n_anomalies=5, magnitude=3):
    df_anom = df.copy()
    if columns is None:
        columns = df.columns.tolist()
    
    for _ in range(n_anomalies):
        col = np.random.choice(columns)
        idx = np.random.randint(0, len(df))
        print(idx)
        df_anom.iloc[idx, df.columns.get_loc(col)] += magnitude * df[col].std()

    return df_anom

df = pd.read_csv("project_metrics_wide_format.csv")
df_anamolies = inject_spike_anomalies(df,columns=['cpu_idle_time','memory_available','latency_seconds'],n_anomalies=5)
train_lstm_from_df(df)
lstm_result = predict_lstm_from_df(df_anamolies)
prophet_result = predict_prophet_from_df(df_anamolies)

for i, l_res in lstm_result.iterrows():
    if l_res["anomaly"] == 1:
        print("Anomaly detected at index:", i)

print("=====================================")