import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib
import os

class LSTMAutoEncoder(nn.Module):
    def __init__(self, encoding_dim=128, hidden_dim=256, n_features=3, n_layers=3, dropout=0.3):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder_fc = nn.Linear(hidden_dim, encoding_dim)
        self.decoder_fc = nn.Linear(encoding_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        encoded = F.relu(self.encoder_fc(encoded[:, -1, :]))
        decoded = F.relu(self.decoder_fc(encoded))
        decoded, _ = self.decoder(decoded.unsqueeze(1).repeat(1, x.size(1), 1))
        decoded = self.out(decoded)
        return decoded[:, -1, :]

class MetricPredictor():
    def __init__(self, feature_cols, seq_len=30, model_path="lstmae.pth", scaler_path="scaler.gz"):
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = LSTMAutoEncoder(n_features=len(feature_cols))
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def make_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.seq_len):
            sequences.append(data[i:i + self.seq_len])
        return np.array(sequences)

    def train(self, df, epochs=80, lr=5e-4, batch_size=64):
        data = df[self.feature_cols].values
        scaled_data = self.scaler.fit_transform(data)
        sequences = self.make_sequences(scaled_data)
        if len(sequences) == 0:
            print("Not enough data to train.")
            return
        X = torch.tensor(sequences, dtype=torch.float32)
        Y = X[:, -1, :]
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.SmoothL1Loss()
        self.model.train()
        for e in range(epochs):
            total_loss = 0
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                output = self.model(xb)
                loss = criterion(output, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            if (e + 1) % 5 == 0:
                print(f"[TRAIN] Epoch {e+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
        torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print("Model and scaler saved.")

    def load(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Model or scaler not found. Train the model first.")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.scaler = joblib.load(self.scaler_path)
        self.model.eval()
        print("Model and scaler loaded.")

    def predict_df(self, df):
        data = df[self.feature_cols].values
        scaled_data = self.scaler.transform(data)
        sequences = self.make_sequences(scaled_data)
        if len(sequences) == 0:
            print("Not enough data to predict.")
            return df
        X = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            reconstructed = self.model(X).cpu().numpy()
        reconstructed = self.scaler.inverse_transform(reconstructed)
        original = data[self.seq_len:]
        errors = np.mean(np.square(original - reconstructed), axis=1)
        threshold = np.mean(errors) + 2.5 * np.std(errors)
        anomalies = errors > threshold
        df_result = df.iloc[self.seq_len:].copy()
        df_result["reconstruction_error"] = errors
        df_result["anomaly"] = anomalies
        return df_result, reconstructed, errors

def run_inference_df(df, model_path="model/lstmae.pth", scaler_path="model/scaler.gz", seq_len=30):
    feature_cols = [col for col in df.columns if col not in ["timestamp", "ds"]]
    predictor = MetricPredictor(feature_cols=feature_cols, seq_len=seq_len, model_path=model_path, scaler_path=scaler_path)
    predictor.load()
    results = predictor.predict_df(df)
    return results