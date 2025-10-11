# Testing and Validation

File: `test.py`

### Key Function
`inject_spike_anomalies(df, columns, n_anomalies, magnitude)`

- Adds random spikes to simulate anomalies.  
- Evaluates LSTM and Prophet models on modified data.  
- Visualizes reconstruction vs actual signals using Matplotlib.