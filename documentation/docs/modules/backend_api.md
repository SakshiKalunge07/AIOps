# Backend API (FastAPI)

### Core Routes

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/` | GET | Welcome message |
| `/metrics/live-short` | GET | Fetch short-term live metrics |
| `/metrics/live-long` | GET | Long-term predictions |
| `/metrics/test` | GET | Test data with artificial spikes |
| `/train` | POST | Retrain LSTM model |

### Implementation
File: `app/main.py`

Functions:
- `get_short_live_data()` → Fetch 20 most recent data points  
- `get_long_live_data()` → Full window analysis  
- `get_test_data()` → Inject anomalies and evaluate model
