from fastapi import FastAPI

app=FastAPI(title="Pulse_Watch",
            description="Real-time anomaly detection")

@app.get("/")
def root():
  return {"msg":"Welcome to PulseWatch"}

