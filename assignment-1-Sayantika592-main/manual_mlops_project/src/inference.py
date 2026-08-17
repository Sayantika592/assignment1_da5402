import yaml
import joblib
import pandas as pd
from fastapi import FastAPI
from datetime import datetime

# load config
cfg = yaml.safe_load(open("config.yaml"))
version = cfg["data"]["current_version"]

model_path = f"models/model_{version}.pkl"

model = joblib.load(model_path)

app = FastAPI()

# log deployment
with open("deployment_log.csv", "a") as f:
    f.write(f"{datetime.now()}, model_{version}.pkl\n")


# health check
@app.get("/")
def home():
    return {"status": "API running"}


# prediction endpoint
@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
