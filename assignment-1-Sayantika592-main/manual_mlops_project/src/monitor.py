import pandas as pd
import requests
import yaml
from datetime import datetime
import json

# config
cfg = yaml.safe_load(open("config.yaml"))

host = cfg["deployment"]["host"]
port = cfg["deployment"]["port"]

BASE_URL = f"http://{host}:{port}"
URL = f"{BASE_URL}/predict"

threshold = cfg["deployment"]["threshold"]
version = cfg["data"]["current_version"]

meta = json.load(open(f"models/metadata_{version}.json"))

train_acc = meta["accuracy"]
if train_acc != "N/A":
    train_error = 1 - float(train_acc)
else:
    train_error = None

# load production data
df = pd.read_csv("data/production/day2.csv")

X = df.drop(columns=["Machine failure"])
y_true = df["Machine failure"]

# call API for predictions
preds = []

for _, row in X.iterrows():
    sample = row.to_dict()
    r = requests.post(URL, json=sample)
    preds.append(r.json()["prediction"])

# compute accuracy
correct = sum(p == t for p, t in zip(preds, y_true))
prod_acc = correct / len(y_true)
prod_error = 1 - prod_acc

print("Training error:", train_error)
print("Production accuracy:", prod_error)

status = "healthy"

if train_error is not None:
    error_increase = prod_error - train_error
    print("Error increase:", error_increase)

    if error_increase > threshold:
        status = "drift"

# append to monitoring log
with open("monitoring_log.csv", "a") as f:
    f.write(f"{datetime.now()},{prod_acc:.4f},{status}\n")

# compare with threshold
if prod_error > threshold:
    print("Drift detected — retraining needed!")
else:
    print("Model healthy")
