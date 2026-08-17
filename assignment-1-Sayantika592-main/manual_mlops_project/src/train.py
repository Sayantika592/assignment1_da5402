import pandas as pd
import yaml
import joblib
import json
import subprocess
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load config
cfg = yaml.safe_load(open("config.yaml"))

processed_dir = cfg["data"]["processed_dir"]
version = cfg["data"]["current_version"]

df = pd.read_csv(f"{processed_dir}/{version}_cleaned.csv")

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

split_idx = cfg["data"]["split_index"]
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

params = cfg["model_params"]

model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"]
)

model.fit(X_train, y_train)

#preds = model.predict(X_test)
#acc = accuracy_score(y_test, preds)

# no more accuracy calculated offline but rather production accuracy will be considered
if len(X_test) > 0:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
else:
    acc = None
    print("No test split : training on full data (production retrain mode)")

model_path = f"models/model_{version}.pkl"
joblib.dump(model, model_path)

try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except:
    commit = "no-git"

metadata = {
    "date": str(datetime.now()),
    "dataset_version": version,
    "accuracy": float(acc) if acc is not None else "N/A",
    "git_commit": commit
}

json.dump(metadata, open(f"models/metadata_{version}.json", "w"), indent=4)

with open("models/model_metadata.log", "a") as f:
    f.write(f"{datetime.now()} | {version} | acc={acc}\n")

print(f"Dataset used: {version}")
if acc is not None:
    print(f"Accuracy: {acc:.4f}")
else:
    print("Accuracy: N/A (using production monitoring)")
print(f"Model saved at: {model_path}")
