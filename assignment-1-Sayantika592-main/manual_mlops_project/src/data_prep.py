import pandas as pd
import yaml
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# read config
cfg = yaml.safe_load(open("config.yaml"))

raw_path = cfg["data"]["raw_path"]
processed_dir = cfg["data"]["processed_dir"]
version = cfg["data"]["current_version"]

# load raw
df = pd.read_csv(raw_path)

df = df.fillna(df.median(numeric_only=True))
num_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols]) # standardized so that features are of similar importance and not biased towards those with larger numbers

df = df.drop_duplicates()
df = df.drop(columns=["UDI", "Product ID"]) # these may not be any use so dropped
df = pd.get_dummies(df, columns=["Type"]) # categories encoded as numbers for better understanding by the model
df = df.drop(columns=["TWF","HDF","PWF","OSF","RNF"]) # machine failure depends on these 5 failure modes, so they are dropeed to prevent leakage


# save new version
save_file = f"{processed_dir}/{version}_cleaned.csv"
df.to_csv(save_file, index=False)

print("Saved:", save_file)

# update manifest
with open("data/manifest.txt", "a") as f:
    f.write(f"{datetime.now()} → created {version}_cleaned.csv using data_prep.py\n")
