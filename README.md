# Manual MLOps Challenge – Predictive Maintenance

## Overview

This project implements a fully manual MLOps pipeline for a predictive maintenance problem using the AI4I 2020 dataset.

The objective of this assignment was to simulate a real-world ML lifecycle **without using automated MLOps tools**, and to manually implement:

- Data versioning
- Configuration isolation
- Model training & artifact tracking
- API deployment
- Monitoring & drift detection
- Retraining workflow

The purpose was to understand the complexity behind production ML systems and the importance of automation.

---

# Project Structure

```
manual_mlops_project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── production/
│   └── manifest.txt
│
├── models/
│   ├── model_vX.pkl
│   ├── metadata_vX.json
│   └── model_metadata.log
│
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── inference.py
│   ├── monitor.py
│   ├── test_api.py
│   └── day2_data.py
│
├── config.yaml
├── deployment_log.csv
└── monitoring_log.csv
```

---

# Phase A – Data & Configuration Management

## Manual Data Versioning

- Raw data is stored in `data/raw/`
- Cleaned datasets are stored as versioned files:
  - `v1_cleaned.csv`
  - `v2_cleaned.csv`
- Every time data is processed, `manifest.txt` is updated manually with:
  - Timestamp
  - Version name
  - Script used

This creates a manual lineage trail from raw → processed data.

## Configuration Isolation

All paths and parameters are stored in `config.yaml`, including:

- Raw data path
- Processed directory
- Current dataset version
- Split index
- Model hyperparameters
- Deployment settings

No file paths or hyperparameters are hardcoded in scripts.

Changing `current_version` in `config.yaml` automatically switches dataset usage.

---

# Phase B – Manual Model Registry

## Training Process

`train.py` performs the following:

- Loads dataset based on `config.yaml`
- Splits data chronologically
- Trains a `RandomForestClassifier`
- Saves model as: models/model_vX.pkl
- Saves metadata as: models/metadata_vX.json


### Metadata Contains

- Training date
- Dataset version
- Accuracy
- Git commit hash

Additionally:

- `model_metadata.log` records all model versions and metrics

This acts as a manual model registry.

---

# Phase C – Manual Deployment

## API Wrapper

`inference.py` wraps the trained model using FastAPI.

Endpoints:

- `GET /` → Health check
- `POST /predict` → Returns prediction

The model loaded depends on `current_version` in `config.yaml`.

## Deployment Logging

Every time the server starts:

`deployment_log.csv` is updated with:

- Timestamp
- Active model version

This helps track which model is currently live.

## Smoke Tests

`test_api.py` includes three basic smoke tests:

1. Health check test
2. Prediction endpoint status test
3. Output format validation test

These ensure that:

- The API is running
- The prediction endpoint is reachable
- The response format is correct

---

# Phase D – Monitoring & Drift Simulation

## Drift Simulation

A "Day 2" dataset was created to simulate production drift by:

- Increasing air and process temperature slightly
- Increasing torque
- Increasing tool wear
- Adding random noise to rotational speed

This dataset is stored in: data/production/day2.csv


## Monitoring Script

`monitor.py`:

- Sends production data to the API
- Computes production accuracy
- Computes production error
- Compares production error with training error
- Flags drift if error increase exceeds threshold

Monitoring results are logged in: monitoring_log.csv


## Retraining Workflow

If drift is detected:

1. Update `current_version` in `config.yaml`
2. Run: python src/data_prep.py
python src/train.py


3. Restart API server

This simulates a manual retraining trigger.

---

# Key Learning Outcomes

Through this manual implementation, I understood that:

- Manual data versioning is fragile and dependent on human discipline.
- It is easy to accidentally overwrite datasets.
- Deployment can serve outdated models if the server is not restarted.
- Reproducibility requires tracking:
  - Dataset version
  - Hyperparameters
  - Code version (Git commit hash)
- Monitoring production performance is critical to detect drift.

This project demonstrated why automated tools such as:

- Experiment tracking systems
- Model registries
- Automated retraining pipelines

are essential in real-world ML systems.

---

# How to Run the Project

## 1. Data Preparation

python src/data_prep.py

## 2. Train Model

python src/train.py

## 3. Run API

uvicorn src.inference:app --reload --port 8001

## 4. Run Tests

python src/test_api.py

## 5. Monitor Production

python src/monitor.py


---

# Final Reflection

This assignment demonstrated that building an MLOps pipeline manually is possible but highly error-prone. While manageable at small scale, manual processes become fragile as the number of dataset versions and model iterations increases. Automated MLOps tools exist to reduce human dependency and improve reproducibility, monitoring, and deployment safety.








