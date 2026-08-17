import requests
import pandas as pd
import yaml

# load config
cfg = yaml.safe_load(open("config.yaml"))
host = cfg["deployment"]["host"]
port = cfg["deployment"]["port"]

BASE_URL = f"http://{host}:{port}"


def get_sample():
    cols = pd.read_csv("data/processed/v1_cleaned.csv") \
            .drop(columns=["Machine failure"]).columns
    return {c: 0 for c in cols}


# Test 1
def test_health():
    r = requests.get(BASE_URL + "/")
    assert r.status_code == 200
    print("Test 1 Passed: API alive")


# Test 2
def test_prediction_status():
    sample = get_sample()
    r = requests.post(BASE_URL + "/predict", json=sample)
    assert r.status_code == 200
    print("Test 2 Passed: Predict endpoint reachable")


# Test 3
def test_prediction_format():
    sample = get_sample()
    r = requests.post(BASE_URL + "/predict", json=sample)
    data = r.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], int)
    print("Test 3 Passed: Output format correct")


if __name__ == "__main__":
    test_health()
    test_prediction_status()
    test_prediction_format()
