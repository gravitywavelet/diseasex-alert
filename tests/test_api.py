import requests
import json

BASE_URL = "http://127.0.0.1:8000"

sample_payload = {
    "DISEASEX_DT": "2024-05-10",
    "PATIENT_AGE": 45,
    "PATIENT_GENDER": "F",
    "NUM_CONDITIONS": 3,
    "PHYSICIAN_TYPE": "internal medicine",
    "PHYSICIAN_STATE": "CA",
    "LOCATION_TYPE": "office"
}

def test_predict():
    url = f"{BASE_URL}/predict"
    resp = requests.post(url, json=sample_payload)
    print(f"[POST /predict] status={resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

def test_alert():
    url = f"{BASE_URL}/alert"
    resp = requests.post(url, json=sample_payload)
    print(f"[POST /alert] status={resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

if __name__ == "__main__":
    print("Running DiseaseX API tests...\n")
    test_predict()
    print("\n-----------------------------\n")
    test_alert()