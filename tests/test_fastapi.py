from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_GOOD = {
    "DISEASEX_DT": "2024-05-10",
    "PATIENT_AGE": 45,
    "PATIENT_GENDER": "F",
    "NUM_CONDITIONS": 3,
    "PHYSICIAN_TYPE": "internal medicine",
    "PHYSICIAN_STATE": "CA",
    "LOCATION_TYPE": "office"
}

def test_predict_ok():
    r = client.post("/predict", json=SAMPLE_GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["P_TREATED"] <= 1.0
    assert "message" in body

def test_alert_ok():
    r = client.post("/alert", json=SAMPLE_GOOD)
    assert r.status_code == 200
    body = r.json()
    assert "alert" in body
    assert "threshold" in body

def test_validation_errors():
    # Missing required field
    bad = {**SAMPLE_GOOD}
    bad.pop("PATIENT_AGE")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422