# app/main.py
import logging
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# ------------------------------------------------
# Logging
# ------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("artifacts/api_requests.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------
# App (define ONCE)
# ------------------------------------------------
app = FastAPI(
    title="DiseaseX DrugA Alert API",
    version="0.1.2",
    description="Predict likelihood of receiving Drug A and flag low-probability eligible patients.",
)

# Serve the demo UI (must be mounted after app is created)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/index.html")

# ------------------------------------------------
# Model & config
# ------------------------------------------------
MODEL_PATH = Path("artifacts/model_minimal.joblib")
if not MODEL_PATH.exists():
    raise RuntimeError("Missing model: artifacts/model_minimal.joblib")
pipe = joblib.load(MODEL_PATH)

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.4"))

# ------------------------------------------------
# Schema
# ------------------------------------------------
class PatientFeatures(BaseModel):
    DISEASEX_DT: str = Field(...)
    PATIENT_AGE: float = Field(...)
    PATIENT_GENDER: str = Field(...)
    NUM_CONDITIONS: int = Field(...)
    PHYSICIAN_TYPE: str = Field(...)
    PHYSICIAN_STATE: str = Field(...)
    LOCATION_TYPE: str = Field(...)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "DISEASEX_DT": "2024-05-10",
                "PATIENT_AGE": 45,
                "PATIENT_GENDER": "F",
                "NUM_CONDITIONS": 3,
                "PHYSICIAN_TYPE": "internal medicine",
                "PHYSICIAN_STATE": "CA",
                "LOCATION_TYPE": "office",
            }]
        }
    }

    @field_validator("PATIENT_GENDER")
    @classmethod
    def norm_gender(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v in {"m", "male"}:
            return "m"
        if v in {"f", "female"}:
            return "f"
        return v

    @field_validator("PHYSICIAN_STATE")
    @classmethod
    def norm_state(cls, v: str) -> str:
        return (v or "").strip().upper()

    @field_validator("PHYSICIAN_TYPE", "LOCATION_TYPE")
    @classmethod
    def norm_text(cls, v: str) -> str:
        return (v or "").strip().lower()

# ------------------------------------------------
# Middleware
# ------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    try:
        response = await call_next(request)
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"{request.method} {request.url.path} [{response.status_code}] {duration:.3f}s")
        return response
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(e)})

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def make_feature_df(patient: PatientFeatures) -> pd.DataFrame:
    anchor = pd.Timestamp("2000-01-01")
    try:
        dx_ts = pd.to_datetime(patient.DISEASEX_DT, format="%Y-%m-%d", errors="raise")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid DISEASEX_DT. Use YYYY-MM-DD.")
    dx_days = int((dx_ts - anchor).days)

    return pd.DataFrame([{
        "DISEASEX_DT": dx_days,
        "PATIENT_AGE": float(patient.PATIENT_AGE),
        "PATIENT_GENDER": patient.PATIENT_GENDER,
        "NUM_CONDITIONS": int(patient.NUM_CONDITIONS),
        "PHYSICIAN_TYPE": patient.PHYSICIAN_TYPE,
        "PHYSICIAN_STATE": patient.PHYSICIAN_STATE,
        "LOCATION_TYPE": patient.LOCATION_TYPE,
    }])

# ------------------------------------------------
# Endpoints
# ------------------------------------------------
@app.post("/predict", summary="Predict likelihood of receiving Drug A")
def predict(patient: PatientFeatures):
    df = make_feature_df(patient)
    p_treated = float(pipe.predict_proba(df)[:, 1])
    logger.info(f"Prediction | Age={patient.PATIENT_AGE} | P_TREATED={p_treated:.3f}")
    return {
        "P_TREATED": round(p_treated, 4),
        "P_UNTREATED": round(1 - p_treated, 4),
        "message": "⚠️ Low likelihood — consider alert" if p_treated < ALERT_THRESHOLD else "✅ Likely to be treated",
    }

@app.post("/alert", summary="Alert if patient likely untreated")
def alert(patient: PatientFeatures):
    df = make_feature_df(patient)
    p_treated = float(pipe.predict_proba(df)[:, 1])
    alert_flag = p_treated < ALERT_THRESHOLD
    logger.info(
        f"Alert | Age={patient.PATIENT_AGE} | P_TREATED={p_treated:.3f} | "
        f"Threshold={ALERT_THRESHOLD} | Alert={alert_flag}"
    )
    return {
        "P_TREATED": round(p_treated, 4),
        "alert": bool(alert_flag),
        "threshold": ALERT_THRESHOLD,
        "message": (
            "🚨 Alert: Eligible but low probability of receiving Drug A"
            if alert_flag else "✅ No alert needed"
        ),
    }