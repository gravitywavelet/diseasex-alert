from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from .schema import (
    PredictRequest, PredictResponse, PredictResponseRow,
    AlertRequest, AlertResponse, AlertRow
)
from .preprocessing import to_frame
from .model import model_service
from .config import settings

app = FastAPI(title="Drug A EMR Alert API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_service.pipe is not None,
        "stats_loaded": bool(model_service.stats),
        "default_coverage": settings.DEFAULT_ALERT_COVERAGE,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X = to_frame(req.patients)
        p = model_service.predict_proba(X)[:, 1]
        results = [PredictResponseRow(p_treated=float(pi), p_untreated=float(1 - pi)) for pi in p]
        return PredictResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.post("/alert", response_model=AlertResponse)
def alert(req: AlertRequest):
    try:
        X = to_frame(req.patients)
        p = model_service.predict_proba(X)[:, 1]

        coverage = req.coverage or settings.DEFAULT_ALERT_COVERAGE
        cutoff = req.cutoff or model_service.cutoff_for_coverage(coverage)
        if cutoff is None:
            cutoff = float(np.quantile(p, coverage))

        alerts = (p <= cutoff).astype(bool)
        results = [
            AlertRow(p_treated=float(pi), p_untreated=float(1 - pi), alert=bool(a))
            for pi, a in zip(p, alerts)
        ]
        return AlertResponse(coverage=float(coverage), cutoff=float(cutoff), results=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Alert generation failed: {e}")