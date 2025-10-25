from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date

class PatientFeatures(BaseModel):
    DISEASEX_DT: Optional[date] = Field(None, description="YYYY-MM-DD")
    PATIENT_AGE: Optional[float]
    PATIENT_GENDER: Optional[str]
    NUM_CONDITIONS: Optional[float]
    PHYSICIAN_TYPE: Optional[str]
    PHYSICIAN_STATE: Optional[str]
    LOCATION_TYPE: Optional[str]

    @validator("PATIENT_GENDER")
    def norm_gender(cls, v):
        if v is None:
            return v
        v = str(v).strip().lower()
        return v if v in {"m", "f"} else None

class PredictRequest(BaseModel):
    patients: List[PatientFeatures]

class PredictResponseRow(BaseModel):
    p_treated: float
    p_untreated: float

class PredictResponse(BaseModel):
    results: List[PredictResponseRow]

class AlertRequest(BaseModel):
    patients: List[PatientFeatures]
    coverage: Optional[float] = None
    cutoff: Optional[float] = None

class AlertRow(BaseModel):
    p_treated: float
    p_untreated: float
    alert: bool

class AlertResponse(BaseModel):
    coverage: float
    cutoff: float
    results: List[AlertRow]