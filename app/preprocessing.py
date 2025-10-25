import pandas as pd

FEATURE_COLS = [
    "DISEASEX_DT","PATIENT_AGE","PATIENT_GENDER","NUM_CONDITIONS",
    "PHYSICIAN_TYPE","PHYSICIAN_STATE","LOCATION_TYPE"
]
CAT_COLS = ["PATIENT_GENDER","PHYSICIAN_TYPE","PHYSICIAN_STATE","LOCATION_TYPE"]
NUM_COLS = ["PATIENT_AGE","NUM_CONDITIONS"]

def to_frame(patients) -> pd.DataFrame:
    """Convert JSON request data into a pandas DataFrame with expected columns."""
    df = pd.DataFrame([p.dict() if hasattr(p, "dict") else p for p in patients])

    # Ensure all expected columns exist
    for c in FEATURE_COLS:
        if c not in df:
            df[c] = None

    # Parse and derive numeric date feature
    df["DISEASEX_DT"] = pd.to_datetime(df["DISEASEX_DT"], errors="coerce")
    df["DX_DAYS_SINCE2000"] = (df["DISEASEX_DT"] - pd.Timestamp("2000-01-01")).dt.days

    # Coerce numerics & categoricals
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_COLS:
        df[c] = df[c].astype("object")

    # Match model training feature order
    return df[["PATIENT_AGE","PATIENT_GENDER","NUM_CONDITIONS",
               "PHYSICIAN_TYPE","PHYSICIAN_STATE","LOCATION_TYPE",
               "DX_DAYS_SINCE2000"]]