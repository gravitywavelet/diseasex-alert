#!/usr/bin/env python
# training/inference.py
# ------------------------------------------------------------
# Batch inference for the trained pipeline:
#  - Loads artifacts/final_pipe.joblib (from training/train.py)
#  - Reads data/processed/model_table.csv by default
#  - Recomputes DISEASEX_DT as days since 2000-01-01 (same as training)
#  - Scores patients (P_TREATED), computes P_UNTREATED = 1 - P_TREATED
#  - Optionally filters to ELIGIBLE==1 before slicing
#  - Exports ranked scores and alert slices at coverage levels
# ------------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Paths (can override via CLI)
DEFAULT_MODEL_PATH = Path("../artifacts/final_pipe.joblib")
DEFAULT_DATA_PATH  = Path("../data/processed/model_table.csv")
DEFAULT_OUT_DIR    = Path("../artifacts")

# Feature list must match training/train.py
FEATURES = [
    "DISEASEX_DT",
    "PATIENT_AGE",
    "PATIENT_GENDER",
    "NUM_CONDITIONS",
    "PHYSICIAN_TYPE",
    "PHYSICIAN_STATE",
    "LOCATION_TYPE",
    "PHYS_TREAT_RATE",
    "ELIGIBLE",
    "CONTRAINDICATION_LEVEL",
    "HIGH_RISK",
    "IS_AGE65PLUS",
    "HAS_UNDERLYING",
    "AGE_GE_12",
    "DAYS_SYMPTOM_TO_DX",
]


def _ensure_diseasex_days(df: pd.DataFrame) -> pd.DataFrame:
    """Make DISEASEX_DT numeric: days since 2000-01-01 (same as training)."""
    if "DISEASEX_DT" not in df.columns:
        raise ValueError("DISEASEX_DT is missing from the input data.")
    anchor = pd.Timestamp("2000-01-01")
    dt = pd.to_datetime(df["DISEASEX_DT"], errors="coerce")
    df = df.copy()
    df["DISEASEX_DT"] = (dt - anchor).dt.days
    return df


def _check_columns(df: pd.DataFrame):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for inference: {missing}")


def run_inference(
    model_path: Path,
    data_path: Path,
    out_dir: Path,
    coverages: list[float],
    eligible_only: bool,
    id_col: str | None = "PATIENT_ID",
):
    # Load
    assert model_path.exists(), f"Model not found: {model_path}"
    assert data_path.exists(),  f"Data not found: {data_path}"
    pipe = joblib.load(model_path)
    df = pd.read_csv(data_path)

    # Prep data (match training)
    _check_columns(df)
    df = _ensure_diseasex_days(df)

    # Score
    X = df[FEATURES].copy()
    # Ensure categoricals as object (robustness)
    for c in ["PATIENT_GENDER", "PHYSICIAN_TYPE", "PHYSICIAN_STATE", "LOCATION_TYPE"]:
        if c in X:
            X[c] = X[c].astype(object)

    proba = pipe.predict_proba(X)[:, 1]
    scored = df.copy()
    scored["P_TREATED"] = proba
    scored["P_UNTREATED"] = 1.0 - proba

    # Rank low→high P_TREATED (i.e., “less likely to be treated” first)
    scored = scored.sort_values("P_TREATED", ascending=True).reset_index(drop=True)

    # Output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full scored table
    scored_path = out_dir / "scored_patients.csv"
    scored.to_csv(scored_path, index=False)
    print(f"💾 Saved full scored table → {scored_path} (n={len(scored)})")

    # Optionally restrict to eligible before slicing
    base = scored
    if eligible_only:
        if "ELIGIBLE" not in scored.columns:
            raise ValueError("ELIGIBLE column missing but --eligible-only set.")
        base = scored[scored["ELIGIBLE"] == 1].copy()
        print(f"ℹ️  Using ELIGIBLE==1 subset for alert slicing (n={len(base)})")

    # Coverage slices: take the bottom X% by P_TREATED
    for cov in coverages:
        if not (0 < cov < 1):
            raise ValueError(f"Coverage must be in (0,1). Got {cov}.")
        cutoff = base["P_TREATED"].quantile(cov)
        alerts = base[base["P_TREATED"] <= cutoff].copy()

        # Save alerts
        pct = int(cov * 100)
        fname = out_dir / f"eligible_alerts_{pct}pct.csv" if eligible_only else out_dir / f"alerts_{pct}pct.csv"
        alerts.to_csv(fname, index=False)
        print(
            f"📣 Coverage={pct}% | cutoff={cutoff:.3f} | "
            f"alerts={len(alerts)} → {fname}"
        )

    # Quick summary for sanity
    n = len(scored)
    msg_id = f" (id_col={id_col})" if id_col and id_col in scored.columns else ""
    print(
        f"\n✅ Inference complete for {n} rows{msg_id}.\n"
        f"   Min/Median/Max P_TREATED: "
        f"{scored['P_TREATED'].min():.3f} / "
        f"{scored['P_TREATED'].median():.3f} / "
        f"{scored['P_TREATED'].max():.3f}"
    )


def _parse_args():
    p = argparse.ArgumentParser(description="Batch inference for Disease X model.")
    p.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH),
                   help="Path to joblib pipeline (default: artifacts/final_pipe.joblib)")
    p.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH),
                   help="CSV with model_table columns (default: data/processed/model_table.csv)")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                   help="Directory to write outputs (default: artifacts)")
    p.add_argument("--coverages", type=str, default="0.60,0.70,0.90",
                   help="Comma-separated coverage levels, e.g. 0.60,0.70,0.90")
    p.add_argument("--eligible-only", action="store_true",
                   help="Slice alerts from ELIGIBLE==1 subset")
    p.add_argument("--id-col", type=str, default="PATIENT_ID",
                   help="Optional ID column name for reference in logs")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    covs = [float(x.strip()) for x in args.coverages.split(",") if x.strip()]
    run_inference(
        model_path=Path(args.model_path),
        data_path=Path(args.data_path),
        out_dir=Path(args.out_dir),
        coverages=covs,
        eligible_only=bool(args.eligible_only),
        id_col=args.id_col or None,
    )