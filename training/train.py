#!/usr/bin/env python
# training/train.py
# ------------------------------------------------------------
# Baseline with extended fields + k-fold CV
#  - Loads data/processed/model_table.csv
#  - Uses ONLY the columns listed in BASELINE_COLUMNS (incl. extended)
#  - TARGET: ever received Drug A (as provided in model_table)
#  - Robust to pandas NA vs numpy nan; OneHotEncoder version-safe
#  - Imputes; trains LR/RF/XGB/LGBM; prints ROC-AUC/PR-AUC/F1/F2
#  - RepeatedStratifiedKFold with per-fold scale_pos_weight
#  - Optional lightweight RandomizedSearchCV for XGBoost
#  - Saves metrics to artifacts/metrics_baseline.json
#  - Saves trained pipeline to artifacts/final_pipe.joblib
#  - No visualization (plots handled elsewhere)
# ------------------------------------------------------------

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from packaging import version

from sklearn import __version__ as sklver
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional imports (gracefully handled if not installed)
try:
    from xgboost import XGBClassifier
except Exception as _xgb_err:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception as _lgbm_err:
    LGBMClassifier = None

from scipy.stats import uniform, randint
import joblib


# -----------------------
# Defaults / Paths
# -----------------------
DEFAULT_DATA_PATH = Path("../data/processed/model_table.csv")
ARTIFACT_DIR = Path("../artifacts")
METRICS_PATH = ARTIFACT_DIR / "metrics_baseline.json"
MODEL_PATH = ARTIFACT_DIR / "final_pipe.joblib"

RANDOM_STATE = 42


# -----------------------
# Columns (required + extended)
# -----------------------
BASELINE_COLUMNS = [
    # --- Required baseline ---
    "DISEASEX_DT",
    "PATIENT_AGE",
    "PATIENT_GENDER",
    "NUM_CONDITIONS",
    "PHYSICIAN_TYPE",
    "PHYSICIAN_STATE",
    "LOCATION_TYPE",

    # --- Extended predictors ---
    "PHYS_TREAT_RATE",        # physician propensity (LOO smoothed)
    "ELIGIBLE",
    "CONTRAINDICATION_LEVEL",
    "HIGH_RISK",
    "IS_AGE65PLUS",
    "HAS_UNDERLYING",
    "AGE_GE_12",
    "DAYS_SYMPTOM_TO_DX",
]
TARGET_COL = "TARGET"


# -----------------------
# Helpers
# -----------------------
def make_ohe() -> OneHotEncoder:
    """Version-robust OneHotEncoder (sparse_output introduced in 1.4)."""
    if version.parse(sklver) >= version.parse("1.4"):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocess(cat_cols, num_cols) -> ColumnTransformer:
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True)),
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
    ])


def build_clf(model_name: str, y_train: np.ndarray):
    """Return a classifier configured with per-fold class balance."""
    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)
    spw = max(neg / max(pos, 1), 1.0)  # scale_pos_weight for imbalanced data

    if model_name == "logreg":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )

    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, class_weight="balanced", random_state=RANDOM_STATE
        )

    if model_name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Please `pip install xgboost`.")
        return XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=1.0,
            min_child_weight=2, tree_method="hist", grow_policy="lossguide",
            max_bin=256, random_state=RANDOM_STATE, n_jobs=-1,
            scale_pos_weight=spw, eval_metric="aucpr"
        )

    if model_name == "lgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed. Please `pip install lightgbm`.")
        return LGBMClassifier(
            n_estimators=1000, learning_rate=0.03, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=3,
            reg_lambda=1.0, reg_alpha=0.0, class_weight={0: 1, 1: spw},
            objective="binary", metric="aucpr", random_state=RANDOM_STATE
        )

    raise ValueError("model_name must be one of: logreg | rf | xgb | lgbm")


def tune_xgb(X_tr_t: np.ndarray, y_tr: np.ndarray) -> "XGBClassifier":
    """Lightweight randomized search in a tight, high-value region."""
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Please `pip install xgboost`.")

    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = max(neg / max(pos, 1), 1.0)

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=900,                    # a bit higher when lr is small
        tree_method="hist",
        grow_policy="lossguide",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=spw,
        subsample=0.9,
        colsample_bytree=0.9,
    )

    # Focused search (as discussed earlier)
    param_dist = {
        "learning_rate": uniform(0.02, 0.02),  # 0.02–0.04
        "max_depth": randint(3, 6),            # 3–5
        "min_child_weight": randint(1, 4),     # 1–3
        "reg_lambda": uniform(1.5, 1.5),       # 1.5–3.0
        "reg_alpha": uniform(0.5, 0.5),        # 0.5–1.0
        "gamma": uniform(0.0, 0.1),
        "subsample": uniform(0.75, 0.25),      # 0.75–1.0
        "colsample_bytree": uniform(0.75, 0.25)
    }

    search = RandomizedSearchCV(
        base, param_distributions=param_dist,
        n_iter=15, scoring="average_precision", cv=3,
        verbose=1, n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X_tr_t, y_tr)
    print(f"Best XGB params: {search.best_params_}")
    return search.best_estimator_


def evaluate_fold(y_true: np.ndarray, proba: np.ndarray):
    """Compute metrics at threshold 0.5 plus AUC/AP."""
    y_pred = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_true, proba)
    pr = average_precision_score(y_true, proba)
    acc = float(np.mean(y_pred == y_true))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = f1_score(y_true, y_pred)

    beta = 2.0
    f2 = (1 + beta**2) * (prec * rec) / max(beta**2 * prec + rec, 1e-12)

    return dict(roc=roc, pr=pr, acc=acc, prec=prec, rec=rec, f1=f1, f2=f2)


def save_metrics_json(path: Path, payload: dict):
    ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(payload, indent=2))


# -----------------------
# Main training routine
# -----------------------
def run_training(
    data_path: Path,
    model_name: str,
    n_splits: int,
    n_repeats: int,
    tune_xgb_flag: bool,
):
    # Load data
    assert data_path.exists(), f"Missing file: {data_path}"
    df = pd.read_csv(data_path)

    missing = [c for c in BASELINE_COLUMNS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in model_table: {missing}")

    # Time → numeric days
    df["DISEASEX_DT"] = pd.to_datetime(df["DISEASEX_DT"], errors="coerce")
    anchor = pd.Timestamp("2000-01-01")
    df["DISEASEX_DT"] = (df["DISEASEX_DT"] - anchor).dt.days

    # Coerce numeric where appropriate
    for c in ["DISEASEX_DT", "PATIENT_AGE", "NUM_CONDITIONS", "DAYS_SYMPTOM_TO_DX",
              "PHYS_TREAT_RATE", "CONTRAINDICATION_LEVEL",
              "IS_AGE65PLUS", "HAS_UNDERLYING", "AGE_GE_12",
              "ELIGIBLE", "HIGH_RISK"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    X = df[BASELINE_COLUMNS].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int).to_numpy()

    # Dtypes & NA harmonization
    cat_cols = ["PATIENT_GENDER", "PHYSICIAN_TYPE", "PHYSICIAN_STATE", "LOCATION_TYPE"]
    num_cols = [c for c in BASELINE_COLUMNS if c not in cat_cols]
    for c in cat_cols:
        X[c] = X[c].astype(object)
    X[cat_cols] = X[cat_cols].replace({pd.NA: np.nan})

    preprocess = build_preprocess(cat_cols, num_cols)

    # CV
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    oof_pred = np.zeros(len(y), dtype=float)
    folds = []
    print("\n===== Cross-Validation Progress =====")

    for fold, (tr_idx, va_idx) in enumerate(rskf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        prep = clone(preprocess)
        prep.fit(X_tr)
        X_tr_t = prep.transform(X_tr)
        X_va_t = prep.transform(X_va)

        if model_name == "xgb" and tune_xgb_flag:
            clf = tune_xgb(X_tr_t, y_tr)
        else:
            clf = build_clf(model_name, y_tr)

        clf.fit(X_tr_t, y_tr)
        proba_va = clf.predict_proba(X_va_t)[:, 1]
        oof_pred[va_idx] = proba_va

        m = evaluate_fold(y_va, proba_va)
        folds.append(m)
        print(
            f"Fold {fold:2d}: ROC-AUC={m['roc']:.3f} | PR-AUC={m['pr']:.3f} | "
            f"Acc={m['acc']:.3f} | P={m['prec']:.3f} | R={m['rec']:.3f} | "
            f"F1={m['f1']:.3f} | F2={m['f2']:.3f}"
        )

    # Summaries
    fold_metrics = {k: np.array([f[k] for f in folds]) for k in folds[0].keys()}
    print("\n===== Cross-Validation Summary (mean ± std) =====")
    for k, arr in fold_metrics.items():
        print(f"{k.upper():<8}: {arr.mean():.3f} ± {arr.std():.3f}")

    roc_oof = roc_auc_score(y, oof_pred)
    pr_oof = average_precision_score(y, oof_pred)
    print(f"\nOverall OOF ROC-AUC={roc_oof:.3f} | PR-AUC={pr_oof:.3f}")

    # Final fit on full data
    final_clf = build_clf(model_name, y)
    final_pipe = Pipeline([("prep", preprocess), ("clf", final_clf)])
    final_pipe.fit(X, y)

    # Save artifacts
    ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(final_pipe, MODEL_PATH)

    # try to persist feature names (helpful for SHAP later)
    try:
        feats = final_pipe.named_steps["prep"].get_feature_names_out()
        pd.Series(feats, name="feature").to_csv(ARTIFACT_DIR / "feature_names.csv", index=False)
    except Exception:
        pass

    payload = {
        "sklearn_version": sklver,
        "model": model_name,
        "columns_used": BASELINE_COLUMNS,
        "n_rows": int(len(X)),
        "cv": {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "per_fold": folds,
            "summary": {k: {"mean": float(v.mean()), "std": float(v.std())}
                        for k, v in fold_metrics.items()},
            "oof": {"roc_auc": float(roc_oof), "pr_auc": float(pr_oof)},
        },
    }
    save_metrics_json(METRICS_PATH, payload)

    print("\n✅ Final model trained on all data and saved.")
    print(f"💾 Pipeline: {MODEL_PATH}")
    print(f"📊 Metrics : {METRICS_PATH}")


# -----------------------
# CLI
# -----------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Train baseline model with extended fields and k-fold CV.")
    p.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH),
                   help="Path to model_table.csv")
    p.add_argument("--model", type=str, choices=["logreg", "rf", "xgb", "lgbm"], default="xgb",
                   help="Classifier to use")
    p.add_argument("--splits", type=int, default=5, help="CV splits (k)")
    p.add_argument("--repeats", type=int, default=2, help="CV repeats")
    p.add_argument("--tune-xgb", action="store_true",
                   help="Run lightweight RandomizedSearchCV for XGBoost each fold")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_training(
        data_path=Path(args.data_path),
        model_name=args.model,
        n_splits=int(args.splits),
        n_repeats=int(args.repeats),
        tune_xgb_flag=bool(args.tune_xgb),
    )