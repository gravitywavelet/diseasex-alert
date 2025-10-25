
# 🧬 DiseaseX — Treatment Propensity & EMR Alert Simulation

## Overview

This project builds a complete, modular pipeline to predict which patients are less likely to be treated with Drug A for Disease X and to simulate EMR alerts that can guide physician follow-up.
It covers data cleaning → feature engineering → machine-learning modeling → explainability → deployment via FastAPI + Docker.

![SHAP Summary Plot](notebooks/artifacts/ema.png)

⸻

## Project Overview

This project follows the full data-to-deployment pipeline:

Steps Description
1. Data Cleaning
2. Feature Engineering
3. Modeling & Evaluation
4. EMA Aleart Simulation
4. API Deployment (FastAPI) 
5. Containerization
6. Software Architecture
7. Version Control 
⸻

🧩 Project Structure

├── app/                    # Main application package
│   ├── api.py              # FastAPI endpoints for prediction
│   ├── config.py           # Configuration and constants
│   ├── main.py             # CLI / app entrypoint
│   ├── main_html.py        # (optional) HTML demo interface
│   ├── model.py            # Model loading & inference
│   ├── preprocessing.py    # Feature preprocessing pipeline
│   ├── schema.py           # Pydantic request/response schemas
│   └── static/             # Front-end assets (if any)
│
├── training/               # Offline model training components
│   ├── train.py            # Model training & CV evaluation
│   └── inference.py        # Batch or API inference utilities
│
├── artifacts/              # Saved models, metrics & plots
│   ├── model_minimal.joblib
│   ├── metrics_baseline.json
│   ├── roc_curve_minimal.png
│   ├── pr_curve_minimal.png
│   ├── threshold.json
│   ├── emr_alerts_60pct.csv
│   └── xgb_shap_global_importance.csv
│
├── data/
│   ├── clean/              # Cleaned tables (Fact and Dim) 
│   └── processed/          # Final model_table.csv (submission file)
│
├── notebooks/              # Jupyter notebooks (EDA, modeling, SHAP)
│   ├── 0_EDA.ipynb
│   ├── 1a_Preprocessing_clean.ipynb
│   ├── 2a_Preprocessing_modeltable.ipynb
│   ├── 3b_modeling.ipynb
│   └── DSI LT Interview Exercise – Oct 2025 (candidate).xlsx (Raw Data)
│
├── tests/                  # Unit tests for API
│   ├── test_api.py
│   └── test_fastapi.py
│
├── Dockerfile
├── requirements.txt
└── README.md

---
##  Step 0 - EDA (Evaluate data and form actions to be taken)
Scripts in `notebooks/0_EDA.ipynb` 

## 🧹 Step 1 – Data Cleaning 
Scripts in `notebooks/1a_Preprocessing_clean.ipynb` and `notebooks/1b_EDA_post_clean.ipynb`  normalize column names, unify string formats, and enforce correct types.  
**Key actions**

| Step | Description | Impact |
|------|--------------|--------|
| **1. Drop duplicates** | Removed exact duplicate rows from `fact_txn` | Ensures one record per event |
| **2. Parse dates** | Converted `TXN_DT` to datetime (`pd.to_datetime`) with coercion | Enables temporal analysis |
| **3. Normalize IDs** | Converted `PATIENT_ID` and `PHYSICIAN_ID` to consistent string or Int64 formats (stripped trailing `.0`, whitespace, “None”) | Prevents join mismatches |
| **4. Text normalization** | Lowercased and trimmed descriptive columns (`TXN_TYPE`, `TXN_DESC`, `TXN_LOCATION_TYPE`, etc.) | Avoids category fragmentation |
| **5. State normalization** | Converted physician `STATE` to uppercase and mapped full names → USPS abbreviations (e.g., “California” → “CA”) | Standardizes geolocation features |
| **6. Gender cleanup** | Restricted to `"m"` / `"f"`, replaced others with `NaN` | Harmonizes patient attributes |
| **7. TXN_TYPE filtering** | Restricted to allowed values `{conditions, symptoms, contraindications, treatments}` | Removes irrelevant transactions |
| **8. Summary reporting** | Printed counts of dropped duplicates, null ratios, and normalized unique values | Ensures transparent QA |

**Output:**  
Clean versions of all three tables:

1b_EDA_post_clean to check and verify the results of 1a.

---

## 🧠 Step 2 – Feature Engineering
Each row in `model_table.csv` corresponds to a **unique patient_id**.


| Category | Features | Description |
|-----------|-----------|-------------|
| **Demographics** | `PATIENT_AGE`, `PATIENT_GENDER`, `IS_AGE65PLUS`, `AGE_GE_12` | Derived from birth year and diagnosis/symptom onset |
| **Clinical risk** | `NUM_CONDITIONS`, `HAS_UNDERLYING`, `HIGH_RISK`, `ELIGIBLE` | Encodes comorbidity and eligibility logic |
| **Contraindications** | `CONTRAINDICATION_LEVEL` | 0–3 ordinal risk level |
| **Timing** | `DAYS_SYMPTOM_TO_DX` | Days between symptom onset and diagnosis |
| **Physician & Location** | `PHYSICIAN_TYPE`, `PHYSICIAN_STATE`, `LOCATION_TYPE`, `PHYS_TREAT_RATE` | Context of care and physician propensity |
| **Target** | `TARGET` | Whether the patient ever received Drug A |

### 🆕 New Features Added

In addition to the base columns described in the data dictionary, the following **engineered features** were created to improve model performance and interpretability:

| Feature | Description | Rationale |
|----------|--------------|------------|
| **DAYS_SYMPTOM_TO_DX** | Number of days between first symptom onset and Disease X diagnosis (−14 → 30 range, clipped) | Captures care-seeking and diagnostic delays, which strongly influence treatment likelihood. |
| **PHYS_TREAT_RATE** | Bayesian leave-one-out (LOO) physician-level treatment propensity using a weak Beta(0.125, 0.125) prior | Models physician behavioral tendency to prescribe Drug A while preventing target leakage. |
| **CONTRAINDICATION_LEVEL** | Ordinal feature (0–3) summarizing maximum contraindication severity per patient | Encodes potential safety constraints affecting treatment decisions. |
| **HIGH_RISK** | Composite flag (`IS_AGE65PLUS or HAS_UNDERLYING`) | Captures elevated risk status relevant for treatment eligibility. |
| **ELIGIBLE** | Composite flag (`AGE_GE_12 and HIGH_RISK`) | Simplifies model learning by pre-encoding clinical eligibility criteria. |


---

So, your new (non-trivial) features are:
- `DAYS_SYMPTOM_TO_DX`
- `PHYS_TREAT_RATE`
- `CONTRAINDICATION_LEVEL`
- `HIGH_RISK`
- `ELIGIBLE`

The first two (`DAYS_SYMPTOM_TO_DX` and `PHYS_TREAT_RATE`) are the **most innovative** — they demonstrate temporal and behavioral modeling, which aligns with data-driven business reasoning.

---


## 🤖 Step 3 – Modeling & Evaluation

| Model | ROC-AUC | PR-AUC | Acc | Prec | Rec | F1 | F2 |
|:------|:--------:|:------:|:---:|:----:|:---:|:--:|:--:|
| Logistic Regression | 0.74 | 0.33 | 0.70 | 0.30 | 0.63 | 0.41 | 0.52 |
| Random Forest | 0.75 | 0.34 | 0.71 | 0.31 | 0.63 | 0.41 | 0.52 |
| LightGBM | 0.76 | 0.35 | 0.70 | 0.30 | 0.67 | 0.41 | 0.53 |
| **XGBoost (final)** | **0.76 – 0.78** | **0.35 – 0.36** | **0.70** | **0.30** | **0.68** | **0.41** | **0.54** |

### 🏆 Final Model: XGBoost (`model_minimal.joblib`)
Chosen for its strong balance of **accuracy, recall, and interpretability**.  
- Robust to mixed categorical / numeric inputs via one-hot encoding  
- Handles **non-linear effects** and **imbalanced classes** using `scale_pos_weight`  
- Provides **SHAP-based explainability**, enabling feature-level insights

### 📈 Model Insights (SHAP Feature Contributions)
**Top positive influencers (increase treatment likelihood):**
- **`PATIENT_AGE`** ↑ — older patients more likely to receive Drug A (<80) 
- **`NUM_CONDITIONS`** ↑ — more comorbidities → higher likelihood  
- **`LOCATION_TYPE = office`** — in-person visits strongly associated with treatment  
- **`PHYSICIAN_TYPE = family / internal medicine`** — higher prescribing tendency  

**Top negative influencers (reduce treatment likelihood):**
- **`LOCATION_TYPE = telehealth` or `independent laboratory`** — lower likelihood of treatment initiation  
- **`younger patients` (< 40 yrs)** — less likely to be treated despite eligibility  
- **`high contraindication levels`** — safety constraints decreasing probability  

> *LightGBM produced comparable performance (ROC ≈ 0.76) but was ultimately not selected due to slightly higher variance in recall and less stable SHAP consistency.*

![SHAP Summary Plot](notebooks/artifacts/SHAP.png)
---

## 🩺 Step 4 – EMR Alert Simulation
Patients ranked by **predicted P(TREATED)** (ascending).  
Alerts target **least likely to be treated**.

| Coverage | Recall (untreated) | Precision (true untreated) | F2 |
|-----------|-------------------|-----------------------------|----|
| 60 % | 0.72 | 0.95 | 0.75 |
| 70 % | 0.81 | 0.93 | 0.83 |
| 90 % | 0.96 | 0.85 | 0.94 |

➡ Recommended coverage **60 – 70 %** to balance recall vs workload.
![SHAP Summary Plot](notebooks/artifacts/cutoff.png)

---

## 🚀 Step 5 – API Deployment (FastAPI + Docker)

### ⚙️ Setup and Local Run

#### 1️⃣ Create a virtual environment
```bash
python -m venv .venv_diseasex
source .venv_diseasex/bin/activate
pip install -r requirements.txt


1️⃣ Create a virtual environment

python -m venv .venv_diseasex
source .venv_diseasex/bin/activate
pip install -r requirements.txt

2️⃣ Train the model

python -m training.train --model xgb

Outputs:
	•	artifacts/final_pipe.joblib — trained model
	•	artifacts/metrics_baseline.json — evaluation metrics

⸻


### 🧠 Model Details

| **Component** | **Description** |
|----------------|-----------------|
| **Algorithm** | XGBoost / Random Forest / Logistic Regression (with balanced class weights) |
| **Core Features** | `DISEASEX_DT`, `PATIENT_AGE`, `PATIENT_GENDER`, `NUM_CONDITIONS`, `PHYSICIAN_TYPE`, `PHYSICIAN_STATE`, `LOCATION_TYPE` |
| **Extended Features** | `PHYS_TREAT_RATE` (physician propensity), `DAYS_SYMPTOM_TO_DX` (onset-to-diagnosis lag), and eligibility flags |
| **Target** | `TARGET = 1` if patient received Drug A; else 0 |
| **Metrics (5×2 CV)** | **ROC-AUC ≈ 0.75**, **PR-AUC ≈ 0.34**, **F2 ≈ 0.54**  *(favoring recall for under-treated patients)* |



⸻

🧩 Run the FastAPI Service

1️⃣ Start API locally

uvicorn app.main:app --reload --port 8000

2️⃣ Example request

Endpoint: POST /predict

Input JSON:

{
  "DISEASEX_DT": "2022-06-14",
  "PATIENT_AGE": 58,
  "PATIENT_GENDER": "f",
  "NUM_CONDITIONS": 2,
  "PHYSICIAN_TYPE": "family medicine",
  "PHYSICIAN_STATE": "TX",
  "LOCATION_TYPE": "office",
  "threshold": 0.60
}

Response:

{
  "P_TREATED": 0.7726,
  "P_UNTREATED": 0.2274,
  "threshold": 0.40,
  "message": "✅ Likely to be treated"
}
⸻

## 🚀 Step 5 Containerization: Docker Deployment

1️⃣ Build the image

docker build -t diseasex-alert .

2️⃣ Run the container

docker run --rm -p 8000:8000 diseasex-alert

3️⃣ Test via curl

curl -s http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "DISEASEX_DT":"2024-05-10",
    "PATIENT_AGE":65,
    "PATIENT_GENDER":"M",
    "NUM_CONDITIONS":2,
    "PHYSICIAN_TYPE":"family medicine",
    "PHYSICIAN_STATE":"TX",
    "LOCATION_TYPE":"hospital",
    "threshold":0.45
  }'
  
  Expected Response: {"P_TREATED":0.68,"P_UNTREATED":0.32,"threshold":0.45,"message":"✅ Likely to be treated"}


⸻

🧪 Running Tests

pytest -q

Expected Output

5 passed, 0 failed


⸻

## 🚀 Step 6  Software Architecture


### 🧩 Components Overview

| **Layer** | **Description** |
|------------|-----------------|
| **Training Layer** | Reads and cleans EMR data, engineers features, builds `model_table.csv`, and trains the ML model. |
| **Model Artifact** | Trained model serialized with `joblib` and stored under `artifacts/` for inference reuse. |
| **Inference Layer** | Lightweight FastAPI REST service that loads the model and predicts treatment likelihood for new patients. |
| **Alert Layer** | Applies thresholds to flag patients with low predicted treatment probability (potential under-treatment). |
| **Containerization** | Dockerized deployment ensuring full reproducibility and environment portability. |

---

### ⚙️ Error Handling & Logging

- Central FastAPI middleware logs every API request with timestamps.  
- All inference steps are wrapped in `try/except` blocks to gracefully handle missing or malformed data.  
- Logging includes both request payloads and model predictions for traceability.  
- Logs are automatically written to:  
  📄 `artifacts/api_requests.log`



⸻

7. Version Control Strategy

## 🧭 Version Control Strategy

| **Element** | **Strategy** |
|--------------|--------------|
| **Code** | Follow a clear Git branching model: `main` (stable), `dev` (active development), and `feature/*` (per-feature branches). |
| **Model Artifacts** | Versioned under `artifacts/` (e.g., `model_minimal_v1.joblib`), tagged with date or commit hash for reproducibility. |
| **Data** | Maintain versioned processed data (e.g., `data/processed/model_table_v1.csv`) while ensuring schema consistency. |
| **Merging** | Use Pull Requests (PRs) for all merges; apply *squash commits* to keep history concise and meaningful. |
| **Conflict Resolution** | Rebase local changes onto `main` before merging and resolve conflicts using VSCode merge tools or CLI (e.g., `git mergetool`). |



⸻

## 🧩 Design Choices


- **XGBoost & Random Forest Models:**  
  Chosen for strong performance on structured clinical data, handling mixed feature types and class imbalance effectively.

- **FastAPI Framework:**  
  Asynchronous, production-grade REST API with automatic OpenAPI docs (`/docs`) and native async I/O for high throughput.

- **Dockerized Environment:**  
  Guarantees consistent runtime across local, staging, and cloud deployments with minimal configuration drift.

- **Modular Architecture:**  
  Training, inference, and API layers are decoupled, improving maintainability and enabling independent updates or scaling.

- **Configurable Alert Threshold:**  
  Sensitivity threshold for patient alerts can be adjusted dynamically through the environment variable `ALERT_THRESHOLD`.

---

### ⚙️ Scalability Notes

- **Batch & Streaming Inference:**  
  Supports batch predictions or can be extended to real-time streaming (e.g., Kafka, AWS Kinesis) for hospital-scale data flow.  

- **Model Registry Integration:**  
  Models and metrics can be tracked and versioned via MLflow or DVC for traceability and rollback.  

- **Horizontal API Scaling:**  
  Containerized FastAPI service can be deployed under load balancers or Kubernetes for high-concurrency environments.  

- **GPU Acceleration:**  
  XGBoost automatically leverages NVIDIA GPUs (`tree_method='gpu_hist'`) when available, reducing training time on large EMR datasets.  

- **Monitoring & Drift Detection:**  
  API logs (`artifacts/api_requests.log`) can be integrated with Prometheus/Grafana or ELK Stack to track performance and detect model drift.  

---

### 🔁 MLOps Integration

- **CI/CD Pipeline:**  
  Continuous integration (via GitHub Actions or GitLab CI) automatically tests, builds, and deploys new model versions.  

- **Automated Retraining:**  
  Scheduled retraining workflows (e.g., Airflow or Prefect) refresh the model as new EMR data is ingested, maintaining predictive stability.  

- **Model Validation Gates:**  
  Each new version must pass ROC-AUC and F2 score thresholds before deployment, preventing performance regression.  

- **Artifact Versioning:**  
  Models, metrics, and preprocessing schemas are version-controlled, ensuring full lineage and rollback capability.  

- **Environment Parity:**  
  Docker and `requirements.txt` ensure identical configurations across dev, staging, and production environments.  
  
### 🔁 MLOps Integration

- **CI/CD Pipeline:**  
  Continuous integration (via GitHub Actions or GitLab CI) automatically tests, builds, and deploys new model versions.  

- **Automated Retraining:**  
  Scheduled retraining workflows (e.g., Airflow or Prefect) refresh the model as new EMR data is ingested, maintaining predictive stability.  

- **Model Validation Gates:**  
  Each new version must pass ROC-AUC and F2 score thresholds before deployment, preventing performance regression.  

- **Artifact Versioning:**  
  Models, metrics, and preprocessing schemas are version-controlled, ensuring full lineage and rollback capability.  

- **Environment Parity:**  
  Docker and `requirements.txt` ensure identical configurations across dev, staging, and production environments.  

---


⸻

🏁 Deliverables Summary

✅ model_table.csv included under data/processed/
✅ ML model saved as artifacts/model_minimal.joblib
✅ REST API (FastAPI) running locally or in Docker
✅ Complete README, test coverage, and reproducible environment

⸻

📜 License

This project is submitted as part of the Disease X Treatment Alert Coding Exercise for internal evaluation.
All patient data are synthetic and non-identifiable.

