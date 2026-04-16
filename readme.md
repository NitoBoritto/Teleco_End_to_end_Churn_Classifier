# Telco Customer Churn Classifier
### End-to-End Machine Learning System | FastAPI · Streamlit · MLflow · Docker · Azure

---

## Overview

A **production-grade binary classification system** that predicts customer churn for a telecommunications provider. The project goes beyond model development — it covers the full MLOps lifecycle: automated data validation, feature engineering, hyperparameter optimization, experiment tracking, model serialization, REST API serving, and containerized cloud deployment.

The system accepts 19 customer attributes and returns a **churn prediction with a calibrated probability score** in real time.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                           │
│                                                                 │
│  Raw CSV ──► Great Expectations ──► Preprocessing ──► Feature   │
│                 (Validation)           Pipeline        Eng.     │
│                                                        │        │
│                                           Optuna Tuning◄────┘   │
│                                                │                │
│                                     LogisticRegression          │
│                                                │                │
│                                        MLflow Logging           │
│                                    (metrics + artifacts)        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        SERVING PHASE                            │
│                                                                 │
│  Streamlit UI ──► POST /predict ──► FastAPI ──► Inference       │
│  (ui.py)            (HTTP)          (main.py)   Pipeline        │
│                                                    │            │
│                                           MLflow Model Load     │
│                                         + Data Validation       │
│                                         + Feature Ordering      │
│                                                │                │
│                                    {"prediction", "probability"}│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT PHASE                           │
│                                                                 │
│  Dockerfile ──► Docker Image ──► Azure Container Registry       │
│  (start.sh)                     ──► Azure Web App / ACI         │
│                                                                 │
│  GitHub Actions CI/CD pipeline on push to main                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Teleco_End_to_end_Churn_Classifier/
│
├── .github/
│   └── workflows/          # GitHub Actions CI/CD pipeline
│
├── notebooks/              # EDA, model selection, and experimentation
│
├── scripts/
│   └── run_pipeline.py     # Orchestrator: end-to-end training pipeline
│
├── src/
│   ├── app/
│   │   ├── main.py         # FastAPI application and /predict endpoint
│   │   └── ui.py           # Streamlit frontend portal
│   │
│   ├── data/
│   │   ├── load_data.py    # CSV ingestion with path validation
│   │   └── preprocess.py   # Cleaning: types, nulls, target encoding
│   │
│   ├── features/
│   │   └── build_features.py   # Custom sklearn transformers + ColumnTransformer pipeline
│   │
│   ├── models/
│   │   └── tune.py             # Optuna hyperparameter optimization (recall-optimized)
│   │
│   ├── serving/
│   │   └── inference.py        # MLflow model loading + prediction logic
│   │
│   └── utils/
│       └── validate_data.py    # Great Expectations data quality suite
│
├── dockerfile              # Multi-service container (FastAPI + Streamlit)
├── start.sh                # Container entrypoint script
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## ML Pipeline — Step by Step

### 1. Data Validation — `Great Expectations`
Before any training begins, the raw data passes through a validation suite that enforces:

- **Schema checks** — all 19 required columns must be present and non-null where critical
- **Business logic** — categorical features are validated against their allowed value sets (e.g., `Contract ∈ {Month-to-month, One year, Two year}`)
- **Numeric range constraints** — `tenure ∈ [0, 120]`, `MonthlyCharges ∈ [0, 200]`
- **Data consistency** — `TotalCharges ≥ MonthlyCharges` holds for ≥95% of records

Validation results are logged to MLflow as a `data_quality_pass` metric. A failed validation halts the pipeline immediately with a structured error report.

---

### 2. Preprocessing — `src/data/preprocess.py`
Lightweight but robust cleaning layer:
- Column name whitespace trimming
- `customerID` removal (identifier leakage prevention)
- `TotalCharges` coerced to numeric (originally stored as string in the dataset)
- `SeniorCitizen` null-filled and cast to `int`
- Numeric nulls filled with `0`; categorical nulls left for the encoder to handle

---

### 3. Feature Engineering — `src/features/build_features.py`

A fully custom `sklearn`-compatible pipeline with three custom transformers:

| Transformer | Logic |
|---|---|
| `custom_binary_encoder` | Maps `Yes/No` and `Male/Female` → `1/0`. Consistent across train and serve. |
| `boolean_encoder` | Casts any boolean columns to integer |
| `aggregate_drop_multicollinear` | Aggregates 6 `No internet service` OHE columns → single `No_internet_service` flag; drops `InternetService_No`, `PhoneService`, `MonthlyCharges` to reduce collinearity |

The full `ColumnTransformer` stack:
```
binary_cols       → custom_binary_encoder
bool_cols         → boolean_encoder
multi_cols (3+)   → OneHotEncoder(drop='first')
numeric_cols      → RobustScaler
```

All transformers implement `get_feature_names_out()` for full sklearn pipeline compatibility and feature name propagation.

---

### 4. Hyperparameter Optimization — `Optuna`

A Bayesian search over `C` (regularization strength) for Logistic Regression:
- **Search space**: `C ∈ [0.001, 1]` (log scale)
- **CV strategy**: `StratifiedKFold(n_splits=5)` to preserve class distribution
- **Objective metric**: `Recall` — deliberately chosen to minimize false negatives (missed churners are more costly than false alarms in retention strategy)
- **Trials**: 20

The entire feature pipeline is cloned fresh inside each trial to prevent data leakage across folds.

---

### 5. Model Training & Experiment Tracking — `MLflow`

The final pipeline is assembled as a single sklearn `Pipeline` object:
```
Pipeline([
    ('preprocessor',   ColumnTransformer),
    ('multicollinear', aggregate_drop_multicollinear),
    ('lgr',            LogisticRegression(**best_params, class_weight='balanced'))
])
```

`class_weight='balanced'` addresses the class imbalance inherent in churn datasets (typically ~20-30% churn rate).

Every training run logs to MLflow:
- `precision`, `recall`, `f1`, `roc_auc`
- `train_time`, `pred_time`
- `data_quality_pass`
- `threshold` (configurable, default `0.35` — tuned to bias toward recall)
- Feature columns JSON (critical for serving consistency)
- The full serialized pipeline via `mlflow.sklearn.log_model()`

---

### 6. Inference & Serving

**`inference.py`** is the core serving contract:
1. Loads the MLflow-serialized pipeline from `/app/model` (Docker) with local fallback
2. Reconstructs a single-row DataFrame from the API input dict
3. Injects a dummy `customerID` for validation compatibility
4. Runs the full Great Expectations validation suite on live input
5. Enforces strict feature column ordering (19 features) to prevent positional bugs
6. Returns `{"prediction": str, "probability": float}`

**`main.py`** (FastAPI):
- Pydantic model `customerdata` validates all 19 input fields at the API boundary before they reach the inference layer
- `POST /predict` — main prediction endpoint
- `GET /` — health check endpoint (for Azure load balancer probes)
- Auto-generated OpenAPI docs at `/docs`

**`ui.py`** (Streamlit):
- Dark-themed portal with dynamic form logic (internet-dependent service fields disable automatically when `No internet service` is selected)
- Sends `POST` to `http://127.0.0.1:8000/predict`
- Renders a styled prediction card with confidence score and a progress bar
- Displays contextual retention strategy recommendation

---

## Input Features

| Feature | Type | Description |
|---|---|---|
| `gender` | categorical | Male / Female |
| `SeniorCitizen` | binary int | 0 or 1 |
| `Partner` | categorical | Yes / No |
| `Dependents` | categorical | Yes / No |
| `tenure` | int | Months with company |
| `PhoneService` | categorical | Yes / No |
| `MultipleLines` | categorical | Yes / No / No phone service |
| `InternetService` | categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | categorical | Yes / No / No internet service |
| `OnlineBackup` | categorical | Yes / No / No internet service |
| `DeviceProtection` | categorical | Yes / No / No internet service |
| `TechSupport` | categorical | Yes / No / No internet service |
| `StreamingTV` | categorical | Yes / No / No internet service |
| `StreamingMovies` | categorical | Yes / No / No internet service |
| `Contract` | categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | categorical | Yes / No |
| `PaymentMethod` | categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | float | Monthly bill in USD |
| `TotalCharges` | float | Cumulative charges in USD |

---

## API Reference

### Health Check
```http
GET /
```
```json
{ "status": "ok" }
```

### Predict Churn
```http
POST /predict
Content-Type: application/json
```

**Request body** (example):
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 24,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 79.85,
  "TotalCharges": 1918.40
}
```

**Response**:
```json
{
  "prediction": "Likely to churn",
  "probability": 73.42
}
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development

**1. Clone and install dependencies**
```bash
git clone https://github.com/NitoBoritto/Teleco_End_to_end_Churn_Classifier.git
cd Teleco_End_to_end_Churn_Classifier
pip install -r requirements.txt
```

**2. Run the training pipeline**
```bash
python scripts/run_pipeline.py \
  --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --target Churn \
  --threshold 0.35
```

**3. Start the FastAPI backend**
```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

**4. Launch the Streamlit UI** (in a separate terminal)
```bash
streamlit run src/app/ui.py
```

**5. View experiment tracking**
```bash
mlflow ui
```
Navigate to `http://localhost:5000`

---

### Docker

**Build the image**
```bash
docker build -t telco-churn-classifier .
```

**Run the container**
```bash
docker run -p 8000:8000 -p 8501:8501 telco-churn-classifier
```

The container runs both FastAPI (port 8000) and Streamlit (port 8501) via `start.sh`.

---

## Deployment — Microsoft Azure

The application is containerized and deployed to Azure via GitHub Actions CI/CD. On every push to `main`, the workflow:
1. Builds the Docker image
2. Deploys to Azure Web App / Azure Container Instances (ACI)

---

## Technology Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Data Validation | Great Expectations |
| API | FastAPI + Pydantic |
| Frontend | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | Microsoft Azure |

---

## Dataset

[IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features, binary churn target.

---

## Author

**NitoBoritto** — Aspiring Full-Stack Data Scientist
