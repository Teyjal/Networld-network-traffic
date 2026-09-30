# NetWorld Backend — FastAPI Foundation

Backend API for **NetWorld — A World-Model AI That Forecasts Cyberattacks** (SIH 2026).

---

## 📁 Directory Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── traffic.py
│   │   ├── forecast.py
│   │   ├── explain.py
│   │   ├── whatif.py
│   │   └── validation.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── inference.py
│   │   ├── mitre_mapper.py
│   │   ├── shap_service.py
│   │   └── counterfactual.py
│   └── model/
│       ├── __init__.py
│       └── loader.py
├── uploads/                       # Temporary storage for ingested traffic CSVs
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup & Requirements

### Prerequisites
- Python 3.9+
- React Vite frontend running on `http://localhost:5173`

---

## 🚀 Installation & Running

### 1. Create and Activate Virtual Environment
From the project root directory:

```bash
# Create virtual environment
python3 -m venv backend/venv

# Activate environment (macOS/Linux)
source backend/venv/bin/activate

# On Windows PowerShell:
# backend\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Start the FastAPI Server
Run Uvicorn from inside the `backend/` directory:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🔍 Verification & Documentation

- **Root API Info**: [`http://localhost:8000/`](http://localhost:8000/)
- **Health Check**: [`http://localhost:8000/health`](http://localhost:8000/health)
- **Interactive OpenAPI / Swagger Documentation**: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **ReDoc Documentation**: [`http://localhost:8000/redoc`](http://localhost:8000/redoc)

---

## 📊 Endpoints & Usage

### 1. Traffic Ingestion (`POST /api/traffic/upload`)
Ingests and validates a CSV dataset containing the 36 NetWorld CIC-IDS2018 model features.

```bash
curl -X POST "http://localhost:8000/api/traffic/upload" \
  -H "accept: application/json" \
  -F "file=@data/processed/networld_combined_features.csv"
```

### 2. K-Step Temporal Horizon Forecast (`POST /api/forecast`)
Executes PyTorch LSTM temporal forecasting with a configurable $K$-step lookahead horizon (default $K=5$).

```bash
curl -X POST "http://localhost:8000/api/forecast?horizon=5" \
  -H "accept: application/json" \
  -F "file=@data/processed/networld_combined_features.csv"
```

### 3. Explainability & Feature Attributions (`POST /api/explain`)
Computes normalized feature importance attributions and risk contribution directions (`increases_risk` / `decreases_risk`) using PyTorch Integrated Gradients.

```bash
curl -X POST "http://localhost:8000/api/explain?top_n=10" \
  -H "accept: application/json" \
  -F "file=@data/processed/networld_combined_features.csv"
```

### 4. What-If Defender Simulator (`POST /api/whatif`)
Simulates defensive interventions (`close_port`, `block_ip`, `isolate_host`, `block_protocol`, `rate_limit`) on a clean copy of dataset flows and compares baseline vs counterfactual risk.

```bash
curl -X POST "http://localhost:8000/api/whatif?action=close_port&target_value=80" \
  -H "accept: application/json" \
  -F "file=@data/processed/networld_combined_features.csv"
```

### 5. Model Validation & Benchmark Metrics (`GET /api/validation`)
Exposes empirical evaluation metrics calculated on the 311,562 test split samples comparing NetWorld Temporal LSTM against the Logistic Regression baseline.

```bash
curl -X GET "http://localhost:8000/api/validation"
```

#### Example Validation Metrics Response
```json
{
  "status": "available",
  "dataset_info": {
    "training_dataset": "CIC-IDS2018",
    "evaluation_split": "Temporal Holdout Test Split (20%)",
    "total_test_samples": 311562,
    "sequence_length": 20,
    "feature_count": 36,
    "benign_samples": 186637,
    "attack_samples": 124925
  },
  "models": {
    "networld_lstm": {
      "model_name": "NetWorld Temporal LSTM",
      "accuracy": 0.6935,
      "precision": 0.6874,
      "recall": 0.4319,
      "f1": 0.5305,
      "false_positive_rate": 0.1315,
      "confusion_matrix": {
        "tn": 162102,
        "fp": 24535,
        "fn": 70968,
        "tp": 53957,
        "matrix": [[162102, 24535], [70968, 53957]]
      }
    },
    "logistic_regression": {
      "model_name": "Logistic Regression Baseline",
      "accuracy": 0.5601,
      "precision": 0.4712,
      "recall": 0.7942,
      "f1": 0.5915,
      "false_positive_rate": 0.5966,
      "confusion_matrix": {
        "tn": 75294,
        "fp": 111343,
        "fn": 25708,
        "tp": 99217,
        "matrix": [[75294, 111343], [25708, 99217]]
      }
    }
  }
}
```
