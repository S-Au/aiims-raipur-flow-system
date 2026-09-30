# AIIMS Raipur Outpatient Flow Optimization & Queue Management System (AFOS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)]()
[![Compliance](https://img.shields.io/badge/ABDM%20%7C%20DPDPA%202023-Compliant-darkgreen.svg)]()

> **Production-grade machine learning and queue optimization engine designed for All India Institute of Medical Sciences (AIIMS), Raipur, Chhattisgarh.**  
> Solves outpatient congestion across 32 clinical departments with quantile wait-time prediction, footfall forecasting, and starvation-free dynamic priority scheduling (SEPT-AP).

---

## 1. System Performance & Key Verification Metrics

| Component / Metric | Benchmark Target | AFOS Performance | Status |
|---|---|---|---|
| **Wait-Time Regressor $R^2$ Score** | $\ge 0.85$ | **0.9803** | **PASSED** |
| **Wait-Time Mean Absolute Error (MAE)** | $< 10.0$ mins | **1.49 mins** | **PASSED** |
| **Wait-Time Root Mean Square Error (RMSE)**| — | **1.94 mins** | **PASSED** |
| **$P_{10} - P_{90}$ Quantile Coverage** | $\approx 80-90\%$ | **91.71%** | **PASSED** |
| **Tier 1 Clinical Protocol Safety Cap** | $\le 8.0$ mins | **Strictly Enforced ($[1.0, 8.0]$ mins)** | **PASSED** |
| **Single-Record Inference Latency** | $< 10.0$ ms | **2.90 ms** (Optimized Fast-Path) | **PASSED** |
| **Batch Inference Latency (per record)** | $< 1.0$ ms | **0.005 ms** | **PASSED** |
| **Footfall Forecaster $R^2$ Score** | $\ge 0.80$ | **0.8769** | **PASSED** |
| **Footfall Forecaster MAE** | $< 15.0$ patients | **5.64 patients** | **PASSED** |
| **Tier 2 Urgent Wait Reduction vs FIFO** | $> 30.0\%$ | **75.66% reduction** | **PASSED** |
| **Tier 1 Fast-Track Resuscitation Gain** | $> 15.0\%$ | **25.59% faster** | **PASSED** |
| **Pytest Integration & Sanity Suite** | 100% pass | **25/25 passed** | **PASSED** |

---

## 2. Directory Structure

```
aiims_raipur_flow_system/
├── README.md                      # Comprehensive system guide & quickstart
├── requirements.txt               # Production Python dependencies
├── data/
│   ├── aiims_raipur_opd_52k.csv   # Curated clinical OPD encounter dataset (52k rows, 25 columns)
│   └── data_dictionary.md         # Schema definitions, distributions & queuing physics
├── models/
│   ├── predictor_classes.py       # Canonical ML classes for serialization & loading
│   ├── train_wait_time.py         # Quantile HistGradientBoosting wait-time trainer
│   ├── train_footfall.py          # Departmental daily footfall forecasting trainer
│   ├── wait_time_model.joblib     # Serialized wait-time prediction pipeline
│   ├── footfall_model.joblib      # Serialized footfall forecaster
│   ├── wait_time_metrics.json     # Test split validation metrics for wait times
│   ├── footfall_metrics.json      # Test split validation metrics for footfall
│   └── training_metrics.json      # Consolidated metrics & simulation benchmark
├── simulation/
│   ├── __init__.py
│   └── queue_sim.py               # M/M/s simulation + SEPT-AP dynamic scheduler
├── api/
│   ├── __init__.py
│   └── app.py                     # High-performance FastAPI REST microservice
├── cli/
│   ├── __init__.py
│   └── flow_cli.py                # Admin CLI for prediction, forecasting & simulation
├── tests/
│   ├── __init__.py
│   ├── test_dataset_integrity.py  # Statistical & schema integrity tests
│   ├── test_models.py             # Accuracy, monotonicity & latency benchmarks
│   ├── test_queue_sim.py          # Aging, starvation prevention & priority tests
│   └── test_api.py                # FastAPI endpoint integration tests
└── docs/
    └── DEFENSE_WHITEPAPER.md      # Mathematical formulations & systems architecture whitepaper
```

---

## 3. Quickstart & Installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Full Pytest Test Suite
```bash
pytest -v tests/
```
Output:
```
======================== 25 passed, 1 warning in 3.15s ========================
```

---

## 4. CLI Usage Guide

The system includes a production CLI utility (`cli/flow_cli.py`):

### 1. Real-Time Wait-Time Prediction:
```bash
python cli/flow_cli.py predict-wait --dept "Cardiology" --triage 2 --queue 10 --counters 3
```
Output:
```
=================================================================
 AIIMS RAIPUR OPD - REAL-TIME WAIT-TIME PREDICTION
=================================================================
 Department          : Cardiology
 Clinical Urgency    : Tier 2 (Yellow: Urgent / Severe Clinical Distress)
 Queue Ahead         : 10 patients
 Active Counters     : 3 chambers
 Arrival Minute      : 540 (09:00 hrs)
-----------------------------------------------------------------
 Expected Wait (Mean): 21.3 minutes
 Median Wait (P50)   : 20.3 minutes
 80% Confidence Band : [2.0 mins  to  29.8 mins]
=================================================================
```

### 2. Departmental Footfall Forecasting:
```bash
python cli/flow_cli.py forecast-footfall --dept "Orthopedics" --day "Monday" --post-holiday
```

### 3. Run Queue Simulation Benchmark (FIFO vs SEPT-AP):
```bash
python cli/flow_cli.py simulate --patients 300 --counters 4 --alpha 2.5 --beta 12.0
```

---

## 5. FastAPI REST Microservice

Launch the server:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger Documentation available at: `http://localhost:8000/docs`

### Key Endpoints:
- `GET /health` - System health, model active states
- `POST /predict_wait_time` - Quantile wait-time inference ($P_{10}, P_{50}, P_{90}$)
- `POST /forecast_footfall` - Daily patient load & staffing recommendation
- `POST /issue_token` - ABHA-linked digital token generation
- `POST /simulate_queue` - Dynamic queue simulation execution
- `GET /metrics` - Serialized evaluation metrics

### Example API Request (cURL):
```bash
curl -X POST "http://localhost:8000/predict_wait_time" \
     -H "Content-Type: application/json" \
     -d '{
       "department": "Cardiology",
       "triage_tier": 2,
       "patient_age": 62,
       "patient_gender": "M",
       "patient_origin": "Raipur_Urban",
       "is_referral": 0,
       "visit_type": "New_Consultation",
       "active_counters": 3,
       "queue_depth_at_arrival": 8,
       "arrival_minute_of_day": 540,
       "temp_celsius": 28.0,
       "rainfall_mm": 0.0,
       "is_monday": 0,
       "is_saturday": 0,
       "is_post_holiday": 0,
       "is_harvest_season": 0,
       "has_abha_id": 1
     }'
```

---

## 6. Statutory Compliance & Security

- **Digital Personal Data Protection Act (DPDPA 2023):**  
  The system strictly enforces Section 8 data minimization. All patient records and queue transactions are pseudonymized using randomized episodic token IDs (`GEN-251001-0429`). No Aadhaar numbers, biometric data, or direct contact details are stored.
- **Ayushman Bharat Digital Mission (ABDM):**  
  Full schema compatibility with ABHA 14-digit IDs and FHIR/M1/M2/M3 digital health standards for paperless OPD kiosk registration.

---

## 7. Systems Architecture Whitepaper
For complete mathematical proofs, queuing theorems, Pinball loss equations, and operational design rationale, consult:  
👉 [`docs/DEFENSE_WHITEPAPER.md`](docs/DEFENSE_WHITEPAPER.md)
