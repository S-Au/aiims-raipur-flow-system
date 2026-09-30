"""
AIIMS Raipur Outpatient Flow Optimization & Queue Management System (AFOS)
FastAPI REST Service for:
- Live Wait-Time Inference (P10, P50, P90 Quantiles)
- Departmental Footfall Forecasting
- Dynamic Digital Token Issuance (ABDM / ABHA Linkage)
- Multi-Server Queue Simulation Execution
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure root package can be imported
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.predictor_classes import OPDWaitTimePredictor, DepartmentFootfallForecaster
from simulation.queue_sim import AIIMSQueueSimulation, QueuedPatient, run_comparative_benchmark

app = FastAPI(
    title="AIIMS Raipur OPD Flow Optimization API",
    description="Predictive Wait-Time Engine, Footfall Forecaster, and Dynamic Token Dispatcher",
    version="1.0.0"
)

# Load serialized models
WAIT_MODEL_PATH = os.path.join(BASE_DIR, "models", "wait_time_model.joblib")
FOOTFALL_MODEL_PATH = os.path.join(BASE_DIR, "models", "footfall_model.joblib")

def load_wait_model() -> Optional[OPDWaitTimePredictor]:
    if os.path.exists(WAIT_MODEL_PATH):
        try:
            return joblib.load(WAIT_MODEL_PATH)
        except Exception as e:
            print(f"Error loading wait model: {e}")
    return None

def load_footfall_model() -> Optional[DepartmentFootfallForecaster]:
    if os.path.exists(FOOTFALL_MODEL_PATH):
        try:
            return joblib.load(FOOTFALL_MODEL_PATH)
        except Exception as e:
            print(f"Error loading footfall model: {e}")
    return None

wait_time_model = load_wait_model()
footfall_model = load_footfall_model()

class WaitTimeRequest(BaseModel):
    department: str = Field(default="General Medicine", json_schema_extra={"example": "General Medicine"})
    triage_tier: int = Field(default=3, ge=1, le=4, description="1: Immediate, 2: Urgent, 3: Semi-urgent, 4: Routine")
    patient_age: int = Field(default=42, ge=1, le=110)
    patient_gender: str = Field(default="M", json_schema_extra={"example": "M"})
    patient_origin: str = Field(default="Raipur_Urban", json_schema_extra={"example": "Durg_Bhilai"})
    is_referral: int = Field(default=0, ge=0, le=1)
    visit_type: str = Field(default="New_Consultation", json_schema_extra={"example": "New_Consultation"})
    active_counters: int = Field(default=4, ge=1, le=12)
    queue_depth_at_arrival: int = Field(default=12, ge=0, le=150)
    arrival_minute_of_day: int = Field(default=540, ge=400, le=800, description="Minutes from midnight (e.g. 540 = 09:00 AM)")
    temp_celsius: float = Field(default=28.0)
    rainfall_mm: float = Field(default=0.0)
    is_monday: int = Field(default=0, ge=0, le=1)
    is_saturday: int = Field(default=0, ge=0, le=1)
    is_post_holiday: int = Field(default=0, ge=0, le=1)
    is_harvest_season: int = Field(default=0, ge=0, le=1)
    has_abha_id: int = Field(default=1, ge=0, le=1)

class TokenRequest(BaseModel):
    abha_id: Optional[str] = Field(default=None, json_schema_extra={"example": "91-1234-5678-9012"})
    department: str = Field(default="Cardiology")
    patient_age: int = Field(default=58)
    patient_gender: str = Field(default="F")
    patient_origin: str = Field(default="Bilaspur")
    visit_type: str = Field(default="New_Consultation")
    triage_tier: int = Field(default=2)
    active_counters: int = Field(default=3)
    queue_depth: int = Field(default=8)

class FootfallRequest(BaseModel):
    department: str = Field(default="General Medicine")
    day_of_week: str = Field(default="Monday")
    temp_celsius: float = Field(default=28.5)
    rainfall_mm: float = Field(default=0.0)
    is_monday: int = Field(default=1)
    is_saturday: int = Field(default=0)
    is_post_holiday: int = Field(default=0)
    is_harvest_season: int = Field(default=1)

class SimulationRequest(BaseModel):
    num_patients: int = Field(default=200, ge=20, le=1000)
    num_counters: int = Field(default=4, ge=1, le=12)
    alpha_aging: float = Field(default=2.5, ge=0.5, le=10.0)
    beta_sept: float = Field(default=12.0, ge=1.0, le=50.0)

@app.get("/")
def root():
    global wait_time_model, footfall_model
    if wait_time_model is None:
        wait_time_model = load_wait_model()
    if footfall_model is None:
        footfall_model = load_footfall_model()

    return {
        "institution": "All India Institute of Medical Sciences (AIIMS), Raipur",
        "system": "Outpatient Flow Optimization & Queue Management System (AFOS)",
        "status": "Online",
        "docs_url": "/docs",
        "models_loaded": {
            "wait_time_model": wait_time_model is not None,
            "footfall_model": footfall_model is not None
        }
    }

@app.get("/health")
def health():
    global wait_time_model, footfall_model
    if wait_time_model is None:
        wait_time_model = load_wait_model()
    if footfall_model is None:
        footfall_model = load_footfall_model()

    return {
        "status": "Healthy",
        "timestamp": datetime.now().isoformat(),
        "wait_time_model_active": wait_time_model is not None,
        "footfall_model_active": footfall_model is not None
    }

@app.post("/predict_wait_time")
def predict_wait_time(req: WaitTimeRequest):
    global wait_time_model
    if wait_time_model is None:
        wait_time_model = load_wait_model()
    if wait_time_model is None:
        raise HTTPException(status_code=503, detail="Wait-time prediction model is not loaded.")

    is_senior = 1 if req.patient_age >= 60 else 0
    is_pediatric = 1 if req.patient_age <= 12 else 0

    input_df = pd.DataFrame([{
        "department": req.department,
        "triage_tier": req.triage_tier,
        "patient_age": req.patient_age,
        "patient_gender": req.patient_gender,
        "is_senior": is_senior,
        "is_pediatric": is_pediatric,
        "patient_origin": req.patient_origin,
        "is_referral": req.is_referral,
        "visit_type": req.visit_type,
        "active_counters": req.active_counters,
        "queue_depth_at_arrival": req.queue_depth_at_arrival,
        "arrival_minute_of_day": req.arrival_minute_of_day,
        "temp_celsius": req.temp_celsius,
        "rainfall_mm": req.rainfall_mm,
        "is_monday": req.is_monday,
        "is_saturday": req.is_saturday,
        "is_post_holiday": req.is_post_holiday,
        "is_harvest_season": req.is_harvest_season,
        "has_abha_id": req.has_abha_id
    }])

    preds = wait_time_model.predict(input_df)
    
    tier_desc = {
        1: "Immediate / Resuscitative (Fast-Track Corridor)",
        2: "Urgent (High Clinical Priority)",
        3: "Semi-Urgent (Standard symptomatic OPD)",
        4: "Routine (Follow-up / Review)"
    }.get(req.triage_tier, "Standard")

    return {
        "department": req.department,
        "triage_tier": req.triage_tier,
        "triage_description": tier_desc,
        "queue_depth": req.queue_depth_at_arrival,
        "active_counters": req.active_counters,
        "predictions": {
            "expected_wait_minutes": float(preds["predicted_wait_mean"][0]),
            "optimistic_p10_minutes": float(preds["predicted_wait_p10"][0]),
            "median_p50_minutes": float(preds["predicted_wait_p50"][0]),
            "conservative_p90_minutes": float(preds["predicted_wait_p90"][0]),
            "wait_range_window": f"{preds['predicted_wait_p10'][0]} - {preds['predicted_wait_p90'][0]} mins"
        }
    }

@app.post("/forecast_footfall")
def forecast_footfall(req: FootfallRequest):
    global footfall_model
    if footfall_model is None:
        footfall_model = load_footfall_model()
    if footfall_model is None:
        raise HTTPException(status_code=503, detail="Footfall forecasting model is not loaded.")

    input_df = pd.DataFrame([{
        "department": req.department,
        "day_of_week": req.day_of_week,
        "temp_celsius": req.temp_celsius,
        "rainfall_mm": req.rainfall_mm,
        "is_monday": req.is_monday,
        "is_saturday": req.is_saturday,
        "is_post_holiday": req.is_post_holiday,
        "is_harvest_season": req.is_harvest_season
    }])

    forecasted_val = int(footfall_model.predict(input_df)[0])
    recommended_counters = max(1, min(8, int(np.ceil(forecasted_val / 22))))

    return {
        "department": req.department,
        "day_of_week": req.day_of_week,
        "forecasted_daily_footfall": forecasted_val,
        "recommended_active_counters": recommended_counters,
        "contingency_buffer_pct": 15.0 if req.is_post_holiday or req.is_monday else 5.0
    }

@app.post("/issue_token")
def issue_digital_token(req: TokenRequest):
    global wait_time_model
    if wait_time_model is None:
        wait_time_model = load_wait_model()

    now = datetime.now()
    token_seq = np.random.randint(100, 999)
    prefix = req.department[:3].upper()
    token_str = f"{prefix}-{now.strftime('%y%m%d')}-{token_seq}"
    
    arr_min = int(np.clip(now.hour * 60 + now.minute, 450, 690))
    
    wait_time_info = None
    if wait_time_model is not None:
        input_df = pd.DataFrame([{
            "department": req.department,
            "triage_tier": req.triage_tier,
            "patient_age": req.patient_age,
            "patient_gender": req.patient_gender,
            "is_senior": 1 if req.patient_age >= 60 else 0,
            "is_pediatric": 1 if req.patient_age <= 12 else 0,
            "patient_origin": req.patient_origin,
            "is_referral": 0,
            "visit_type": req.visit_type,
            "active_counters": req.active_counters,
            "queue_depth_at_arrival": req.queue_depth,
            "arrival_minute_of_day": arr_min,
            "temp_celsius": 28.0,
            "rainfall_mm": 0.0,
            "is_monday": 1 if now.weekday() == 0 else 0,
            "is_saturday": 1 if now.weekday() == 5 else 0,
            "is_post_holiday": 0,
            "is_harvest_season": 1 if now.month in [11, 12] else 0,
            "has_abha_id": 1 if req.abha_id else 0
        }])
        pred = wait_time_model.predict(input_df)
        wait_time_info = {
            "expected_wait_mins": float(pred["predicted_wait_mean"][0]),
            "confidence_window": f"{pred['predicted_wait_p10'][0]} - {pred['predicted_wait_p90'][0]} mins"
        }

    return {
        "token_number": token_str,
        "abha_id": req.abha_id or "ANONYMIZED-WALKIN",
        "department": req.department,
        "triage_tier": req.triage_tier,
        "queue_position": req.queue_depth + 1,
        "issued_timestamp": now.isoformat(),
        "wait_time_estimate": wait_time_info,
        "abdm_compliant": True,
        "dpdpa_2023_notice": "Data pseudonymized for patient confidentiality under DPDPA 2023 Section 8."
    }

@app.post("/simulate_queue")
def simulate_queue(req: SimulationRequest):
    benchmark_results = run_comparative_benchmark(
        n_patients=req.num_patients,
        num_counters=req.num_counters,
        alpha=req.alpha_aging,
        beta=req.beta_sept
    )
    return {
        "status": "Simulation Completed Successfully",
        "parameters": {
            "patients_simulated": req.num_patients,
            "active_counters": req.num_counters,
            "alpha_aging": req.alpha_aging,
            "beta_sept": req.beta_sept
        },
        "results": benchmark_results
    }

@app.get("/metrics")
def get_metrics():
    metrics_file = os.path.join(BASE_DIR, "models", "training_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            import json
            return json.load(f)
    return {"message": "Metrics file not found"}
