"""
Integration tests for FastAPI REST API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "institution" in data
    assert "AIIMS" in data["institution"]

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Healthy"
    assert data["wait_time_model_active"] is True
    assert data["footfall_model_active"] is True

def test_predict_wait_time_endpoint():
    payload = {
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
    }
    response = client.post("/predict_wait_time", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "expected_wait_minutes" in data["predictions"]
    assert "optimistic_p10_minutes" in data["predictions"]
    assert "conservative_p90_minutes" in data["predictions"]
    assert data["predictions"]["optimistic_p10_minutes"] <= data["predictions"]["conservative_p90_minutes"]

def test_forecast_footfall_endpoint():
    payload = {
        "department": "Orthopedics",
        "day_of_week": "Monday",
        "temp_celsius": 29.0,
        "rainfall_mm": 0.0,
        "is_monday": 1,
        "is_saturday": 0,
        "is_post_holiday": 1,
        "is_harvest_season": 1
    }
    response = client.post("/forecast_footfall", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "forecasted_daily_footfall" in data
    assert data["forecasted_daily_footfall"] > 0
    assert "recommended_active_counters" in data

def test_issue_token_endpoint():
    payload = {
        "abha_id": "91-8888-7777-6666",
        "department": "Pediatrics",
        "patient_age": 7,
        "patient_gender": "M",
        "patient_origin": "Durg_Bhilai",
        "visit_type": "New_Consultation",
        "triage_tier": 2,
        "active_counters": 3,
        "queue_depth": 5
    }
    response = client.post("/issue_token", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "token_number" in data
    assert data["token_number"].startswith("PED-")
    assert data["abdm_compliant"] is True
    assert "wait_time_estimate" in data

def test_simulate_queue_endpoint():
    payload = {
        "num_patients": 50,
        "num_counters": 2,
        "alpha_aging": 2.5,
        "beta_sept": 12.0
    }
    response = client.post("/simulate_queue", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Simulation Completed Successfully"
    assert "results" in data
    assert "fifo" in data["results"]
    assert "sept_ap" in data["results"]

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "wait_time_model" in data
    assert "footfall_forecasting_model" in data

def test_predict_tier1_emergency_endpoint():
    payload = {
        "department": "Cardiology",
        "triage_tier": 1,
        "patient_age": 70,
        "patient_gender": "M",
        "patient_origin": "Raipur_Urban",
        "active_counters": 1,
        "queue_depth_at_arrival": 25,
        "arrival_minute_of_day": 540
    }
    response = client.post("/predict_wait_time", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["triage_tier"] == 1
    # Under protocol, emergency wait should be capped at 8 mins
    assert data["predictions"]["expected_wait_minutes"] <= 8.0
    assert data["predictions"]["conservative_p90_minutes"] <= 8.0

def test_simulate_queue_custom_parameters():
    payload = {
        "num_patients": 40,
        "num_counters": 2,
        "alpha_aging": 4.0,
        "beta_sept": 20.0
    }
    response = client.post("/simulate_queue", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["parameters"]["alpha_aging"] == 4.0
    assert data["parameters"]["beta_sept"] == 20.0
    assert data["results"]["sept_ap"]["total_patients_served"] == 40
