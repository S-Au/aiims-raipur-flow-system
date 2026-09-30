"""
Unit and benchmark tests for AIIMS Raipur ML Models
"""

import os
import sys
import time
import joblib
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.predictor_classes import OPDWaitTimePredictor, DepartmentFootfallForecaster

WAIT_MODEL_PATH = os.path.join(BASE_DIR, "models", "wait_time_model.joblib")
FOOTFALL_MODEL_PATH = os.path.join(BASE_DIR, "models", "footfall_model.joblib")

@pytest.fixture(scope="module")
def wait_model():
    assert os.path.exists(WAIT_MODEL_PATH), f"Wait model missing at {WAIT_MODEL_PATH}"
    return joblib.load(WAIT_MODEL_PATH)

@pytest.fixture(scope="module")
def footfall_model():
    assert os.path.exists(FOOTFALL_MODEL_PATH), f"Footfall model missing at {FOOTFALL_MODEL_PATH}"
    return joblib.load(FOOTFALL_MODEL_PATH)

from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = os.path.join(BASE_DIR, "data", "aiims_raipur_opd_52k.csv")

def test_wait_model_criteria_and_metrics(wait_model):
    # Deep verification: actually evaluate the model on real dataset rows
    assert os.path.exists(DATA_PATH), f"Dataset missing at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    
    features = [
        "department", "triage_tier", "patient_age", "patient_gender",
        "is_senior", "is_pediatric", "patient_origin", "is_referral",
        "visit_type", "active_counters", "queue_depth_at_arrival",
        "arrival_minute_of_day", "temp_celsius", "rainfall_mm",
        "is_monday", "is_saturday", "is_post_holiday", "is_harvest_season",
        "has_abha_id"
    ]
    # Sample 2,000 validation records from tail of dataset
    eval_slice = df.tail(2000)
    X_eval = eval_slice[features]
    y_eval = eval_slice["actual_wait_time_mins"]

    preds = wait_model.predict(X_eval)
    y_pred = preds["predicted_wait_mean"]

    real_r2 = float(r2_score(y_eval, y_pred))
    real_mae = float(mean_absolute_error(y_eval, y_pred))

    assert real_r2 >= 0.85, f"Deep evaluation R2 score {real_r2:.4f} < 0.85"
    assert real_mae < 10.0, f"Deep evaluation MAE {real_mae:.2f} mins >= 10.0 mins"

def test_tier1_emergency_safety_cap(wait_model):
    """Verify that Tier 1 emergency predictions strictly adhere to clinical protocol cap (<= 8.0 mins)."""
    stress_cases = [
        {"department": "Cardiology", "triage_tier": 1, "patient_age": 65, "active_counters": 1, "queue_depth_at_arrival": 50},
        {"department": "Pediatrics", "triage_tier": 1, "patient_age": 4, "active_counters": 1, "queue_depth_at_arrival": 30},
        {"department": "General Medicine", "triage_tier": 1, "patient_age": 45, "active_counters": 2, "queue_depth_at_arrival": 60}
    ]
    df = pd.DataFrame(stress_cases)
    df["patient_gender"] = "M"
    df["is_senior"] = (df["patient_age"] >= 60).astype(int)
    df["is_pediatric"] = (df["patient_age"] <= 12).astype(int)
    df["patient_origin"] = "Raipur_Urban"
    df["is_referral"] = 0
    df["visit_type"] = "New_Consultation"
    df["arrival_minute_of_day"] = 540
    df["temp_celsius"] = 28.0
    df["rainfall_mm"] = 0.0
    df["is_monday"] = 1
    df["is_saturday"] = 0
    df["is_post_holiday"] = 1
    df["is_harvest_season"] = 1
    df["has_abha_id"] = 1

    preds = wait_model.predict(df)
    assert np.all(preds["predicted_wait_mean"] <= 8.0), f"Tier 1 mean exceeded 8 mins: {preds['predicted_wait_mean']}"
    assert np.all(preds["predicted_wait_p90"] <= 8.0), f"Tier 1 P90 exceeded 8 mins: {preds['predicted_wait_p90']}"
    assert np.all(preds["predicted_wait_p10"] >= 1.0), "Tier 1 P10 cannot be < 1.0 min"

def test_quantile_monotonicity(wait_model):
    # Test batch of diverse inputs to confirm P10 <= P50 <= P90 with non-degenerate positive spread
    test_cases = [
        {"department": "General Medicine", "triage_tier": 3, "patient_age": 40, "active_counters": 4, "queue_depth_at_arrival": 20},
        {"department": "Cardiology", "triage_tier": 1, "patient_age": 68, "active_counters": 3, "queue_depth_at_arrival": 5},
        {"department": "Pediatrics", "triage_tier": 2, "patient_age": 5, "active_counters": 2, "queue_depth_at_arrival": 15},
        {"department": "Orthopedics", "triage_tier": 4, "patient_age": 30, "active_counters": 5, "queue_depth_at_arrival": 35}
    ]
    df = pd.DataFrame(test_cases)
    df["patient_gender"] = "M"
    df["is_senior"] = (df["patient_age"] >= 60).astype(int)
    df["is_pediatric"] = (df["patient_age"] <= 12).astype(int)
    df["patient_origin"] = "Raipur_Urban"
    df["is_referral"] = 0
    df["visit_type"] = "New_Consultation"
    df["arrival_minute_of_day"] = 540
    df["temp_celsius"] = 28.0
    df["rainfall_mm"] = 0.0
    df["is_monday"] = 0
    df["is_saturday"] = 0
    df["is_post_holiday"] = 0
    df["is_harvest_season"] = 0
    df["has_abha_id"] = 1

    preds = wait_model.predict(df)
    p10 = preds["predicted_wait_p10"]
    p50 = preds["predicted_wait_p50"]
    p90 = preds["predicted_wait_p90"]

    assert len(p10) == len(df)
    assert np.all(p10 <= p50 + 1e-5), f"P10 ({p10}) exceeded P50 ({p50})"
    assert np.all(p50 <= p90 + 1e-5), f"P50 ({p50}) exceeded P90 ({p90})"
    # Confidence window should be non-degenerate
    assert np.all(p90 > p10), f"P90 must be strictly greater than P10"

def test_inference_latency_benchmark(wait_model):
    # Verify sub-10ms latency under single-record inference
    single_record = pd.DataFrame([{
        "department": "Neurology",
        "triage_tier": 2,
        "patient_age": 52,
        "patient_gender": "F",
        "is_senior": 0,
        "is_pediatric": 0,
        "patient_origin": "Bilaspur",
        "is_referral": 1,
        "visit_type": "New_Consultation",
        "active_counters": 3,
        "queue_depth_at_arrival": 14,
        "arrival_minute_of_day": 520,
        "temp_celsius": 26.5,
        "rainfall_mm": 0.0,
        "is_monday": 1,
        "is_saturday": 0,
        "is_post_holiday": 1,
        "is_harvest_season": 1,
        "has_abha_id": 1
    }])

    latencies = []
    # Warm-up
    _ = wait_model.predict(single_record)

    for _ in range(50):
        t0 = time.perf_counter()
        _ = wait_model.predict(single_record)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = np.mean(latencies)
    assert avg_latency < 10.0, f"Average single inference latency {avg_latency:.2f}ms exceeds 10ms target"

def test_footfall_forecaster_accuracy(footfall_model):
    # Deep verification: evaluate forecaster on panel aggregation
    assert os.path.exists(DATA_PATH), f"Dataset missing at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    agg_df = df.groupby(["visit_date", "department"]).agg(
        daily_footfall=("token_id", "count"),
        day_of_week=("day_of_week", "first"),
        temp_celsius=("temp_celsius", "mean"),
        rainfall_mm=("rainfall_mm", "mean"),
        is_monday=("is_monday", "first"),
        is_saturday=("is_saturday", "first"),
        is_post_holiday=("is_post_holiday", "first"),
        is_harvest_season=("is_harvest_season", "first")
    ).reset_index()

    features = [
        "department", "day_of_week", "temp_celsius", "rainfall_mm",
        "is_monday", "is_saturday", "is_post_holiday", "is_harvest_season"
    ]
    eval_panel = agg_df.tail(300)
    X_panel = eval_panel[features]
    y_panel = eval_panel["daily_footfall"]

    y_pred = footfall_model.predict(X_panel)
    real_r2 = float(r2_score(y_panel, y_pred))
    real_mae = float(mean_absolute_error(y_panel, y_pred))

    assert real_r2 >= 0.80, f"Footfall deep evaluation R2 {real_r2:.4f} < 0.80"
    assert real_mae < 15.0, f"Footfall deep evaluation MAE {real_mae:.2f} >= 15.0 patients"
