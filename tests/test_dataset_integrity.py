"""
Unit and statistical integrity tests for AIIMS Raipur OPD Dataset
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "aiims_raipur_opd_52k.csv")

@pytest.fixture(scope="module")
def opd_data():
    assert os.path.exists(DATA_PATH), f"Dataset missing at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    return df

def test_dataset_size_and_departments(opd_data):
    assert len(opd_data) >= 50000, f"Expected >= 50,000 records, got {len(opd_data)}"
    assert opd_data["department"].nunique() >= 30, f"Expected >= 30 departments, got {opd_data['department'].nunique()}"

def test_no_missing_critical_values(opd_data):
    critical_cols = [
        "token_id", "department", "triage_tier", "patient_age", "active_counters",
        "queue_depth_at_arrival", "actual_wait_time_mins", "arrival_minute_of_day"
    ]
    for col in critical_cols:
        assert col in opd_data.columns, f"Missing column {col}"
        assert opd_data[col].isnull().sum() == 0, f"Column {col} has null values"

def test_triage_distribution_and_values(opd_data):
    tiers = set(opd_data["triage_tier"].unique())
    assert tiers.issubset({1, 2, 3, 4}), f"Unexpected triage tiers: {tiers}"
    
    # Tier 1 should be low frequency (critical/emergency, ~1-4%)
    tier1_pct = (opd_data["triage_tier"] == 1).mean() * 100
    assert 0.5 <= tier1_pct <= 6.0, f"Tier 1 frequency abnormal: {tier1_pct}%"

def test_physical_wait_time_bounds(opd_data):
    # Wait times must be strictly positive
    assert (opd_data["actual_wait_time_mins"] > 0).all(), "Wait times must be > 0"
    
    # Tier 1 wait times must be very short on average (< 10 mins)
    tier1_waits = opd_data[opd_data["triage_tier"] == 1]["actual_wait_time_mins"]
    assert tier1_waits.mean() < 8.0, f"Tier 1 avg wait should be < 8 min, got {tier1_waits.mean()}"

def test_registration_hours_coverage(opd_data):
    # Registration is between 07:15 (435 min) and 11:45 (705 min)
    assert opd_data["arrival_minute_of_day"].min() >= 430
    assert opd_data["arrival_minute_of_day"].max() <= 710
