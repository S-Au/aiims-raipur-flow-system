"""
AIIMS Raipur OPD Wait-Time Prediction Engine Training Pipeline
"""

import os
import sys
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.predictor_classes import OPDWaitTimePredictor

def train_and_export():
    data_path = os.path.join(BASE_DIR, "data", "aiims_raipur_opd_52k.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records.")

    features = [
        "department", "triage_tier", "patient_age", "patient_gender",
        "is_senior", "is_pediatric", "patient_origin", "is_referral",
        "visit_type", "active_counters", "queue_depth_at_arrival",
        "arrival_minute_of_day", "temp_celsius", "rainfall_mm",
        "is_monday", "is_saturday", "is_post_holiday", "is_harvest_season",
        "has_abha_id"
    ]
    target = "actual_wait_time_mins"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictor = OPDWaitTimePredictor()
    predictor.fit(X_train, y_train)

    metrics = predictor.evaluate(X_test, y_test)
    print("\n--- Model Evaluation Results ---")
    print(json.dumps(metrics, indent=2))

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "wait_time_model.joblib")
    metrics_path = os.path.join(models_dir, "wait_time_metrics.json")

    joblib.dump(predictor, model_path)
    print(f"Saved model to {model_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    train_and_export()
