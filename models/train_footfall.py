"""
AIIMS Raipur Departmental Daily Footfall Forecaster Training Pipeline
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

from models.predictor_classes import DepartmentFootfallForecaster

def train_and_export():
    data_path = os.path.join(BASE_DIR, "data", "aiims_raipur_opd_52k.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Loading data from {data_path} to build daily departmental aggregates...")
    df = pd.read_csv(data_path)

    # Group by visit_date and department to get daily footfall
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

    print(f"Aggregated daily departmental records: {len(agg_df)} panel rows.")

    features = [
        "department", "day_of_week", "temp_celsius", "rainfall_mm",
        "is_monday", "is_saturday", "is_post_holiday", "is_harvest_season"
    ]
    target = "daily_footfall"

    X = agg_df[features]
    y = agg_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    forecaster = DepartmentFootfallForecaster()
    forecaster.fit(X_train, y_train)

    metrics = forecaster.evaluate(X_test, y_test)
    print("\n--- Footfall Forecaster Evaluation Results ---")
    print(json.dumps(metrics, indent=2))

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "footfall_model.joblib")
    metrics_path = os.path.join(models_dir, "footfall_metrics.json")

    joblib.dump(forecaster, model_path)
    print(f"Saved footfall model to {model_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved footfall metrics to {metrics_path}")

if __name__ == "__main__":
    train_and_export()
