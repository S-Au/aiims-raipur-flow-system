"""
AIIMS Raipur OPD Flow Optimization - Machine Learning Model Definitions
Canonical classes for OPD Wait-Time Prediction and Departmental Footfall Forecasting.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class OPDWaitTimePredictor:
    def __init__(self):
        self.preprocessor = None
        self.model_mean = None
        self.model_p10 = None
        self.model_p50 = None
        self.model_p90 = None
        self.feature_names = None
        self.metrics = {}

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        q_depth = df["queue_depth_at_arrival"].values
        counters = np.maximum(1, df["active_counters"].values)
        df_out["queue_per_counter"] = q_depth / counters
        df_out["urgency_weight"] = 5 - df["triage_tier"].values
        arr = df["arrival_minute_of_day"].values
        df_out["is_rush_hour"] = ((arr >= 480) & (arr <= 615)).astype(int)
        df_out["patient_vulnerability"] = (df["is_senior"].values | df["is_pediatric"].values).astype(int)
        return df_out

    def _init_fast_path(self):
        if self.preprocessor is None:
            return
        cat_trans = self.preprocessor.named_transformers_.get("cat")
        num_trans = self.preprocessor.named_transformers_.get("num")
        if cat_trans is None or num_trans is None:
            return
        self.cat_cols = ["department", "patient_gender", "patient_origin", "visit_type"]
        self.cat_maps = []
        offset = 0
        for cats in cat_trans.categories_:
            self.cat_maps.append({val: offset + idx for idx, val in enumerate(cats)})
            offset += len(cats)
        self.num_cols = [
            "arrival_minute_of_day", "triage_tier", "patient_age", "active_counters",
            "queue_depth_at_arrival", "temp_celsius", "rainfall_mm",
            "is_senior", "is_pediatric", "is_referral", "is_monday", "is_saturday",
            "is_post_holiday", "is_harvest_season", "has_abha_id",
            "queue_per_counter", "urgency_weight", "is_rush_hour", "patient_vulnerability"
        ]
        self.num_mean = num_trans.mean_
        self.num_scale = num_trans.scale_
        self.n_total_features = offset + len(self.num_cols)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        categorical_cols = [
            "department", "patient_gender", "patient_origin", "visit_type"
        ]
        numeric_cols = [
            "arrival_minute_of_day", "triage_tier", "patient_age", "active_counters",
            "queue_depth_at_arrival", "temp_celsius", "rainfall_mm",
            "is_senior", "is_pediatric", "is_referral", "is_monday", "is_saturday",
            "is_post_holiday", "is_harvest_season", "has_abha_id",
            "queue_per_counter", "urgency_weight", "is_rush_hour", "patient_vulnerability"
        ]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
                ("num", StandardScaler(), numeric_cols)
            ]
        )

        X_train_eng = self.engineer_features(X_train)
        X_trans = self.preprocessor.fit_transform(X_train_eng)

        # 1. Mean model (Tuned for ultra-fast < 10ms latency & R^2 > 0.95)
        self.model_mean = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.10,
            max_iter=55,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=42
        )
        self.model_mean.fit(X_trans, y_train)

        # 2. Quantile P10 (Optimistic lower bound)
        self.model_p10 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.10,
            learning_rate=0.10,
            max_iter=45,
            max_leaf_nodes=25,
            min_samples_leaf=20,
            random_state=42
        )
        self.model_p10.fit(X_trans, y_train)

        # 3. Quantile P50 (Median)
        self.model_p50 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.50,
            learning_rate=0.10,
            max_iter=45,
            max_leaf_nodes=25,
            min_samples_leaf=20,
            random_state=42
        )
        self.model_p50.fit(X_trans, y_train)

        # 4. Quantile P90 (Conservative upper bound)
        self.model_p90 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.90,
            learning_rate=0.10,
            max_iter=45,
            max_leaf_nodes=25,
            min_samples_leaf=20,
            random_state=42
        )
        self.model_p90.fit(X_trans, y_train)
        self._init_fast_path()

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        if getattr(self, "cat_maps", None) is None:
            self._init_fast_path()

        if len(X) == 1 and getattr(self, "cat_maps", None) is not None:
            # Ultra-fast single-record transform (sub-1ms execution bypasses DataFrame reflection)
            row_dict = X.iloc[0].to_dict()
            q_depth = float(row_dict.get("queue_depth_at_arrival", 0))
            counters = max(1, float(row_dict.get("active_counters", 1)))
            row_dict["queue_per_counter"] = q_depth / counters
            row_dict["urgency_weight"] = 5 - float(row_dict.get("triage_tier", 3))
            arr = float(row_dict.get("arrival_minute_of_day", 540))
            row_dict["is_rush_hour"] = 1 if (480 <= arr <= 615) else 0
            is_sen = int(row_dict.get("is_senior", 0))
            is_ped = int(row_dict.get("is_pediatric", 0))
            row_dict["patient_vulnerability"] = 1 if (is_sen or is_ped) else 0

            arr_np = np.zeros((1, self.n_total_features), dtype=np.float64)
            for c_idx, col in enumerate(self.cat_cols):
                val = row_dict.get(col, "")
                pos = self.cat_maps[c_idx].get(val, -1)
                if pos >= 0:
                    arr_np[0, pos] = 1.0

            num_vals = np.array([float(row_dict.get(c, 0.0)) for c in self.num_cols], dtype=np.float64)
            arr_np[0, len(arr_np[0]) - len(self.num_cols):] = (num_vals - self.num_mean) / self.num_scale

            pred_mean = np.maximum(1.0, self.model_mean.predict(arr_np))
            pred_p10 = np.maximum(1.0, self.model_p10.predict(arr_np))
            pred_p50 = np.maximum(pred_p10, self.model_p50.predict(arr_np))
            pred_p90 = np.maximum(pred_p50 + 0.5, self.model_p90.predict(arr_np))

            # Enforce clinical protocol safety cap for Tier 1 Emergency Resuscitation (protocol max 8 mins)
            triage_val = int(row_dict.get("triage_tier", 3))
            if triage_val == 1:
                pred_mean = np.clip(pred_mean, 1.0, 8.0)
                pred_p10 = np.clip(pred_p10, 1.0, 4.0)
                pred_p50 = np.clip(pred_p50, 1.0, 6.0)
                pred_p90 = np.clip(pred_p90, 1.5, 8.0)

            return {
                "predicted_wait_mean": np.round(pred_mean, 1),
                "predicted_wait_p10": np.round(pred_p10, 1),
                "predicted_wait_p50": np.round(pred_p50, 1),
                "predicted_wait_p90": np.round(pred_p90, 1)
            }

        # Batch transform
        X_eng = self.engineer_features(X)
        X_trans = self.preprocessor.transform(X_eng)
        
        pred_mean = np.maximum(1.0, self.model_mean.predict(X_trans))
        pred_p10 = np.maximum(1.0, self.model_p10.predict(X_trans))
        pred_p50 = np.maximum(pred_p10, self.model_p50.predict(X_trans))
        pred_p90 = np.maximum(pred_p50 + 0.5, self.model_p90.predict(X_trans))

        # Enforce clinical protocol safety cap for Tier 1 Emergency Resuscitation
        if "triage_tier" in X.columns:
            is_tier1 = (X["triage_tier"].values == 1)
            if np.any(is_tier1):
                pred_mean[is_tier1] = np.clip(pred_mean[is_tier1], 1.0, 8.0)
                pred_p10[is_tier1] = np.clip(pred_p10[is_tier1], 1.0, 4.0)
                pred_p50[is_tier1] = np.clip(pred_p50[is_tier1], 1.0, 6.0)
                pred_p90[is_tier1] = np.clip(pred_p90[is_tier1], 1.5, 8.0)

        return {
            "predicted_wait_mean": np.round(pred_mean, 1),
            "predicted_wait_p10": np.round(pred_p10, 1),
            "predicted_wait_p50": np.round(pred_p50, 1),
            "predicted_wait_p90": np.round(pred_p90, 1)
        }

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        t0 = time.perf_counter()
        preds = self.predict(X_test)
        t_elapsed = time.perf_counter() - t0
        latency_ms_per_record = (t_elapsed / len(X_test)) * 1000

        y_pred = preds["predicted_wait_mean"]
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        # Quantile coverage check
        p10 = preds["predicted_wait_p10"]
        p90 = preds["predicted_wait_p90"]
        p80_coverage = float(np.mean((y_test >= p10) & (y_test <= p90)))

        metrics = {
            "r2_score": round(r2, 4),
            "mae_minutes": round(mae, 2),
            "rmse_minutes": round(rmse, 2),
            "p80_interval_coverage": round(p80_coverage, 4),
            "inference_latency_ms": round(latency_ms_per_record, 3),
            "test_sample_size": len(y_test),
            "criteria_satisfied": {
                "r2_ge_0_85": bool(r2 >= 0.85),
                "mae_lt_10_mins": bool(mae < 10.0),
                "latency_lt_10ms": bool(latency_ms_per_record < 10.0)
            }
        }
        self.metrics = metrics
        return metrics

class DepartmentFootfallForecaster:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.metrics = {}

    def _init_fast_path(self):
        if self.preprocessor is None:
            return
        cat_trans = self.preprocessor.named_transformers_.get("cat")
        num_trans = self.preprocessor.named_transformers_.get("num")
        if cat_trans is None or num_trans is None:
            return
        self.cat_cols = ["department", "day_of_week"]
        self.cat_maps = []
        offset = 0
        for cats in cat_trans.categories_:
            self.cat_maps.append({val: offset + idx for idx, val in enumerate(cats)})
            offset += len(cats)
        self.num_cols = [
            "temp_celsius", "rainfall_mm", "is_monday", "is_saturday",
            "is_post_holiday", "is_harvest_season"
        ]
        self.num_mean = num_trans.mean_
        self.num_scale = num_trans.scale_
        self.n_total_features = offset + len(self.num_cols)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        cat_cols = ["department", "day_of_week"]
        num_cols = [
            "temp_celsius", "rainfall_mm", "is_monday", "is_saturday",
            "is_post_holiday", "is_harvest_season"
        ]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols)
            ]
        )

        X_trans = self.preprocessor.fit_transform(X_train)

        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.08,
            max_iter=60,
            max_leaf_nodes=31,
            min_samples_leaf=5,
            random_state=42
        )
        self.model.fit(X_trans, y_train)
        self._init_fast_path()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if getattr(self, "cat_maps", None) is None:
            self._init_fast_path()

        if len(X) == 1 and getattr(self, "cat_maps", None) is not None:
            row_dict = X.iloc[0].to_dict()
            arr_np = np.zeros((1, self.n_total_features), dtype=np.float64)
            for c_idx, col in enumerate(self.cat_cols):
                val = row_dict.get(col, "")
                pos = self.cat_maps[c_idx].get(val, -1)
                if pos >= 0:
                    arr_np[0, pos] = 1.0
            num_vals = np.array([float(row_dict.get(c, 0.0)) for c in self.num_cols], dtype=np.float64)
            arr_np[0, len(arr_np[0]) - len(self.num_cols):] = (num_vals - self.num_mean) / self.num_scale
            preds = self.model.predict(arr_np)
        else:
            X_trans = self.preprocessor.transform(X)
            preds = self.model.predict(X_trans)

        return np.maximum(5, np.round(preds)).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        t0 = time.perf_counter()
        y_pred = self.predict(X_test)
        t_elapsed = time.perf_counter() - t0
        latency_ms_per_record = (t_elapsed / len(X_test)) * 1000

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        metrics = {
            "r2_score": round(r2, 4),
            "mae_patients": round(mae, 2),
            "rmse_patients": round(rmse, 2),
            "inference_latency_ms": round(latency_ms_per_record, 3),
            "test_sample_size": len(y_test),
            "criteria_satisfied": {
                "r2_ge_0_80": bool(r2 >= 0.80),
                "mae_lt_15_patients": bool(mae < 15.0)
            }
        }
        self.metrics = metrics
        return metrics
