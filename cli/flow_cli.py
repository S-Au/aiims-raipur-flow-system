"""
AIIMS Raipur Outpatient Flow Optimization & Queue Management CLI
Interactive command-line tool for clinical administrators, OPD superintendents,
and triage nursing officers.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Ensure parent path is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.predictor_classes import OPDWaitTimePredictor, DepartmentFootfallForecaster
from simulation.queue_sim import run_comparative_benchmark

def get_wait_model():
    path = os.path.join(BASE_DIR, "models", "wait_time_model.joblib")
    if not os.path.exists(path):
        print(f"Error: Model not found at {path}. Run models/train_wait_time.py first.")
        sys.exit(1)
    return joblib.load(path)

def get_footfall_model():
    path = os.path.join(BASE_DIR, "models", "footfall_model.joblib")
    if not os.path.exists(path):
        print(f"Error: Model not found at {path}. Run models/train_footfall.py first.")
        sys.exit(1)
    return joblib.load(path)

def cmd_predict_wait(args):
    model = get_wait_model()
    is_senior = 1 if args.age >= 60 else 0
    is_pediatric = 1 if args.age <= 12 else 0

    input_data = pd.DataFrame([{
        "department": args.dept,
        "triage_tier": args.triage,
        "patient_age": args.age,
        "patient_gender": args.gender,
        "is_senior": is_senior,
        "is_pediatric": is_pediatric,
        "patient_origin": args.origin,
        "is_referral": args.referral,
        "visit_type": args.visit_type,
        "active_counters": args.counters,
        "queue_depth_at_arrival": args.queue,
        "arrival_minute_of_day": args.arrival_minute,
        "temp_celsius": args.temp,
        "rainfall_mm": args.rain,
        "is_monday": 1 if args.monday else 0,
        "is_saturday": 1 if args.saturday else 0,
        "is_post_holiday": 1 if args.post_holiday else 0,
        "is_harvest_season": 1 if args.harvest else 0,
        "has_abha_id": 1 if args.abha else 0
    }])

    preds = model.predict(input_data)
    triage_labels = {
        1: "Tier 1 (Red: Critical / Immediate Resuscitation)",
        2: "Tier 2 (Yellow: Urgent / Severe Clinical Distress)",
        3: "Tier 3 (Green: Semi-Urgent / Standard Outpatient)",
        4: "Tier 4 (White: Routine / Follow-up / Report Review)"
    }

    print("\n" + "="*65)
    print(" AIIMS RAIPUR OPD - REAL-TIME WAIT-TIME PREDICTION")
    print("="*65)
    print(f" Department          : {args.dept}")
    print(f" Clinical Urgency    : {triage_labels.get(args.triage, 'Unknown')}")
    print(f" Queue Ahead         : {args.queue} patients")
    print(f" Active Counters     : {args.counters} chambers")
    print(f" Arrival Minute      : {args.arrival_minute} ({args.arrival_minute//60:02d}:{args.arrival_minute%60:02d} hrs)")
    print("-" * 65)
    print(f" Expected Wait (Mean): {preds['predicted_wait_mean'][0]} minutes")
    print(f" Median Wait (P50)   : {preds['predicted_wait_p50'][0]} minutes")
    print(f" 80% Confidence Band : [{preds['predicted_wait_p10'][0]} mins  to  {preds['predicted_wait_p90'][0]} mins]")
    print("="*65 + "\n")

def cmd_forecast_footfall(args):
    model = get_footfall_model()
    input_data = pd.DataFrame([{
        "department": args.dept,
        "day_of_week": args.day,
        "temp_celsius": args.temp,
        "rainfall_mm": args.rain,
        "is_monday": 1 if args.day.lower() == "monday" else 0,
        "is_saturday": 1 if args.day.lower() == "saturday" else 0,
        "is_post_holiday": 1 if args.post_holiday else 0,
        "is_harvest_season": 1 if args.harvest else 0
    }])

    count = int(model.predict(input_data)[0])
    recommended_counters = max(1, min(8, int(np.ceil(count / 22))))

    print("\n" + "="*65)
    print(" AIIMS RAIPUR OPD - DEPARTMENTAL FOOTFALL FORECAST")
    print("="*65)
    print(f" Department          : {args.dept}")
    print(f" Forecast Day        : {args.day}")
    print(f" Harvest Season Flag : {'YES' if args.harvest else 'NO'}")
    print(f" Post-Holiday Surge  : {'YES' if args.post_holiday else 'NO'}")
    print("-" * 65)
    print(f" Predicted Footfall  : {count} patients")
    print(f" Recommended Doctors : {recommended_counters} active consulting counters")
    print(f" Recommended Staffing: {recommended_counters * 2} nursing & support officers")
    print("="*65 + "\n")

def cmd_simulate(args):
    print("\n" + "="*70)
    print(f" RUNNING DISCRETE-EVENT QUEUE SIMULATION ({args.patients} patients, {args.counters} counters, alpha={args.alpha}, beta={args.beta})")
    print("="*70)
    res = run_comparative_benchmark(n_patients=args.patients, num_counters=args.counters, alpha=args.alpha, beta=args.beta)
    fifo = res["fifo"]
    sept = res["sept_ap"]

    print(f"{'Metric':<32} | {'Traditional FIFO':<16} | {'AIIMS SEPT-AP':<16}")
    print("-" * 70)
    print(f"{'Overall Avg Wait (mins)':<32} | {fifo['avg_wait_mins']:<16.2f} | {sept['avg_wait_mins']:<16.2f}")
    print(f"{'P50 Median Wait (mins)':<32} | {fifo['p50_wait_mins']:<16.2f} | {sept['p50_wait_mins']:<16.2f}")
    print(f"{'P90 Conservative Wait (mins)':<32} | {fifo['p90_wait_mins']:<16.2f} | {sept['p90_wait_mins']:<16.2f}")
    print(f"{'Max Patient Wait (mins)':<32} | {fifo['max_wait_mins']:<16.2f} | {sept['max_wait_mins']:<16.2f}")
    print(f"{'Tier 1 (Critical) Avg Wait':<32} | {fifo['tier_avg_wait_mins']['Tier_1']:<16.2f} | {sept['tier_avg_wait_mins']['Tier_1']:<16.2f}")
    print(f"{'Tier 2 (Urgent) Avg Wait':<32} | {fifo['tier_avg_wait_mins']['Tier_2']:<16.2f} | {sept['tier_avg_wait_mins']['Tier_2']:<16.2f}")
    print(f"{'Tier 3 (Semi-Urgent) Avg Wait':<32} | {fifo['tier_avg_wait_mins']['Tier_3']:<16.2f} | {sept['tier_avg_wait_mins']['Tier_3']:<16.2f}")
    print(f"{'Tier 4 (Routine) Avg Wait':<32} | {fifo['tier_avg_wait_mins']['Tier_4']:<16.2f} | {sept['tier_avg_wait_mins']['Tier_4']:<16.2f}")
    print(f"{'Counter Utilization Rate':<32} | {fifo['counter_utilization']*100:<15.1f}% | {sept['counter_utilization']*100:<15.1f}%")
    print("-" * 70)
    print(f"[*] Tier 1 Critical Fast-Track Improvement : {res['tier1_fast_track_reduction_pct']}% reduction")
    print(f"[*] Tier 2 Urgent Patient Wait Reduction   : {res['tier2_urgent_reduction_pct']}% reduction")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="AIIMS Raipur OPD Flow Optimization CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational commands")

    # Command: predict-wait
    p_wait = subparsers.add_parser("predict-wait", help="Predict wait time for a patient")
    p_wait.add_argument("--dept", type=str, default="General Medicine", help="OPD Department")
    p_wait.add_argument("--triage", type=int, default=3, choices=[1, 2, 3, 4], help="Clinical Triage Tier")
    p_wait.add_argument("--queue", type=int, default=15, help="Current queue depth ahead")
    p_wait.add_argument("--counters", type=int, default=4, help="Active consultation chambers")
    p_wait.add_argument("--age", type=int, default=45, help="Patient age")
    p_wait.add_argument("--gender", type=str, default="M", choices=["M", "F", "Other"])
    p_wait.add_argument("--origin", type=str, default="Raipur_Urban")
    p_wait.add_argument("--referral", type=int, default=0, choices=[0, 1])
    p_wait.add_argument("--visit-type", type=str, default="New_Consultation", dest="visit_type")
    p_wait.add_argument("--arrival-minute", type=int, default=540, dest="arrival_minute", help="Minutes from midnight")
    p_wait.add_argument("--temp", type=float, default=28.0)
    p_wait.add_argument("--rain", type=float, default=0.0)
    p_wait.add_argument("--monday", action="store_true")
    p_wait.add_argument("--saturday", action="store_true")
    p_wait.add_argument("--post-holiday", action="store_true", dest="post_holiday")
    p_wait.add_argument("--harvest", action="store_true")
    p_wait.add_argument("--abha", action="store_true")
    p_wait.set_defaults(func=cmd_predict_wait)

    # Command: forecast-footfall
    p_foot = subparsers.add_parser("forecast-footfall", help="Forecast daily departmental footfall")
    p_foot.add_argument("--dept", type=str, default="General Medicine", help="OPD Department")
    p_foot.add_argument("--day", type=str, default="Monday", help="Day of week")
    p_foot.add_argument("--temp", type=float, default=28.0)
    p_foot.add_argument("--rain", type=float, default=0.0)
    p_foot.add_argument("--post-holiday", action="store_true", dest="post_holiday")
    p_foot.add_argument("--harvest", action="store_true")
    p_foot.set_defaults(func=cmd_forecast_footfall)

    # Command: simulate
    p_sim = subparsers.add_parser("simulate", help="Run M/M/s simulation comparing FIFO vs SEPT-AP")
    p_sim.add_argument("--patients", type=int, default=300, help="Number of arriving patients")
    p_sim.add_argument("--counters", type=int, default=4, help="Active counters")
    p_sim.add_argument("--alpha", type=float, default=2.5, help="Starvation aging parameter (alpha)")
    p_sim.add_argument("--beta", type=float, default=12.0, help="Shortest expected processing time weight (beta)")
    p_sim.set_defaults(func=cmd_simulate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
    else:
        args.func(args)

if __name__ == "__main__":
    main()
