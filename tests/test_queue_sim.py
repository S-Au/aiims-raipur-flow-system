"""
Unit and benchmark tests for Queue Simulation & SEPT-AP Dispatcher
"""

import pytest
import numpy as np
from simulation.queue_sim import (
    AIIMSQueueSimulation,
    QueuedPatient,
    DynamicSEPTScheduler,
    run_comparative_benchmark
)

def test_sept_priority_and_aging():
    scheduler = DynamicSEPTScheduler(alpha=2.0, beta=10.0)

    # Patient A: Tier 4 (Routine), arrived at t = 0
    patient_a = QueuedPatient(
        sort_key=0.0,
        token_id="TKN-001",
        triage_tier=4,
        arrival_time=0.0,
        expected_duration=10.0,
        actual_duration=10.0,
        patient_age=30,
        is_senior=False,
        visit_type="Routine"
    )

    # Patient B: Tier 3 (Semi-Urgent), arrived at t = 100
    patient_b = QueuedPatient(
        sort_key=0.0,
        token_id="TKN-002",
        triage_tier=3,
        arrival_time=100.0,
        expected_duration=10.0,
        actual_duration=10.0,
        patient_age=30,
        is_senior=False,
        visit_type="Routine"
    )

    # At t = 100: Patient A has waited 100 mins.
    # Score A = 20 (base) + 2.0 * 100 (aging) + 10 = 230
    # Score B = 150 (base) + 2.0 * 0 (aging) + 10 = 160
    # Patient A should now have higher score than fresh Patient B due to aging!
    score_a = scheduler.compute_score(patient_a, current_time=100.0)
    score_b = scheduler.compute_score(patient_b, current_time=100.0)

    assert score_a > score_b, f"Starvation aging failed: Score A ({score_a}) <= Score B ({score_b})"

def test_tier1_fast_track_priority():
    sim = AIIMSQueueSimulation(num_counters=1, policy="SEPT_AP")
    
    # Add multiple routine patients
    for i in range(5):
        sim.add_patient(QueuedPatient(
            sort_key=0.0,
            token_id=f"TKN-ROUTINE-{i}",
            triage_tier=4,
            arrival_time=0.0,
            expected_duration=15.0,
            actual_duration=15.0,
            patient_age=35,
            is_senior=False,
            visit_type="New_Consultation"
        ))
    
    # Add Tier 1 emergency arriving at minute 5
    sim.add_patient(QueuedPatient(
        sort_key=0.0,
        token_id="TKN-EMERGENCY-1",
        triage_tier=1,
        arrival_time=5.0,
        expected_duration=12.0,
        actual_duration=12.0,
        patient_age=60,
        is_senior=True,
        visit_type="Emergency"
    ))

    results = sim.run()
    assert results["total_patients_served"] == 6
    
    # Emergency patient must be served right after the first ongoing patient finishes
    completed = sim.completed_patients
    # Patient 0 started at 0.0, ends at 15.0
    # Emergency arrived at 5.0, so at 15.0 counter frees up and emergency should be served at 15.0
    em_patient = next(p for p in completed if p.token_id == "TKN-EMERGENCY-1")
    assert em_patient.start_service_time == 15.0, f"Emergency should start at 15.0, got {em_patient.start_service_time}"

def test_comparative_benchmark_reduction():
    res = run_comparative_benchmark(n_patients=100, num_counters=3)
    assert "fifo" in res
    assert "sept_ap" in res
    assert res["tier2_urgent_reduction_pct"] > 30.0, f"Expected > 30% reduction in urgent wait, got {res['tier2_urgent_reduction_pct']}%"

def test_fifo_tier1_earliest_arrival_ordering():
    """Verify that among multiple Tier 1 arrivals, FIFO always serves the earlier arrival first."""
    sim = AIIMSQueueSimulation(num_counters=1, policy="FIFO")
    p_later = QueuedPatient(
        sort_key=0.0, token_id="TKN-T1-LATER", triage_tier=1, arrival_time=15.0,
        expected_duration=10.0, actual_duration=10.0, patient_age=45, is_senior=False, visit_type="New"
    )
    p_earlier = QueuedPatient(
        sort_key=0.0, token_id="TKN-T1-EARLIER", triage_tier=1, arrival_time=5.0,
        expected_duration=10.0, actual_duration=10.0, patient_age=50, is_senior=False, visit_type="New"
    )
    # Add in reverse arrival order to verify algorithm doesn't just pick index 0
    sim.waiting_pool = [p_later, p_earlier]
    selected = sim._select_next_patient(current_time=20.0)
    assert selected.token_id == "TKN-T1-EARLIER", f"Expected earlier arrival TKN-T1-EARLIER, got {selected.token_id}"

def test_simulation_max_duration_cutoff():
    """Verify that max_duration_mins halts late arrivals and counts unserved queue properly."""
    sim = AIIMSQueueSimulation(num_counters=1, policy="FIFO")
    # Patient 1 arrives at 0, takes 60 mins
    sim.add_patient(QueuedPatient(0.0, "TKN-1", 3, 0.0, 60.0, 60.0, 30, False, "New"))
    # Patient 2 arrives at 30, takes 60 mins
    sim.add_patient(QueuedPatient(0.0, "TKN-2", 3, 30.0, 60.0, 60.0, 30, False, "New"))
    # Patient 3 arrives at 150 (after cutoff of 100)
    sim.add_patient(QueuedPatient(0.0, "TKN-3", 3, 150.0, 60.0, 60.0, 30, False, "New"))

    results = sim.run(max_duration_mins=100.0)
    # TKN-3 arriving at 150 should be skipped due to cutoff
    assert results["total_patients_served"] == 2
    assert "unserved_patients" in results

def test_custom_alpha_beta_benchmark():
    """Verify that custom alpha and beta parameters are passed cleanly to benchmark."""
    res = run_comparative_benchmark(n_patients=50, num_counters=2, alpha=5.0, beta=25.0)
    assert "fifo" in res
    assert "sept_ap" in res
    assert res["sept_ap"]["total_patients_served"] == 50
