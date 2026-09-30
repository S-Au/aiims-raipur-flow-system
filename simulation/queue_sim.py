"""
AIIMS Raipur Multi-Server Queue Simulation & Dynamic Token Dispatcher
Implements M/M/s and M/G/s Queuing Models with:
- SEPT-AP (Shortest Expected Processing Time with Dynamic Aging and Clinical Priority)
- Starvation prevention via linear aging coefficients
- Dynamic counter load balancing
- Baseline First-Come First-Served (FIFO) vs SEPT-AP benchmark engine
"""

import heapq
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass(order=True)
class QueuedPatient:
    # Priority queue orders by lowest sort_key first, so we use negative effective priority
    sort_key: float
    token_id: str = field(compare=False)
    triage_tier: int = field(compare=False)
    arrival_time: float = field(compare=False)  # in minutes from start of sim
    expected_duration: float = field(compare=False)
    actual_duration: float = field(compare=False)
    patient_age: int = field(compare=False)
    is_senior: bool = field(compare=False)
    visit_type: str = field(compare=False)
    start_service_time: Optional[float] = field(default=None, compare=False)
    end_service_time: Optional[float] = field(default=None, compare=False)
    counter_id: Optional[int] = field(default=None, compare=False)

class DynamicSEPTScheduler:
    """
    SEPT-AP (Shortest Expected Processing Time with Aging and Clinical Priority)
    Calculates dynamic priority score:
      Score(p, t) = TriageWeight(p) + alpha * WaitTime(p, t) + beta * (1 / ExpectedDuration(p))
    """
    def __init__(self, alpha: float = 2.5, beta: float = 12.0):
        # alpha: aging rate (points per minute waited). Prevents starvation.
        # beta: shortest processing time bonus weight.
        self.alpha = alpha
        self.beta = beta
        self.triage_base = {
            1: 10000.0, # Immediate / Resuscitative (always next)
            2: 450.0,   # Urgent
            3: 150.0,   # Semi-urgent
            4: 20.0     # Routine
        }

    def compute_score(self, patient: QueuedPatient, current_time: float) -> float:
        wait_mins = max(0.0, current_time - patient.arrival_time)
        triage_pts = self.triage_base.get(patient.triage_tier, 20.0)
        
        # Senior citizen courtesy boost
        if patient.is_senior and patient.triage_tier > 1:
            triage_pts += 50.0

        aging_pts = self.alpha * wait_mins
        sept_pts = self.beta * (10.0 / max(1.0, patient.expected_duration))
        return triage_pts + aging_pts + sept_pts

class AIIMSQueueSimulation:
    def __init__(
        self,
        department: str = "General Medicine",
        num_counters: int = 4,
        policy: str = "SEPT_AP", # "SEPT_AP" or "FIFO"
        alpha: float = 2.5,
        beta: float = 12.0
    ):
        self.department = department
        self.num_counters = num_counters
        self.policy = policy
        self.scheduler = DynamicSEPTScheduler(alpha=alpha, beta=beta)
        self.counter_free_time = [0.0] * num_counters
        self.waiting_pool: List[QueuedPatient] = []
        self.completed_patients: List[QueuedPatient] = []
        self.current_time = 0.0

    def add_patient(self, patient: QueuedPatient):
        self.waiting_pool.append(patient)

    def _select_next_patient(self, current_time: float) -> Optional[QueuedPatient]:
        if not self.waiting_pool:
            return None

        if self.policy == "FIFO":
            # Tier 1 emergency always bypasses FIFO; among multiple Tier 1, earliest arrival wins
            tier1_indices = [idx for idx, p in enumerate(self.waiting_pool) if p.triage_tier == 1]
            if tier1_indices:
                best_idx = min(tier1_indices, key=lambda idx: self.waiting_pool[idx].arrival_time)
            else:
                best_idx = min(range(len(self.waiting_pool)), key=lambda idx: self.waiting_pool[idx].arrival_time)
            return self.waiting_pool.pop(best_idx)
        else: # SEPT_AP
            best_idx = 0
            best_score = -1e9
            best_arrival = 1e9
            for idx, p in enumerate(self.waiting_pool):
                score = self.scheduler.compute_score(p, current_time)
                # Maximize priority score; break ties with earliest arrival time
                if (score > best_score) or (abs(score - best_score) < 1e-6 and p.arrival_time < best_arrival):
                    best_score = score
                    best_arrival = p.arrival_time
                    best_idx = idx
            return self.waiting_pool.pop(best_idx)

    def run(self, max_duration_mins: Optional[float] = None) -> Dict[str, Any]:
        """Runs the discrete-event queue execution."""
        # Sort initial pool by arrival time
        self.waiting_pool.sort(key=lambda p: p.arrival_time)
        upcoming_arrivals = self.waiting_pool.copy()
        self.waiting_pool.clear()

        current_time = 0.0
        # Priority queue for events: (event_time, event_type, event_id, counter_id, patient)
        # event_types: 1 = counter_free, 2 = patient_arrival
        event_heap = []
        event_counter = 0

        # Push all arrivals into event queue
        for p in upcoming_arrivals:
            heapq.heappush(event_heap, (p.arrival_time, 2, event_counter, -1, p))
            event_counter += 1

        # Initially all counters are free at time 0.0
        idle_counters = list(range(self.num_counters))

        while event_heap:
            ev_time, ev_type, _, c_id, payload = heapq.heappop(event_heap)
            current_time = ev_time

            # If max_duration_mins is set and we exceeded it, ignore future arrivals
            if max_duration_mins is not None and current_time > max_duration_mins and ev_type == 2:
                continue

            if ev_type == 2:  # Patient Arrival
                patient = payload
                self.waiting_pool.append(patient)
                # Work-conserving dispatch: if any counter is idle, immediately dispatch best candidate
                if idle_counters:
                    counter = idle_counters.pop(0)
                    next_patient = self._select_next_patient(current_time)
                    if next_patient:
                        next_patient.start_service_time = current_time
                        next_patient.end_service_time = current_time + next_patient.actual_duration
                        next_patient.counter_id = counter
                        self.completed_patients.append(next_patient)
                        heapq.heappush(event_heap, (next_patient.end_service_time, 1, event_counter, counter, None))
                        event_counter += 1

            elif ev_type == 1:  # Counter Free
                counter = c_id
                # Select next patient from waiting pool based on policy
                next_patient = self._select_next_patient(current_time)
                if next_patient:
                    next_patient.start_service_time = current_time
                    next_patient.end_service_time = current_time + next_patient.actual_duration
                    next_patient.counter_id = counter
                    self.completed_patients.append(next_patient)
                    heapq.heappush(event_heap, (next_patient.end_service_time, 1, event_counter, counter, None))
                    event_counter += 1
                else:
                    idle_counters.append(counter)

        # Compute summary metrics
        wait_times = [p.start_service_time - p.arrival_time for p in self.completed_patients]
        tier_waits = {1: [], 2: [], 3: [], 4: []}
        for p in self.completed_patients:
            tier_waits[p.triage_tier].append(p.start_service_time - p.arrival_time)

        service_times = [p.actual_duration for p in self.completed_patients]
        total_busy_time = sum(service_times)
        total_available_capacity = max(1.0, current_time * self.num_counters)
        counter_utilization = min(1.0, total_busy_time / total_available_capacity)

        metrics = {
            "department": self.department,
            "policy": self.policy,
            "num_counters": self.num_counters,
            "total_patients_served": len(self.completed_patients),
            "unserved_patients": len(self.waiting_pool),
            "sim_duration_mins": round(current_time, 1),
            "avg_wait_mins": round(float(np.mean(wait_times)), 2) if wait_times else 0.0,
            "p50_wait_mins": round(float(np.median(wait_times)), 2) if wait_times else 0.0,
            "p90_wait_mins": round(float(np.percentile(wait_times, 90)), 2) if wait_times else 0.0,
            "max_wait_mins": round(float(np.max(wait_times)), 2) if wait_times else 0.0,
            "counter_utilization": round(float(counter_utilization), 3),
            "tier_avg_wait_mins": {
                f"Tier_{k}": round(float(np.mean(v)), 2) if v else 0.0
                for k, v in tier_waits.items()
            }
        }
        return metrics

def run_comparative_benchmark(
    n_patients: int = 300,
    num_counters: int = 4,
    alpha: float = 2.5,
    beta: float = 12.0
) -> Dict[str, Any]:
    """Runs a direct benchmark comparison: Traditional FIFO vs AIIMS SEPT-AP."""
    np.random.seed(42)

    patients_fifo = []
    patients_sept = []

    current_arr = 0.0
    for i in range(n_patients):
        inter_arrival = np.random.exponential(scale=1.4) # Arrival rate ~0.71 patients/min
        current_arr += inter_arrival
        tier = int(np.random.choice([1, 2, 3, 4], p=[0.02, 0.20, 0.55, 0.23]))
        age = int(np.clip(np.random.normal(45, 18), 5, 85))
        is_senior = age >= 60
        visit_type = np.random.choice(["New_Consultation", "Follow_Up", "Report_Review"], p=[0.6, 0.3, 0.1])
        
        # Expected duration
        meanlog = 2.4
        expected_dur = np.exp(meanlog) * (0.75 if visit_type == "Follow_Up" else (0.45 if visit_type == "Report_Review" else 1.0))
        actual_dur = float(np.clip(np.random.lognormal(np.log(expected_dur), 0.25), 3.0, 35.0))

        p_fifo = QueuedPatient(
            sort_key=0.0,
            token_id=f"TKN-{i+1:04d}",
            triage_tier=tier,
            arrival_time=current_arr,
            expected_duration=expected_dur,
            actual_duration=actual_dur,
            patient_age=age,
            is_senior=is_senior,
            visit_type=visit_type
        )
        p_sept = QueuedPatient(
            sort_key=0.0,
            token_id=f"TKN-{i+1:04d}",
            triage_tier=tier,
            arrival_time=current_arr,
            expected_duration=expected_dur,
            actual_duration=actual_dur,
            patient_age=age,
            is_senior=is_senior,
            visit_type=visit_type
        )
        patients_fifo.append(p_fifo)
        patients_sept.append(p_sept)

    sim_fifo = AIIMSQueueSimulation(num_counters=num_counters, policy="FIFO")
    for p in patients_fifo:
        sim_fifo.add_patient(p)
    fifo_results = sim_fifo.run()

    sim_sept = AIIMSQueueSimulation(num_counters=num_counters, policy="SEPT_AP", alpha=alpha, beta=beta)
    for p in patients_sept:
        sim_sept.add_patient(p)
    sept_results = sim_sept.run()

    # Calculate improvements
    wait_reduction_pct = (fifo_results["avg_wait_mins"] - sept_results["avg_wait_mins"]) / max(0.01, fifo_results["avg_wait_mins"]) * 100
    tier1_improvement_pct = (fifo_results["tier_avg_wait_mins"]["Tier_1"] - sept_results["tier_avg_wait_mins"]["Tier_1"]) / max(0.01, fifo_results["tier_avg_wait_mins"]["Tier_1"]) * 100
    tier2_improvement_pct = (fifo_results["tier_avg_wait_mins"]["Tier_2"] - sept_results["tier_avg_wait_mins"]["Tier_2"]) / max(0.01, fifo_results["tier_avg_wait_mins"]["Tier_2"]) * 100

    return {
        "fifo": fifo_results,
        "sept_ap": sept_results,
        "overall_wait_reduction_pct": round(wait_reduction_pct, 2),
        "tier1_fast_track_reduction_pct": round(tier1_improvement_pct, 2),
        "tier2_urgent_reduction_pct": round(tier2_improvement_pct, 2)
    }

if __name__ == "__main__":
    print("Executing AIIMS Raipur Queue Benchmark Simulation...")
    benchmark = run_comparative_benchmark(n_patients=300, num_counters=4)
    import pprint
    pprint.pprint(benchmark)
