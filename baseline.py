"""
baseline.py — Baseline agents + reproducible benchmark runner.

Run:
    python baseline.py

Produces a score table for each difficulty level using three baseline strategies:
  1. Random agent
  2. Rule-based heuristic agent (priority queue by acuity)
  3. Greedy agent (first-in-first-out with correct triage)
"""

from __future__ import annotations
import random
import json
from typing import Dict, Any, List, Tuple
from environment import HospitalSchedulingEnv, PatientAcuity
from tasks import load_task, TaskSpec


# ─── Agent Implementations ────────────────────────────────────────────────────

class RandomAgent:
    """Randomly picks a waiting patient and random resources."""
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        waiting = obs.get("waiting_patients", [])
        if not waiting:
            return {"patient_id": -1, "assign_bed_type": 2,
                    "assign_or": 0, "assign_ventilator": 0}
        patient = self._rng.choice(waiting)
        return {
            "patient_id":       patient["id"],
            "assign_bed_type":  self._rng.randint(0, 1),
            "assign_or":        self._rng.randint(0, 1),
            "assign_ventilator": self._rng.randint(0, 1),
        }


class HeuristicAgent:
    """
    Priority-queue heuristic:
      - Always pick the highest-acuity waiting patient.
      - Assign ICU for CRITICAL/HIGH, general for MEDIUM/LOW.
      - Assign ventilator iff patient needs one and one is free.
      - Schedule OR if ventilator-free and OR available.
    """
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        waiting = obs.get("waiting_patients", [])
        if not waiting:
            return {"patient_id": -1, "assign_bed_type": 2,
                    "assign_or": 0, "assign_ventilator": 0}

        # Sort by acuity (0=critical = highest priority), then wait_hours
        waiting_sorted = sorted(waiting, key=lambda p: (p["acuity"], -p["wait_hours"]))
        patient = waiting_sorted[0]

        acuity = patient["acuity"]
        if acuity in (0, 1):   # CRITICAL or HIGH
            bed_type = 0 if obs["icu_beds_free"] > 0 else 1
        else:
            bed_type = 1

        assign_vent = (
            1 if patient["needs_ventilator"] and obs["ventilators_free"] > 0
            else 0
        )
        assign_or = (
            1 if obs["or_rooms_free"] > 0 and acuity <= 1
            else 0
        )

        return {
            "patient_id":        patient["id"],
            "assign_bed_type":   bed_type,
            "assign_or":         assign_or,
            "assign_ventilator": assign_vent,
        }


class GreedyFIFOAgent:
    """
    FIFO with correct triage: processes patients in arrival order
    but uses proper bed-type rules.
    """
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        waiting = obs.get("waiting_patients", [])
        if not waiting:
            return {"patient_id": -1, "assign_bed_type": 2,
                    "assign_or": 0, "assign_ventilator": 0}

        patient = min(waiting, key=lambda p: p["arrival_hour"])
        acuity  = patient["acuity"]
        bed_type = 0 if acuity == 0 and obs["icu_beds_free"] > 0 else 1

        assign_vent = 1 if patient["needs_ventilator"] and obs["ventilators_free"] > 0 else 0

        return {
            "patient_id":        patient["id"],
            "assign_bed_type":   bed_type,
            "assign_or":         0,
            "assign_ventilator": assign_vent,
        }


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_episode(
    env: HospitalSchedulingEnv,
    task: TaskSpec,
    agent,
    seed: int = 42,
) -> Tuple[float, List[Dict]]:
    """Run one full episode; return (grader_score, trajectory)."""
    obs = env.reset(seed=seed)
    trajectory: List[Dict] = []
    done = False

    while not done:
        action = agent.act(obs)

        # Snapshot patient for grader
        patient_snap = None
        for p in obs.get("waiting_patients", []):
            if p["id"] == action.get("patient_id"):
                patient_snap = p
                break

        next_obs, reward, done, info = env.step(action)
        trajectory.append({
            "obs":              obs,
            "action":           action,
            "reward":           reward,
            "patient_snapshot": patient_snap,
            "info":             info,
        })
        obs = next_obs

    score = task.grader(env, trajectory)
    return score, trajectory


def benchmark(n_seeds: int = 5) -> Dict:
    """Run all agents on all difficulties and report mean ± std scores."""
    import statistics
    agents = {
        "random":   RandomAgent,
        "heuristic": HeuristicAgent,
        "greedy_fifo": GreedyFIFOAgent,
    }
    difficulties = ["easy", "medium", "hard"]
    results: Dict = {}

    for diff in difficulties:
        results[diff] = {}
        for agent_name, AgentClass in agents.items():
            scores = []
            for seed in range(n_seeds):
                env, task = load_task(diff)
                agent = AgentClass() if agent_name == "random" else AgentClass()
                score, _ = run_episode(env, task, agent, seed=seed)
                scores.append(score)
            mean = round(statistics.mean(scores), 4)
            std  = round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 4)
            results[diff][agent_name] = {"mean": mean, "std": std, "scores": scores}

    return results


def print_results(results: Dict):
    print("\n" + "=" * 64)
    print("  HOSPITAL SCHEDULING ENV — BASELINE BENCHMARK")
    print("=" * 64)
    header = f"{'Difficulty':<10} {'Agent':<15} {'Mean Score':>12} {'Std':>8}"
    print(header)
    print("-" * 64)
    for diff in ["easy", "medium", "hard"]:
        for agent_name, stats in results[diff].items():
            print(f"{diff:<10} {agent_name:<15} {stats['mean']:>12.4f} {stats['std']:>8.4f}")
        print()
    print("=" * 64)
    print("Scores are in [0, 1]. Higher is better.")


if __name__ == "__main__":
    print("Running baseline benchmark (5 seeds × 3 agents × 3 difficulties)...")
    results = benchmark(n_seeds=5)
    print_results(results)

    # Save JSON
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to baseline_results.json")
