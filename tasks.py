"""
tasks.py — Three graded tasks for HospitalSchedulingEnv.

Each task returns a TaskSpec with:
  - description
  - difficulty
  - a grader function: grader(env, trajectory) -> score in [0, 1]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Any
from environment import HospitalSchedulingEnv, HospitalConfig, PatientAcuity


@dataclass
class TaskSpec:
    name: str
    difficulty: str          # "easy" | "medium" | "hard"
    description: str
    config: HospitalConfig
    grader: Callable         # grader(env, trajectory) -> float in [0,1]
    max_steps: int


# ─── Grader Helpers ────────────────────────────────────────────────────────────

def _triage_accuracy(trajectory: List[Dict]) -> float:
    """Fraction of critical patients assigned to ICU beds."""
    critical_admitted_correct = 0
    critical_admitted_total   = 0
    for step in trajectory:
        action = step.get("action", {})
        patient = step.get("patient_snapshot")
        if patient and patient.get("acuity") == int(PatientAcuity.CRITICAL):
            critical_admitted_total += 1
            if action.get("assign_bed_type") == 0:   # ICU
                critical_admitted_correct += 1
    if critical_admitted_total == 0:
        return 1.0  # no critical patients → trivially correct
    return critical_admitted_correct / critical_admitted_total


def _occupancy_efficiency(env: HospitalSchedulingEnv) -> float:
    """How well the agent kept beds filled (utilisation vs overcrowding)."""
    discharged = env._discharged_count
    max_possible = env.cfg.max_steps * 0.8   # rough upper bound
    utilisation = min(discharged / max(max_possible, 1), 1.0)
    return utilisation


def _adverse_event_rate(env: HospitalSchedulingEnv) -> float:
    """Score drops linearly with adverse events (critical waits > 2 h)."""
    events = env._adverse_events
    # allow up to 5 events before score hits 0
    return max(0.0, 1.0 - events / 10.0)


# ─── Task 1: EASY — Basic Triage ───────────────────────────────────────────────

def grader_easy(env: HospitalSchedulingEnv, trajectory: List[Dict]) -> float:
    """
    Score purely on whether the agent correctly triages critical patients to ICU.
    Secondary: not leaving critical patients waiting > 4 h.
    """
    triage_score  = _triage_accuracy(trajectory)
    adverse_score = _adverse_event_rate(env)
    return round(0.7 * triage_score + 0.3 * adverse_score, 4)

TASK_EASY = TaskSpec(
    name="basic_triage",
    difficulty="easy",
    description=(
        "A lightly-loaded hospital with mostly routine and high-acuity patients. "
        "The agent must correctly assign patients to appropriate bed types (ICU for CRITICAL, "
        "general for MEDIUM/LOW). No OR or ventilator decisions needed. "
        "Scored on triage accuracy and avoiding adverse events."
    ),
    config=HospitalConfig(
        n_icu_beds=10,
        n_general_beds=40,
        n_or_rooms=5,
        n_ventilators=8,
        n_nurses=20,
        n_doctors=8,
        max_patients_per_step=3,   # low load
        max_steps=48,              # 2-day episode
    ),
    grader=grader_easy,
    max_steps=48,
)


# ─── Task 2: MEDIUM — Full Scheduling ─────────────────────────────────────────

def grader_medium(env: HospitalSchedulingEnv, trajectory: List[Dict]) -> float:
    """
    Score on triage accuracy + resource utilisation + adverse events.
    Agent must now also handle OR assignment and ventilator allocation.
    """
    triage_score  = _triage_accuracy(trajectory)
    util_score    = _occupancy_efficiency(env)
    adverse_score = _adverse_event_rate(env)

    # Check ventilator appropriateness
    vent_ok   = 0
    vent_total = 0
    for step in trajectory:
        patient = step.get("patient_snapshot")
        action  = step.get("action", {})
        if patient and action.get("assign_ventilator") == 1:
            vent_total += 1
            if patient.get("needs_ventilator"):
                vent_ok += 1
    vent_score = (vent_ok / max(vent_total, 1)) if vent_total > 0 else 0.8

    return round(
        0.35 * triage_score +
        0.25 * util_score   +
        0.25 * adverse_score +
        0.15 * vent_score,
        4
    )

TASK_MEDIUM = TaskSpec(
    name="full_scheduling",
    difficulty="medium",
    description=(
        "A moderately busy hospital. The agent must handle full resource allocation: "
        "bed type selection (ICU vs general), OR room scheduling, and ventilator assignment. "
        "Staff constraints (nurses, doctors) now bind. "
        "Scored on triage accuracy, utilisation, ventilator appropriateness, and adverse events."
    ),
    config=HospitalConfig(
        n_icu_beds=10,
        n_general_beds=40,
        n_or_rooms=5,
        n_ventilators=8,
        n_nurses=20,
        n_doctors=8,
        max_patients_per_step=5,
        max_steps=96,              # 4-day episode
    ),
    grader=grader_medium,
    max_steps=96,
)


# ─── Task 3: HARD — Crisis Surge ──────────────────────────────────────────────

def grader_hard(env: HospitalSchedulingEnv, trajectory: List[Dict]) -> float:
    """
    Score on all dimensions under surge conditions.
    Penalties for overflow, adverse events, and wasted scarce resources are amplified.
    Bonus for efficiently discharging patients to free capacity.
    """
    triage_score  = _triage_accuracy(trajectory)
    util_score    = _occupancy_efficiency(env)
    adverse_score = _adverse_event_rate(env)   # weight doubled vs medium

    # Discharge throughput bonus
    discharged  = env._discharged_count
    throughput  = min(discharged / max(env.cfg.max_steps * 0.5, 1), 1.0)

    # Overflow penalty: did the agent ever leave critical patients waiting >4h?
    severe_overflows = sum(
        1 for step in trajectory
        if step.get("obs", {}).get("overflow_flag") == 1
           and step.get("obs", {}).get("n_waiting_critical", 0) > 0
    )
    overflow_penalty = max(0.0, 1.0 - severe_overflows / 20.0)

    return round(
        0.25 * triage_score    +
        0.20 * util_score      +
        0.30 * adverse_score   +   # heavy weight on patient safety
        0.15 * throughput      +
        0.10 * overflow_penalty,
        4
    )

TASK_HARD = TaskSpec(
    name="crisis_surge",
    difficulty="hard",
    description=(
        "Mass-casualty surge: the hospital is overwhelmed with high patient volumes, "
        "constrained ICU and ventilator capacity, and a mix of critical and routine patients. "
        "The agent must prioritise ruthlessly: discharge stable patients early, prevent "
        "critical patients from waiting, and manage scarce ventilators wisely. "
        "Scored on all dimensions with heavier weight on patient safety (adverse events)."
    ),
    config=HospitalConfig(
        n_icu_beds=8,            # reduced ICU capacity
        n_general_beds=30,
        n_or_rooms=3,
        n_ventilators=5,         # very scarce
        n_nurses=14,
        n_doctors=5,
        max_patients_per_step=6,  # high surge load
        max_steps=168,            # full week episode
    ),
    grader=grader_hard,
    max_steps=168,
)


# ─── Registry ─────────────────────────────────────────────────────────────────

TASKS: Dict[str, TaskSpec] = {
    "easy":   TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard":   TASK_HARD,
}


def load_task(difficulty: str) -> tuple[HospitalSchedulingEnv, TaskSpec]:
    """Instantiate env + task for a given difficulty level."""
    task = TASKS[difficulty]
    env  = HospitalSchedulingEnv(config=task.config)
    return env, task
