"""
HospitalSchedulingEnv - OpenEnv-compliant Hospital Resource Scheduling Environment
Simulates a real hospital resource allocation problem for AI agent training.
"""

from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import IntEnum


# ─── Action / Observation Types ───────────────────────────────────────────────

class ResourceType(IntEnum):
    ICU_BED       = 0
    GENERAL_BED   = 1
    OR_ROOM       = 2
    VENTILATOR    = 3
    NURSE         = 4
    DOCTOR        = 5

class PatientAcuity(IntEnum):
    CRITICAL  = 0   # Must be in ICU; needs ventilator possibly
    HIGH      = 1   # Needs close monitoring
    MEDIUM    = 2   # Standard ward
    LOW       = 3   # Routine / elective

ACUITY_LABEL = {0: "CRITICAL", 1: "HIGH", 2: "MEDIUM", 3: "LOW"}


@dataclass
class Patient:
    id: int
    acuity: PatientAcuity
    arrival_hour: int
    los_remaining: int          # length-of-stay hours remaining
    assigned_bed: Optional[int] = None    # bed index or None
    assigned_or: Optional[int]  = None
    needs_ventilator: bool = False
    assigned_ventilator: bool = False
    wait_hours: int = 0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "acuity": int(self.acuity),
            "acuity_label": ACUITY_LABEL[int(self.acuity)],
            "arrival_hour": self.arrival_hour,
            "los_remaining": self.los_remaining,
            "assigned_bed": self.assigned_bed,
            "assigned_or": self.assigned_or,
            "needs_ventilator": self.needs_ventilator,
            "assigned_ventilator": self.assigned_ventilator,
            "wait_hours": self.wait_hours,
        }


@dataclass
class HospitalConfig:
    """Static hospital capacity configuration."""
    n_icu_beds:     int = 10
    n_general_beds: int = 40
    n_or_rooms:     int = 5
    n_ventilators:  int = 8
    n_nurses:       int = 20
    n_doctors:      int = 8
    max_patients_per_step: int = 6   # new arrivals sampled per hour
    max_steps: int = 168             # 1 week in hours


# ─── Main Environment ─────────────────────────────────────────────────────────

class HospitalSchedulingEnv:
    """
    OpenEnv-compliant environment for hospital resource scheduling.

    Observation (flat dict, all int/float):
        - hour: current simulation hour
        - icu_beds_free, general_beds_free, or_rooms_free,
          ventilators_free, nurses_free, doctors_free  (capacities)
        - n_waiting_critical, n_waiting_high, n_waiting_medium, n_waiting_low
        - n_admitted_critical, n_admitted_high, n_admitted_medium, n_admitted_low
        - avg_wait_hours: average wait time of currently-waiting patients
        - overflow_flag: 1 if any critical patient has been waiting > 2 h

    Action (dict):
        {
          "patient_id": int,          # which waiting patient to schedule next
          "assign_bed_type": int,     # 0=ICU, 1=General, 2=None (discharge-only)
          "assign_or": int,           # 0=no OR, 1=schedule OR
          "assign_ventilator": int,   # 0=no, 1=yes
        }
        Special no-op: patient_id = -1  (hold / do nothing this step)
    """

    metadata = {
        "name": "HospitalSchedulingEnv",
        "version": "1.0.0",
        "observation_space": "Dict[str, float]",
        "action_space": "Dict[str, int]",
    }

    def __init__(self, config: Optional[HospitalConfig] = None, seed: int = 42):
        self.cfg   = config or HospitalConfig()
        self._seed = seed
        self._rng  = random.Random(seed)
        self._step_count  = 0
        self._patient_ctr = 0

        # Resource pools (available count)
        self._icu_free  = self.cfg.n_icu_beds
        self._gen_free  = self.cfg.n_general_beds
        self._or_free   = self.cfg.n_or_rooms
        self._vent_free = self.cfg.n_ventilators
        self._nur_free  = self.cfg.n_nurses
        self._doc_free  = self.cfg.n_doctors

        self._waiting : List[Patient] = []
        self._admitted: List[Patient] = []
        self._discharged_count = 0
        self._total_wait_hours = 0.0
        self._adverse_events   = 0   # critical patients who waited too long

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment and return initial observation."""
        if seed is not None:
            self._seed = seed
        self._rng  = random.Random(self._seed)
        self._step_count  = 0
        self._patient_ctr = 0

        self._icu_free  = self.cfg.n_icu_beds
        self._gen_free  = self.cfg.n_general_beds
        self._or_free   = self.cfg.n_or_rooms
        self._vent_free = self.cfg.n_ventilators
        self._nur_free  = self.cfg.n_nurses
        self._doc_free  = self.cfg.n_doctors

        self._waiting           = []
        self._admitted          = []
        self._discharged_count  = 0
        self._total_wait_hours  = 0.0
        self._adverse_events    = 0

        # Seed with a few initial patients
        for _ in range(self._rng.randint(3, 7)):
            self._waiting.append(self._new_patient())

        return self.state()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one hour step.
        Returns: (observation, reward, done, info)
        """
        assert self._step_count < self.cfg.max_steps, "Episode already done. Call reset()."

        # 1. Process agent action
        reward_action = self._process_action(action)

        # 2. Advance time: discharge recovered patients, age wait times
        reward_discharge = self._advance_time()

        # 3. Generate new arrivals
        self._generate_arrivals()

        self._step_count += 1
        done = self._step_count >= self.cfg.max_steps

        reward = reward_action + reward_discharge
        info   = {
            "step": self._step_count,
            "discharged_total": self._discharged_count,
            "adverse_events":   self._adverse_events,
            "total_wait_hours": self._total_wait_hours,
        }
        return self.state(), reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return current observation as a flat dict."""
        w_counts = {a: 0 for a in range(4)}
        a_counts = {a: 0 for a in range(4)}
        for p in self._waiting:
            w_counts[int(p.acuity)] += 1
        for p in self._admitted:
            a_counts[int(p.acuity)] += 1

        total_wait  = sum(p.wait_hours for p in self._waiting)
        avg_wait    = total_wait / max(len(self._waiting), 1)
        overflow    = any(p.acuity == PatientAcuity.CRITICAL and p.wait_hours > 2
                          for p in self._waiting)

        return {
            "hour":                  self._step_count,
            "icu_beds_free":         self._icu_free,
            "general_beds_free":     self._gen_free,
            "or_rooms_free":         self._or_free,
            "ventilators_free":      self._vent_free,
            "nurses_free":           self._nur_free,
            "doctors_free":          self._doc_free,
            "n_waiting_critical":    w_counts[0],
            "n_waiting_high":        w_counts[1],
            "n_waiting_medium":      w_counts[2],
            "n_waiting_low":         w_counts[3],
            "n_admitted_critical":   a_counts[0],
            "n_admitted_high":       a_counts[1],
            "n_admitted_medium":     a_counts[2],
            "n_admitted_low":        a_counts[3],
            "avg_wait_hours":        round(avg_wait, 2),
            "overflow_flag":         int(overflow),
            # expose waiting list for agent decision-making
            "waiting_patients":      [p.to_dict() for p in self._waiting],
        }

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _new_patient(self) -> Patient:
        self._patient_ctr += 1
        acuity = self._rng.choices(
            [PatientAcuity.CRITICAL, PatientAcuity.HIGH,
             PatientAcuity.MEDIUM,  PatientAcuity.LOW],
            weights=[5, 20, 40, 35]
        )[0]
        los_map = {
            PatientAcuity.CRITICAL: (24, 120),
            PatientAcuity.HIGH:     (12,  72),
            PatientAcuity.MEDIUM:   (6,   48),
            PatientAcuity.LOW:      (2,   24),
        }
        lo, hi = los_map[acuity]
        los = self._rng.randint(lo, hi)
        needs_vent = (acuity == PatientAcuity.CRITICAL and self._rng.random() < 0.4)
        return Patient(
            id=self._patient_ctr,
            acuity=acuity,
            arrival_hour=self._step_count,
            los_remaining=los,
            needs_ventilator=needs_vent,
        )

    def _process_action(self, action: Dict[str, Any]) -> float:
        patient_id   = action.get("patient_id", -1)
        bed_type     = action.get("assign_bed_type", 2)   # 0=ICU,1=Gen,2=None
        assign_or    = action.get("assign_or", 0)
        assign_vent  = action.get("assign_ventilator", 0)

        if patient_id == -1:
            return 0.0   # no-op

        patient = next((p for p in self._waiting if p.id == patient_id), None)
        if patient is None:
            return -0.5  # tried to assign a non-waiting patient

        reward = 0.0

        # Validate bed assignment
        can_assign = False
        if bed_type == 0 and self._icu_free > 0:
            self._icu_free -= 1
            patient.assigned_bed = bed_type
            can_assign = True
            # reward for correctly assigning critical to ICU
            if patient.acuity == PatientAcuity.CRITICAL:
                reward += 2.0
            else:
                reward -= 0.5   # wasted ICU bed
        elif bed_type == 1 and self._gen_free > 0:
            self._gen_free -= 1
            patient.assigned_bed = bed_type
            can_assign = True
            if patient.acuity in (PatientAcuity.MEDIUM, PatientAcuity.LOW):
                reward += 1.0
            elif patient.acuity == PatientAcuity.HIGH:
                reward += 0.5
            else:
                reward -= 2.0   # critical in general bed — dangerous!

        if not can_assign:
            return -1.0  # no bed available of requested type

        # OR room
        if assign_or and self._or_free > 0:
            self._or_free -= 1
            patient.assigned_or = 1
            reward += 0.3
        elif assign_or:
            reward -= 0.2   # wanted OR but none free

        # Ventilator
        if assign_vent and patient.needs_ventilator and self._vent_free > 0:
            self._vent_free -= 1
            patient.assigned_ventilator = True
            reward += 1.5
        elif assign_vent and not patient.needs_ventilator:
            reward -= 0.5   # unnecessary ventilator use

        # Nurse/doctor consumption (1 nurse + doctor per admission)
        if self._nur_free > 0 and self._doc_free > 0:
            self._nur_free -= 1
            self._doc_free -= 1
        else:
            reward -= 0.5   # staff stretched

        # Penalise long waits
        reward -= patient.wait_hours * 0.1
        self._total_wait_hours += patient.wait_hours

        # Move to admitted
        self._waiting.remove(patient)
        self._admitted.append(patient)

        return reward

    def _advance_time(self) -> float:
        reward = 0.0
        still_admitted = []
        for p in self._admitted:
            p.los_remaining -= 1
            if p.los_remaining <= 0:
                # Discharge — free resources
                if p.assigned_bed == 0:
                    self._icu_free  = min(self._icu_free  + 1, self.cfg.n_icu_beds)
                elif p.assigned_bed == 1:
                    self._gen_free  = min(self._gen_free  + 1, self.cfg.n_general_beds)
                if p.assigned_or is not None:
                    self._or_free   = min(self._or_free   + 1, self.cfg.n_or_rooms)
                if p.assigned_ventilator:
                    self._vent_free = min(self._vent_free + 1, self.cfg.n_ventilators)
                self._nur_free = min(self._nur_free + 1, self.cfg.n_nurses)
                self._doc_free = min(self._doc_free + 1, self.cfg.n_doctors)
                self._discharged_count += 1
                reward += 0.5
            else:
                still_admitted.append(p)

        self._admitted = still_admitted

        # Age waiting patients
        for p in self._waiting:
            p.wait_hours += 1
            if p.acuity == PatientAcuity.CRITICAL and p.wait_hours > 2:
                self._adverse_events += 1
                reward -= 3.0   # severe penalty for critical patients waiting

        return reward

    def _generate_arrivals(self):
        hour_of_day = self._step_count % 24
        # Poisson-like: higher arrivals during day
        if 8 <= hour_of_day <= 18:
            n = self._rng.randint(1, self.cfg.max_patients_per_step)
        else:
            n = self._rng.randint(0, 2)
        for _ in range(n):
            self._waiting.append(self._new_patient())
