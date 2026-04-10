"""
server/hospital_environment.py — OpenEnv Environment implementation.
Wraps HospitalSchedulingEnv in the proper OpenEnv server interface.
"""
from uuid import uuid4
from typing import Tuple

try:
    from ..models import HospitalAction, HospitalObservation, HospitalState
except ImportError:
    from models import HospitalAction, HospitalObservation, HospitalState

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import HospitalSchedulingEnv as _CoreEnv
from tasks import load_task, TASKS


class HospitalEnvironment:
    """
    OpenEnv-compatible hospital resource scheduling environment.

    Wraps the core HospitalSchedulingEnv and exposes reset/step/state
    following the OpenEnv Environment interface pattern.
    """

    def __init__(self, difficulty: str = "hard"):
        self.difficulty = difficulty
        self._env = None
        self._task = None
        self._state = HospitalState(
            episode_id=str(uuid4()),
            step_count=0,
            difficulty=difficulty,
            seed=42,
        )

    def reset(self, seed: int = 42, difficulty: str = None) -> HospitalObservation:
        diff = difficulty or self.difficulty
        self._env, self._task = load_task(diff)
        raw_obs = self._env.reset(seed=seed)
        self._state = HospitalState(
            episode_id=str(uuid4()),
            step_count=0,
            difficulty=diff,
            seed=seed,
        )
        return self._obs_to_model(raw_obs)

    def step(self, action: HospitalAction) -> Tuple[HospitalObservation, float, bool, dict]:
        if self._env is None:
            self.reset()

        raw_action = {
            "patient_id":        action.patient_id,
            "assign_bed_type":   action.assign_bed_type,
            "assign_or":         action.assign_or,
            "assign_ventilator": action.assign_ventilator,
        }
        raw_obs, reward, done, info = self._env.step(raw_action)

        self._state.step_count += 1
        self._state.done = done
        self._state.discharged_count = self._env._discharged_count
        self._state.adverse_events = self._env._adverse_events
        self._state.total_wait_hours = self._env._total_wait_hours

        return self._obs_to_model(raw_obs), reward, done, info

    def state(self) -> HospitalState:
        return self._state

    def get_observation(self) -> HospitalObservation:
        if self._env is None:
            self.reset()
        return self._obs_to_model(self._env.state())

    def get_task_info(self) -> dict:
        return {
            task_name: {
                "difficulty": spec.difficulty,
                "description": spec.description,
                "max_steps": spec.max_steps,
            }
            for task_name, spec in TASKS.items()
        }

    @staticmethod
    def _obs_to_model(raw: dict) -> HospitalObservation:
        return HospitalObservation(
            hour=raw["hour"],
            icu_beds_free=raw["icu_beds_free"],
            general_beds_free=raw["general_beds_free"],
            or_rooms_free=raw["or_rooms_free"],
            ventilators_free=raw["ventilators_free"],
            nurses_free=raw["nurses_free"],
            doctors_free=raw["doctors_free"],
            n_waiting_critical=raw["n_waiting_critical"],
            n_waiting_high=raw["n_waiting_high"],
            n_waiting_medium=raw["n_waiting_medium"],
            n_waiting_low=raw["n_waiting_low"],
            n_admitted_critical=raw["n_admitted_critical"],
            n_admitted_high=raw["n_admitted_high"],
            n_admitted_medium=raw["n_admitted_medium"],
            n_admitted_low=raw["n_admitted_low"],
            avg_wait_hours=raw["avg_wait_hours"],
            overflow_flag=raw["overflow_flag"],
            waiting_patients=raw.get("waiting_patients", []),
        )
