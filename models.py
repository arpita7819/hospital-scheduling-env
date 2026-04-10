"""
models.py — Typed Action, Observation, and State models for HospitalSchedulingEnv.
Follows the OpenEnv spec: pydantic models with clear field descriptions.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HospitalAction(BaseModel):
    """Action for the hospital scheduling environment."""
    patient_id: int = Field(
        default=-1,
        description="ID of the waiting patient to schedule. Use -1 for no-op."
    )
    assign_bed_type: int = Field(
        default=2,
        description="Bed type to assign: 0=ICU, 1=General, 2=None"
    )
    assign_or: int = Field(
        default=0,
        description="Whether to schedule an OR room: 0=No, 1=Yes"
    )
    assign_ventilator: int = Field(
        default=0,
        description="Whether to assign a ventilator: 0=No, 1=Yes"
    )


class PatientRecord(BaseModel):
    """A single patient in the waiting queue."""
    id: int
    acuity: int = Field(description="0=Critical, 1=High, 2=Medium, 3=Low")
    acuity_label: str
    arrival_hour: int
    los_remaining: int
    assigned_bed: Optional[int] = None
    assigned_or: Optional[int] = None
    needs_ventilator: bool
    assigned_ventilator: bool
    wait_hours: int


class HospitalObservation(BaseModel):
    """Observation from the hospital scheduling environment."""
    hour: int = Field(description="Current simulation hour")
    icu_beds_free: int = Field(description="Available ICU beds")
    general_beds_free: int = Field(description="Available general beds")
    or_rooms_free: int = Field(description="Available OR rooms")
    ventilators_free: int = Field(description="Available ventilators")
    nurses_free: int = Field(description="Available nurses")
    doctors_free: int = Field(description="Available doctors")
    n_waiting_critical: int = Field(description="Critical patients in queue")
    n_waiting_high: int = Field(description="High-acuity patients in queue")
    n_waiting_medium: int = Field(description="Medium-acuity patients in queue")
    n_waiting_low: int = Field(description="Low-acuity patients in queue")
    n_admitted_critical: int = Field(description="Critical patients admitted")
    n_admitted_high: int = Field(description="High-acuity patients admitted")
    n_admitted_medium: int = Field(description="Medium-acuity patients admitted")
    n_admitted_low: int = Field(description="Low-acuity patients admitted")
    avg_wait_hours: float = Field(description="Mean wait time of queued patients")
    overflow_flag: int = Field(description="1 if critical patients waiting >2h")
    waiting_patients: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full patient records for scheduling decisions"
    )


class HospitalState(BaseModel):
    """Full internal state of the hospital environment."""
    episode_id: str
    step_count: int
    difficulty: str
    seed: int
    discharged_count: int = 0
    adverse_events: int = 0
    total_wait_hours: float = 0.0
    done: bool = False
