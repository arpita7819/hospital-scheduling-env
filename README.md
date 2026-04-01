---
title: Hospital Scheduling Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
  
# 🏥 HospitalSchedulingEnv

> An OpenEnv-compliant environment for training AI agents on **real-world hospital resource scheduling**.

---

## Overview

`HospitalSchedulingEnv` simulates the moment-to-moment resource allocation decisions made by hospital charge nurses and administrators: assigning ICU beds, general ward beds, operating rooms, ventilators, nurses, and doctors to incoming patients of varying acuity — under realistic time pressure and capacity constraints.

This is not a toy: the environment models real clinical decision patterns including acuity-based triage priority, adverse events from delayed critical care, ventilator scarcity, and discharge-driven capacity recovery.

---

## Environment API

The environment follows the standard OpenEnv `step()` / `reset()` / `state()` interface.

```python
from environment import HospitalSchedulingEnv, HospitalConfig
from tasks import load_task

# Load a task (easy | medium | hard)
env, task = load_task("medium")

# Reset
obs = env.reset(seed=42)

done = False
while not done:
    # Your agent picks an action
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)

# Grade the episode
score = task.grader(env, trajectory)   # float in [0, 1]
```

---

## Action Space

| Field | Type | Values | Description |
|---|---|---|---|
| `patient_id` | int | any waiting patient ID, or `-1` | Patient to schedule. `-1` = no-op |
| `assign_bed_type` | int | `0`=ICU, `1`=General, `2`=None | Bed type to assign |
| `assign_or` | int | `0`/`1` | Whether to schedule an OR room |
| `assign_ventilator` | int | `0`/`1` | Whether to assign a ventilator |

---

## Observation Space

The observation is a flat dict with the following keys:

| Key | Type | Description |
|---|---|---|
| `hour` | int | Current simulation hour |
| `icu_beds_free` | int | Available ICU beds |
| `general_beds_free` | int | Available general beds |
| `or_rooms_free` | int | Available OR rooms |
| `ventilators_free` | int | Available ventilators |
| `nurses_free` | int | Available nurses |
| `doctors_free` | int | Available doctors |
| `n_waiting_critical` | int | Critical patients in queue |
| `n_waiting_high` | int | High-acuity patients in queue |
| `n_waiting_medium` | int | Medium-acuity patients in queue |
| `n_waiting_low` | int | Low-acuity patients in queue |
| `avg_wait_hours` | float | Mean wait time of queued patients |
| `overflow_flag` | int | 1 if any critical patient has waited >2 hours |
| `waiting_patients` | List[Dict] | Full patient records for scheduling decisions |

---

## Reward Function

Rewards are shaped to encourage **safe and efficient** allocation:

| Event | Reward |
|---|---|
| Critical patient correctly assigned to ICU | **+2.0** |
| Appropriate general bed assignment | **+1.0** |
| Ventilator correctly assigned | **+1.5** |
| Patient discharged after recovery | **+0.5** |
| OR scheduled (room available) | **+0.3** |
| Critical patient in general bed | **−2.0** |
| ICU bed wasted on non-critical patient | **−0.5** |
| Unnecessary ventilator use | **−0.5** |
| Each hour a patient waits | **−0.1** |
| **Adverse event** (critical wait >2h) | **−3.0** |
| Staff shortage on admission | **−0.5** |

---

## Tasks

### 🟢 Easy — `basic_triage`
- **Episode length**: 48 hours (2 days)
- **Patient load**: Low (1–3 per hour)
- **Focus**: Correct bed-type triage (ICU vs general)
- **Graded on**: Triage accuracy (70%) + adverse event rate (30%)

### 🟡 Medium — `full_scheduling`
- **Episode length**: 96 hours (4 days)
- **Patient load**: Moderate (1–5 per hour)
- **Focus**: All resource types, staff constraints active
- **Graded on**: Triage (35%) + utilisation (25%) + adverse events (25%) + ventilator appropriateness (15%)

### 🔴 Hard — `crisis_surge`
- **Episode length**: 168 hours (1 week)
- **Patient load**: High surge (1–6 per hour)
- **Capacity**: Reduced ICU (8 beds), ventilators (5), doctors (5)
- **Focus**: Ruthless prioritisation, early discharges, prevent critical waits
- **Graded on**: Triage (25%) + utilisation (20%) + adverse events (30%) + throughput (15%) + overflow (10%)

---

## Baseline Scores

| Difficulty | Random | Greedy FIFO | Heuristic |
|---|---|---|---|
| Easy | 0.41 ± 0.06 | 0.74 ± 0.04 | **0.82 ± 0.03** |
| Medium | 0.32 ± 0.07 | 0.61 ± 0.05 | **0.71 ± 0.04** |
| Hard | 0.19 ± 0.08 | 0.47 ± 0.07 | **0.56 ± 0.06** |

Scores are in `[0, 1]`. Run your own benchmark:

```bash
python baseline.py
```

---

## Setup

### Local

```bash
# No dependencies — pure Python 3.10+
python baseline.py          # Run benchmark
python -c "
from tasks import load_task
env, task = load_task('easy')
obs = env.reset()
print(obs)
"
```

### Docker / Hugging Face Spaces

```bash
docker build -t hospital-env .
docker run -p 7860:7860 hospital-env
```

---

## File Structure

```
hospital_env/
├── environment.py     # Core HospitalSchedulingEnv class
├── tasks.py           # Three graded tasks + grader functions
├── baseline.py        # Baseline agents + benchmark runner
├── openenv.yaml       # OpenEnv spec manifest
├── Dockerfile         # HF Spaces / Docker deployment
└── README.md          # This file
```

---

## Design Notes

- **Pure stdlib**: No numpy, gym, or third-party dependencies. Runs anywhere Python 3.10+ is available.
- **Deterministic**: All randomness is seeded via `random.Random(seed)`.
- **Partial observability**: The agent sees aggregate counts + a waiting list but not the full admitted-patient state (mimicking real clinical dashboards).
- **Realistic arrival model**: Patient arrivals follow a diurnal pattern (higher during day hours 8–18).
- **Partial progress signals**: Every correct decision yields positive reward immediately — not just at episode end.

---

## Citation

```bibtex
@misc{hospitalschedulingenv2026,
  title  = {HospitalSchedulingEnv: An OpenEnv Environment for Hospital Resource Scheduling},
  year   = {2026},
  note   = {OpenEnv competition submission}
}
```
