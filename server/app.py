"""
server/app.py — FastAPI server for HospitalSchedulingEnv.
Matches the real OpenEnv server pattern with FastAPI + proper endpoints.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import threading
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from ..models import HospitalAction, HospitalObservation, HospitalState
    from .hospital_environment import HospitalEnvironment
except ImportError:
    from models import HospitalAction, HospitalObservation, HospitalState
    from server.hospital_environment import HospitalEnvironment

# ── Global env instance (single session for HF Spaces) ─────────────────────
_env = HospitalEnvironment(difficulty="hard")
_trajectory = []
_results = None
_lock = threading.Lock()


def run_benchmark():
    global _results
    from baseline import benchmark
    _results = benchmark(n_seeds=5)
    print("Benchmark complete.", flush=True)


threading.Thread(target=run_benchmark, daemon=True).start()

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hospital Scheduling Environment",
    description="OpenEnv-compliant hospital resource scheduling environment.",
    version="1.0.0",
)


# ── Request/Response schemas ────────────────────────────────────────────────
class ResetRequest(BaseModel):
    seed: int = 42
    difficulty: str = "hard"


class StepRequest(BaseModel):
    patient_id: int = -1
    assign_bed_type: int = 2
    assign_or: int = 0
    assign_ventilator: int = 0


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "service": "hospital-scheduling-env"}


@app.get("/validate")
def validate():
    return {
        "valid": True,
        "name": "HospitalSchedulingEnv",
        "version": "1.0.0",
        "sdk": "openenv",
        "endpoints": ["/reset", "/step", "/state", "/validate", "/health"],
        "tasks": ["basic_triage", "full_scheduling", "crisis_surge"],
        "observation_space": "HospitalObservation",
        "action_space": "HospitalAction",
    }


@app.post("/reset")
def reset(req: ResetRequest):
    global _trajectory
    with _lock:
        obs = _env.reset(seed=req.seed, difficulty=req.difficulty)
        _trajectory = []
    return {
        "observation": obs.model_dump(),
        "seed": req.seed,
        "difficulty": req.difficulty,
        "episode_id": _env.state().episode_id,
    }


@app.post("/step")
def step(req: StepRequest):
    action = HospitalAction(
        patient_id=req.patient_id,
        assign_bed_type=req.assign_bed_type,
        assign_or=req.assign_or,
        assign_ventilator=req.assign_ventilator,
    )
    with _lock:
        obs, reward, done, info = _env.step(action)
        _trajectory.append({
            "action": action.model_dump(),
            "reward": reward,
            "done": done,
        })
        score = None
        if done:
            from tasks import load_task
            _, task = load_task(_env.difficulty)
            score = task.grader(_env._env, _trajectory)

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
        "score": score,
        "step_count": _env.state().step_count,
    }


@app.get("/state")
def state():
    with _lock:
        obs = _env.get_observation()
        st = _env.state()
    return {
        "observation": obs.model_dump(),
        "state": st.model_dump(),
    }


@app.get("/tasks")
def tasks():
    return _env.get_task_info()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if _results is None:
        return HTMLResponse(content=_loading_html(), status_code=200)
    return HTMLResponse(content=_results_html(_results), status_code=200)


# ── HTML helpers ─────────────────────────────────────────────────────────────
def _loading_html():
    return """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="4">
<title>Hospital Scheduling Env</title>
<style>body{font-family:monospace;background:#0d1117;color:#e6edf3;
max-width:800px;margin:60px auto;text-align:center}h1{color:#58a6ff}
.sp{display:inline-block;width:36px;height:36px;border:4px solid #30363d;
border-top-color:#58a6ff;border-radius:50%;animation:spin 1s linear infinite;margin:20px}
@keyframes spin{to{transform:rotate(360deg)}}p{color:#8b949e}</style>
</head><body><h1>🏥 Hospital Scheduling Env</h1>
<div class="sp"></div><p>Running benchmark...</p></body></html>"""


def _results_html(res):
    rows = ""
    for diff in ["easy", "medium", "hard"]:
        col = {"easy": "#3fb950", "medium": "#d29922", "hard": "#f85149"}[diff]
        best = max(v["mean"] for v in res[diff].values())
        for agent in ["random", "greedy_fifo", "heuristic"]:
            s = res[diff][agent]
            bold = "font-weight:bold;color:#58a6ff" if abs(s["mean"] - best) < 0.001 else ""
            rows += (f'<tr><td style="color:{col}">{diff}</td>'
                     f'<td>{agent}</td>'
                     f'<td style="{bold}">{s["mean"]:.4f}</td>'
                     f'<td>{s["std"]:.4f}</td></tr>')
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Hospital Scheduling Env</title>
<style>body{{font-family:monospace;background:#0d1117;color:#e6edf3;
max-width:900px;margin:40px auto;padding:20px}}
h1{{color:#58a6ff}}h2{{color:#79c0ff;border-bottom:1px solid #21262d;padding-bottom:6px}}
p{{color:#8b949e}}table{{border-collapse:collapse;width:100%;margin-top:12px}}
th{{background:#161b22;color:#8b949e;padding:8px 16px;text-align:left;
border-bottom:1px solid #30363d;font-size:12px}}
td{{padding:8px 16px;border-bottom:1px solid #161b22;font-size:13px}}
pre{{background:#161b22;padding:16px;border-radius:8px;font-size:12px;overflow:auto}}
code{{background:#161b22;padding:2px 6px;border-radius:4px;font-size:12px}}
.badge{{background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb;
border-radius:6px;padding:2px 10px;font-size:12px;margin-left:8px}}</style>
</head><body>
<h1>🏥 Hospital Scheduling Env <span class="badge">Running</span></h1>
<p>OpenEnv-compliant hospital resource scheduling ·
{datetime.now().strftime("%Y-%m-%d %H:%M UTC")}</p>

<h2>Baseline benchmark</h2>
<p>5 seeds × 3 agents × 3 difficulty levels</p>
<table>
<tr><th>Difficulty</th><th>Agent</th><th>Mean score</th><th>Std</th></tr>
{rows}
</table>

<h2>API endpoints</h2>
<pre>POST /reset     {{"seed": 42, "difficulty": "hard"}}
POST /step      {{"patient_id": 1, "assign_bed_type": 0, "assign_or": 0, "assign_ventilator": 1}}
GET  /state     → current observation + state
GET  /validate  → OpenEnv compliance info
GET  /tasks     → task descriptions
GET  /health    → {{"status": "healthy"}}</pre>

<h2>Install as package</h2>
<pre>pip install git+https://huggingface.co/spaces/arpita1204/hospital</pre>

<h2>Tasks</h2>
<table>
<tr><th>Name</th><th>Difficulty</th><th>Episode</th><th>Focus</th></tr>
<tr><td>basic_triage</td><td style="color:#3fb950">easy</td>
  <td>48h</td><td>Correct ICU vs general bed assignment</td></tr>
<tr><td>full_scheduling</td><td style="color:#d29922">medium</td>
  <td>96h</td><td>All resources — OR, ventilators, staff</td></tr>
<tr><td>crisis_surge</td><td style="color:#f85149">hard</td>
  <td>168h</td><td>Surge capacity, scarce ICU + ventilators</td></tr>
</table>
</body></html>"""


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"===== Hospital Scheduling Env — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====",
          flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
