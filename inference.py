"""
inference.py — LLM-powered inference using competition LiteLLM proxy.
Uses API_BASE_URL and API_KEY environment variables.
Prints [START]/[STEP]/[END] structured output to stdout.
"""
import os
import json
from openai import OpenAI
from tasks import load_task
from models import HospitalAction

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY", "placeholder")
MODEL        = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert hospital resource scheduling agent.
You receive the current hospital state and must decide how to allocate resources.

Respond with ONLY a valid JSON object:
{
  "patient_id": <int, -1 for no-op>,
  "assign_bed_type": <0=ICU, 1=General, 2=None>,
  "assign_or": <0 or 1>,
  "assign_ventilator": <0 or 1>
}

Critical rules:
- CRITICAL patients (acuity=0): always assign ICU (bed_type=0) if icu_beds_free > 0
- HIGH patients (acuity=1): prefer ICU, use general if ICU full
- MEDIUM/LOW patients (acuity=2/3): use general beds (bed_type=1)
- Assign ventilator ONLY if needs_ventilator=true AND ventilators_free > 0
- Prioritise highest-acuity patient with longest wait
- Use -1 if no beds available
"""


def llm_action(obs: dict) -> HospitalAction:
    waiting = obs.get("waiting_patients", [])
    state_summary = {
        "hour": obs["hour"],
        "icu_beds_free": obs["icu_beds_free"],
        "general_beds_free": obs["general_beds_free"],
        "or_rooms_free": obs["or_rooms_free"],
        "ventilators_free": obs["ventilators_free"],
        "overflow_flag": obs["overflow_flag"],
        "waiting_patients": sorted(
            waiting[:6], key=lambda p: (p["acuity"], -p["wait_hours"])
        ),
    }
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(state_summary)},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        return HospitalAction(
            patient_id=int(data.get("patient_id", -1)),
            assign_bed_type=int(data.get("assign_bed_type", 2)),
            assign_or=int(data.get("assign_or", 0)),
            assign_ventilator=int(data.get("assign_ventilator", 0)),
        )
    except Exception as e:
        print(f"[WARN] LLM error: {e} — no-op", flush=True)
        return HospitalAction()


def run_inference(difficulty: str = "hard", seed: int = 42) -> float:
    from server.hospital_environment import HospitalEnvironment
    env = HospitalEnvironment(difficulty=difficulty)
    obs_model = env.reset(seed=seed, difficulty=difficulty)
    obs = obs_model.model_dump()
    done = False
    trajectory = []
    step_num = 0

    print(f"[START] task={difficulty}", flush=True)

    while not done:
        action = llm_action(obs)
        patient_snap = next(
            (p for p in obs.get("waiting_patients", [])
             if p["id"] == action.patient_id), None
        )
        obs_model, reward, done, info = env.step(action)
        obs = obs_model.model_dump()
        step_num += 1
        trajectory.append({
            "obs": obs, "action": action.model_dump(),
            "reward": reward, "patient_snapshot": patient_snap,
            "info": info,
        })
        print(f"[STEP] step={step_num} reward={round(reward, 4)}", flush=True)

    from tasks import load_task as _load
    _, task = _load(difficulty)
    score = task.grader(env._env, trajectory)
    print(f"[END] task={difficulty} score={round(score, 4)} steps={step_num}", flush=True)
    return score


if __name__ == "__main__":
    for diff in ["easy", "medium", "hard"]:
        run_inference(difficulty=diff, seed=42)
