"""
inference.py — OpenEnv inference script using LLM agent via LiteLLM proxy.
Uses API_BASE_URL and API_KEY environment variables injected by the competition.
Prints [START]/[STEP]/[END] structured output blocks to stdout.
"""
import os
import json
from openai import OpenAI
from tasks import load_task

# Use competition-injected LiteLLM proxy credentials
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "placeholder")
MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are a hospital resource scheduling agent.
Each step you receive the current hospital state and must decide how to allocate resources to a waiting patient.

You must respond with ONLY a valid JSON object in this exact format:
{
  "patient_id": <int or -1 for no-op>,
  "assign_bed_type": <0=ICU, 1=General, 2=None>,
  "assign_or": <0 or 1>,
  "assign_ventilator": <0 or 1>
}

Rules:
- CRITICAL patients (acuity=0) must go to ICU (assign_bed_type=0) if ICU beds are free
- HIGH patients (acuity=1) prefer ICU, but general is acceptable
- MEDIUM/LOW patients (acuity=2/3) go to general beds (assign_bed_type=1)
- Only assign ventilator if patient needs_ventilator=true and ventilators_free > 0
- If no beds are available, use patient_id=-1 (no-op)
- Always prioritise the most critical waiting patient first
"""


def llm_action(obs: dict) -> dict:
    """Ask the LLM to pick an action given the current observation."""
    waiting = obs.get("waiting_patients", [])

    state_summary = {
        "icu_beds_free": obs["icu_beds_free"],
        "general_beds_free": obs["general_beds_free"],
        "or_rooms_free": obs["or_rooms_free"],
        "ventilators_free": obs["ventilators_free"],
        "nurses_free": obs["nurses_free"],
        "doctors_free": obs["doctors_free"],
        "overflow_flag": obs["overflow_flag"],
        "waiting_patients": waiting[:5],  # top 5 to stay within context
    }

    user_msg = f"Current hospital state:\n{json.dumps(state_summary, indent=2)}\n\nChoose your action:"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action = json.loads(raw.strip())
        # Validate required keys
        return {
            "patient_id":        int(action.get("patient_id", -1)),
            "assign_bed_type":   int(action.get("assign_bed_type", 2)),
            "assign_or":         int(action.get("assign_or", 0)),
            "assign_ventilator": int(action.get("assign_ventilator", 0)),
        }
    except Exception as e:
        print(f"[WARN] LLM parse error: {e} — using no-op", flush=True)
        return {"patient_id": -1, "assign_bed_type": 2, "assign_or": 0, "assign_ventilator": 0}


def run_inference(difficulty: str = "hard", seed: int = 42) -> float:
    env, task = load_task(difficulty)
    obs = env.reset(seed=seed)
    done = False
    trajectory = []
    step_num = 0

    print(f"[START] task={difficulty}", flush=True)

    while not done:
        action = llm_action(obs)
        patient_snap = next(
            (p for p in obs.get("waiting_patients", [])
             if p["id"] == action.get("patient_id")), None
        )
        next_obs, reward, done, info = env.step(action)
        trajectory.append({
            "obs": obs, "action": action,
            "reward": reward, "patient_snapshot": patient_snap,
            "info": info,
        })
        step_num += 1
        print(f"[STEP] step={step_num} reward={round(reward, 4)}", flush=True)
        obs = next_obs

    score = task.grader(env, trajectory)
    print(f"[END] task={difficulty} score={round(score, 4)} steps={step_num}", flush=True)
    return score


if __name__ == "__main__":
    for diff in ["easy", "medium", "hard"]:
        run_inference(difficulty=diff, seed=42)
