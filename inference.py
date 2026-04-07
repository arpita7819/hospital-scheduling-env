"""
inference.py — OpenEnv inference script with required structured output.
Prints [START], [STEP], and [END] blocks to stdout as required by the validator.
"""
from environment import HospitalSchedulingEnv
from tasks import load_task
from baseline import HeuristicAgent


def run_inference(difficulty: str = "hard", seed: int = 42) -> float:
    env, task = load_task(difficulty)
    agent = HeuristicAgent()
    obs = env.reset(seed=seed)
    done = False
    trajectory = []
    step_num = 0

    print(f"[START] task={difficulty}", flush=True)

    while not done:
        action = agent.act(obs)
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

 
