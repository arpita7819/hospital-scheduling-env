"""
inference.py — OpenEnv inference script.
Runs a single episode with the heuristic agent and prints the score.
Required by the OpenEnv competition spec.
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
        obs = next_obs

    score = task.grader(env, trajectory)
    return score


if __name__ == "__main__":
    import json
    results = {}
    for diff in ["easy", "medium", "hard"]:
        score = run_inference(difficulty=diff, seed=42)
        results[diff] = round(score, 4)
        print(f"{diff:8s}: {score:.4f}")

    print("\nInference complete.")
    print(json.dumps(results, indent=2))
