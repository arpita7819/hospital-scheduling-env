"""
client.py — HTTP client for HospitalSchedulingEnv.
Follows the OpenEnv EnvClient pattern for typed, async interaction.
"""
import httpx
from typing import Optional
from models import HospitalAction, HospitalObservation, HospitalState


class HospitalEnv:
    """
    Typed HTTP client for HospitalSchedulingEnv.

    Usage:
        env = HospitalEnv(base_url="https://arpita1204-hospital.hf.space")
        obs = env.reset(seed=42, difficulty="hard")
        while not done:
            action = HospitalAction(patient_id=1, assign_bed_type=0)
            result = env.step(action)
            obs, reward, done = result["observation"], result["reward"], result["done"]
    """

    def __init__(self, base_url: str = "https://arpita1204-hospital.hf.space"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)

    def reset(self, seed: int = 42, difficulty: str = "hard") -> HospitalObservation:
        resp = self._client.post(
            f"{self.base_url}/reset",
            json={"seed": seed, "difficulty": difficulty},
        )
        resp.raise_for_status()
        data = resp.json()
        return HospitalObservation(**data["observation"])

    def step(self, action: HospitalAction) -> dict:
        resp = self._client.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "observation": HospitalObservation(**data["observation"]),
            "reward":      data["reward"],
            "done":        data["done"],
            "info":        data["info"],
            "score":       data.get("score"),
        }

    def state(self) -> dict:
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def validate(self) -> dict:
        resp = self._client.get(f"{self.base_url}/validate")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
