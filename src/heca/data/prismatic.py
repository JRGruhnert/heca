from dataclasses import dataclass
from typing import Any

import numpy as np

from heca.data.entity import Entity


class PrismaticEntity(Entity):
    TYPE_ID = 2

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        threshold: float = 0.6

    @property
    def measurement(self) -> dict:
        return {
            "pose": {
                "model": "gaussian_diag",
                "n_columns": 7,
            },
            "state": {
                "model": "categorical",
                "n_columns": 1,
            },
        }

    def extra_part(self, label: str, obs: dict) -> np.ndarray:
        min_pos = obs[f"heca_{label}_sca_min"]
        max_pos = obs[f"heca_{label}_sca_max"]
        current_pos = obs[f"heca_{label}_sca"]
        relative = (current_pos - min_pos) / (max_pos - min_pos)
        relative = 2 * relative - 1
        return np.array([relative])

    def make_agent_key(self, label: str, obs: Any, start: int, end: int) -> str:
        start_val = obs[f"heca_{label}_sca"][start][0]
        target_val = obs[f"heca_{label}_sca"][end][0]
        direction = "a_b" if target_val > start_val else "b_a"
        return f"{label}_{direction}"
