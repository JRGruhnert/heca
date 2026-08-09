from dataclasses import dataclass
from typing import Any

import numpy as np

from heca.data.entity import Entity


class StaticEntity(Entity):
    TYPE_ID = 1

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        threshold: float = 0.8

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
        raise NotImplementedError

    def make_agent_key(
        self, label: str, obs: Any, start: int, end: int
    ) -> str:
        ste_id_start = obs[f"heca_{label}_ste"][start][0]
        ste_id_end = obs[f"heca_{label}_ste"][end][0]
        ste_label_start = self.cfg.states[ste_id_start]
        ste_label_end = self.cfg.states[ste_id_end]
        return f"{label}_{ste_label_start}_{ste_label_end}"
