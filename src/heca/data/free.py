from dataclasses import dataclass
from typing import Any

import numpy as np

from heca.data.entity import Entity


def _clean(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


class FreeEntity(Entity):
    TYPE_ID = 0

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        threshold: float = 0.4

    @property
    def measurement(self) -> dict:
        return {
            "pose": {
                "model": "gaussian_diag",
                "n_columns": 6,
            },
            "state": {
                "model": "categorical",
                "n_columns": 1,
            },
        }

    def extra_part(self, label: str, obs: dict) -> np.ndarray:
        return np.zeros(0)

    def make_agent_key(self, label: str, obs: Any, start: int, end: int) -> str:
        start_val = _clean(obs[f"heca_{label}_loc"][start])
        target_val = _clean(obs[f"heca_{label}_loc"][end])
        return f"{label}_{start_val}_{target_val}"
