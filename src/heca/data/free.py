from dataclasses import dataclass
from typing import Any

import numpy as np

from heca.data.data import DCScene
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
                "n_columns": 3 + self.rot_dim,
            },
            "state": {
                "model": "categorical",
                "n_columns": 1,
            },
        }

    def extra_part(self, label: str, obs: dict) -> np.ndarray:
        return np.zeros(0)

    def env_state_value(self, label: str, x: DCScene) -> dict[str, Any]:
        dc = x.get(label)

        pos = self.unnormalize_position(
            dc.pos, x.extras["meta_xyz_center"], x.extras["meta_xyz_scaler"]
        )
        return {
            f"heca_{label}_pos": pos,
            f"heca_{label}_rot": dc.rot,
            f"heca_{label}_ste": dc.ste,
        }

    def make_agent_key(self, label: str, obs: Any, start: int, end: int) -> str:
        start_val = _clean(obs[f"heca_{label}_loc"][start])
        target_val = _clean(obs[f"heca_{label}_loc"][end])
        return f"{label}_{start_val}_{target_val}"
