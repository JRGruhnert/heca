from dataclasses import dataclass
from typing import Any

import numpy as np

from heca.data.data import DCEntity, DCScene
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
        ste_id_start = obs[f"heca_{label}_ste"][start][0]
        ste_id_end = obs[f"heca_{label}_ste"][end][0]
        return f"{label}_s{ste_id_start}_s{ste_id_end}"
