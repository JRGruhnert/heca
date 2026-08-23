from dataclasses import dataclass
from typing import Any

import numpy as np

from heca.data.data import DCEntity, DCScene
from heca.data.entity import Entity


class RevoluteEntity(Entity):
    TYPE_ID = 3

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        threshold: float = 0.6

    @property
    def measurement(self) -> dict:
        return {
            "pose": {
                "model": "gaussian_diag",
                "n_columns": 3 + self.rot_dim + 2,
            },
            "state": {
                "model": "categorical",
                "n_columns": 1,
            },
        }

    def extra_part(self, label: str, obs: dict) -> np.ndarray:
        ang = obs[f"heca_{label}_ang"]
        return np.array([np.sin(ang), np.cos(ang)])

        relative = (current - min_angle) / (max_angle - min_angle)  # Already [0,1]
        range_size = max_angle - min_angle  # Scale info
        midpoint = (max_angle + min_angle) / 2  # Offset info

        # Normalize range_size and midpoint using pre-computed dataset statistics (Z-score)
        range_norm = (range_size - range_mean) / range_std
        midpoint_norm = (midpoint - midpoint_mean) / midpoint_std

        network_input = [relative, range_norm, midpoint_norm]

    def env_state_value(self, label: str, x: DCScene) -> dict[str, Any]:
        dc = x.get(label)
        pos = self.unnormalize_position(
            dc.pos, x.extras["meta_xyz_center"], x.extras["meta_xyz_scaler"]
        )
        return {
            f"heca_{label}_pos": pos,
            f"heca_{label}_rot": dc.rot,
            f"heca_{label}_ste": dc.ste,
            # Inverse of extra_part: extra = [sin(ang), cos(ang)].
            f"heca_{label}_ang": float(np.arctan2(dc.ext[0], dc.ext[1])),
        }

    def make_agent_key(self, label: str, obs: Any, start: int, end: int) -> str:
        start_val = obs[f"heca_{label}_ang"][start][0]
        target_val = obs[f"heca_{label}_ang"][end][0]
        direction = "a_b" if target_val > start_val else "b_a"
        return f"{label}_{direction}"
