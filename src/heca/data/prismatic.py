from dataclasses import dataclass

import numpy as np

from heca.data.data import DCEntity
from heca.data.entity import Entity
from heca.utils.quaternion import Quaternion


class PrismaticEntity(Entity):
    BASE_LOGSTD = -10.0
    LOGIT_CONFIDENCE = 10.0

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        threshold: float = 0.6

    @property
    def measurement(self) -> dict:
        return {
            "pose": {
                "model": "gaussian_diag",
                "n_columns": 10,
            },
            "state": {
                "model": "categorical",
                "n_columns": 1,
            },
        }

    def gnn_format(self, value: np.ndarray):
        feat = np.zeros((self.input_feat_dim), dtype=np.float32)
        feat[0:3] = value[0:3]
        feat[3:6] = self.BASE_LOGSTD
        feat[6:10] = Quaternion.normalize(value[3:7])
        feat[10:13] = self.BASE_LOGSTD
        state_ids = value[7].astype(int)  # [N]
        feat[13 : 13 + self.n_states] = -self.LOGIT_CONFIDENCE
        feat[13 + state_ids] = self.LOGIT_CONFIDENCE
        return feat

        # Inputs: current_pos, min_pos, max_pos (all in same unit, e.g., meters)

    def extra_part(self, label: str, obs: dict) -> np.ndarray:
        raise NotImplementedError

        relative = (current_pos - min_pos) / (max_pos - min_pos)  # Always [0, 1]
        range_raw = max_pos - min_pos  # Absolute stroke
        midpoint_raw = (max_pos + min_pos) / 2  # Absolute center

        # Optional: If range spans multiple orders of magnitude, use log transform
        # range_transformed = np.log(range_raw + 1e-6)

        # Normalize using pre-computed dataset statistics (mean and std)
        range_norm = (range_raw - range_mean) / range_std
        midpoint_norm = (midpoint_raw - midpoint_mean) / midpoint_std

        # Feed these 3 numbers into your network
        network_input = [relative, range_norm, midpoint_norm]

    def make_agent_key(
        self, label: str, obs: dict[str, list], start: int, end: int
    ) -> str:
        start_val = obs[f"heca_{label}_displacement"][start][0]
        target_val = obs[f"heca_{label}_displacement"][end][0]
        direction = "a_b" if target_val > start_val else "b_a"
        return f"{label}_{direction}"
