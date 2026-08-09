from dataclasses import dataclass

import numpy as np

from heca.data.data import DCEntity
from heca.data.entity import Entity
from heca.utils.quaternion import Quaternion


class RevoluteEntity(Entity):
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
                "n_columns": 9,
            },
            "state": {
                "model": "categorical",
                "n_columns": 1,
            },
        }

    def extra_part(self, label: str, obs: dict) -> np.ndarray:
        raise NotImplementedError

        relative = (current - min_angle) / (max_angle - min_angle)  # Already [0,1]
        range_size = max_angle - min_angle  # Scale info
        midpoint = (max_angle + min_angle) / 2  # Offset info

        # Normalize range_size and midpoint using pre-computed dataset statistics (Z-score)
        range_norm = (range_size - range_mean) / range_std
        midpoint_norm = (midpoint - midpoint_mean) / midpoint_std

        network_input = [relative, range_norm, midpoint_norm]

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

    def make_agent_key(
        self, label: str, obs: dict[str, list], start: int, end: int
    ) -> str:
        start_val = obs[f"heca_{label}_ang"][start][0]
        target_val = obs[f"heca_{label}_ang"][end][0]
        direction = "a_b" if target_val > start_val else "b_a"
        return f"{label}_{direction}"
