from dataclasses import dataclass

import numpy as np

from heca.data.data import DCEntity
from heca.data.entity import Entity
from heca.utils.quaternion import Quaternion


class FreeEntity(Entity):
    BASE_LOGSTD = -10.0
    LOGIT_CONFIDENCE = 10.0

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        threshold: float = 0.4

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
        return f"{label}_move"
