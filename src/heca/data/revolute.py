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
        pass

    def value_from_gt(self, label: str, obs: dict) -> DCEntity:
        pos = obs[f"heca_{label}_pos"]
        rot = obs[f"heca_{label}_rot"]
        ste = obs[f"heca_{label}_ste"]
        rot = np.array([rot[1], rot[2], rot[3], rot[0]], dtype=np.float32)
        ste = np.atleast_1d(ste)
        value = np.concatenate((pos, rot, ste))
        feature = self.gnn_format(value)
        return DCEntity(value=value, feature=feature)

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
        start_val = obs[f"heca_{label}_angle"][start][0]
        target_val = obs[f"heca_{label}_angle"][end][0]
        direction = "a_b" if target_val > start_val else "b_a"
        return f"{label}_{direction}"
