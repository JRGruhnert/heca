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
        # Initialize with zeros
        feat = np.zeros((self.input_feat_dim), dtype=np.float32)
        feat[0:3] = value[0:3]
        feat[3:6] = self.BASE_LOGSTD
        feat[6:10] = Quaternion.normalize(value[3:7])
        feat[10:13] = self.BASE_LOGSTD
        state_ids = value[7].astype(int)  # [N]
        feat[13 : 13 + self.n_states] = -self.LOGIT_CONFIDENCE
        feat[13 + state_ids] = self.LOGIT_CONFIDENCE
        return feat

    def agent_key(self, label: str, obs: dict[str, list], start: int, end: int) -> str:
        pos_key = f"privileged_{label}_pos"
        target_key = f"heca_target_{label}_pos"
        start_val = obs[pos_key][start][0]
        target_val = obs[target_key][end][0]
        direction = "open" if target_val > start_val else "close"
        return f"{label}_{direction}"

        for i in range(len(data)):
            if oracle_done[i] == 1.0:
                # Success of PREVIOUS oracle (last step before switch)
                success = oracle_success[i - 1] == 1.0
                # Direction: compare start vs target
                start = privileged_faucet_0_pos[seg_start]
                target = heca_target_faucet_0_pos[i - 1]
                direction = "open" if target > start else "close"

                if success:
                    save_segment(data[seg_start:i], direction)
                seg_start = i
