from dataclasses import dataclass

import numpy as np
import torch

from calvin_env_modified.envs.observation import CalvinEnvObservation
from hoopgn.converters.converter import HoopConverter
from hoopgn.misc.td import TDProperties, TDEntities


class CalvinHoopConverter(HoopConverter):
    @dataclass(kw_only=True)
    class Config(HoopConverter.Config):
        label: str = "v1"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, obs: CalvinEnvObservation) -> TDEntities:
        state_dict = {}
        state_dict["ee_position"] = torch.tensor(obs.ee_pose[:3], dtype=torch.float32)
        state_dict["ee_rotation"] = torch.tensor(obs.ee_pose[-4:], dtype=torch.float32)
        state_dict["ee_scalar"] = torch.tensor(
            np.array([obs.ee_state]), dtype=torch.float32
        )

        for label, value in obs.object_poses.items():
            k = label.removeprefix("base__")
            state_dict[f"{k}_position"] = torch.tensor(value[:3], dtype=torch.float32)
            state_dict[f"{k}_rotation"] = torch.tensor(value[-4:], dtype=torch.float32)

        for label, value in obs.object_states.items():
            k = label.removeprefix("base__")
            state_dict[f"{k}_scalar"] = torch.tensor(
                np.array([value]), dtype=torch.float32
            )

        return TDEntities({"v1": TDProperties(state_dict)})
