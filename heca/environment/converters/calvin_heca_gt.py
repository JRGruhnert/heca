from dataclasses import dataclass

import numpy as np
import torch

from calvin_env_modified.envs.observation import CalvinEnvObservation
from heca.environment.converters.converter import ObsConverter
from heca.environment.scenes.calvin.area import CalvinAreaConfig
from heca.misc.state import State
from heca.misc.td import TDProperties, TDEntities


class CalvinHecaConverter(ObsConverter):
    @dataclass(kw_only=True)
    class Config(ObsConverter.Config):
        label: str = "v1"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.overrides = set(
            [
                "block_red_position",
                "block_blue_position",
                "block_pink_position",
            ]
        )

        self.state = State.from_area_config(
            CalvinAreaConfig(),
        )

    def __call__(self, obs: CalvinEnvObservation) -> tuple[TDEntities, bool]:
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

        valid = True
        for k in self.overrides:
            state_dict[k] = torch.cat([state_dict[k], self.state(state_dict[k])])
            if torch.all(state_dict[k][3:] == 0):
                valid = False
        return TDEntities({"v1": TDProperties(state_dict)}), valid
