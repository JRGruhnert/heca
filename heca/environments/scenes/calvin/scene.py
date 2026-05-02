import numpy as np
from dataclasses import dataclass

import torch
from tapas_gmm_modified.env.calvin import Calvin, CalvinConfig

from heca.entities.entity import Entity
from heca.environments.scenes.scene import Scene
from heca.environments.scenes.calvin.area import CalvinAreaConfig
from heca.environments.scenes.calvin import v1, v2
from heca.misc.state import State
from heca.misc.td import TDScene


from heca.converters.calvin_heca import CalvinHecaConverter
from heca.converters.calvin_tapas import CalvinTapasConverter
from heca.converters.converter import HecaConverter, LeafConverter
from heca.properties.property import Property


class CalvinEnvironment(Scene):
    @dataclass(frozen=True, kw_only=True)
    class Query(Scene.Query):
        label: str = "calvin"

    @dataclass(kw_only=True)
    class Config(Scene.Config):
        hoop_cv: HecaConverter.Config = CalvinHecaConverter.Config()
        leaf_cv: LeafConverter.Config = CalvinTapasConverter.Config()

        cc: CalvinConfig = CalvinConfig(
            task="Undefined",
            cameras=("wrist", "front"),
            camera_pose={},
            image_size=(256, 256),
            static=False,
            headless=False,
            scale_action=False,
            delay_gripper=False,
            gripper_plot=False,
            postprocess_actions=False,
            eval_mode=False,
            real_time=False,
            pybullet_vis=False,
        )

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.env = Calvin(self.cfg.cc)
        self.state = State.from_area_config(
            CalvinAreaConfig(),
        )
        self.overrides = set(
            [
                "block_red_position",
                "block_blue_position",
                "block_pink_position",
            ]
        )

    def close(self):
        self.env.close()

    def reset(self):
        obs = self.env.reset()[0]
        x = self.to_observation(obs)
        return self.modify(x)

    def step(self, action: np.ndarray) -> TDScene:
        obs = self.env.step(action, render=False)[0]
        return self.to_observation(obs)

    def modify(self, x: TDScene) -> TDScene:
        v1 = x.data["scene"]["v1"]
        for k in self.overrides:
            v1[k] = torch.cat([v1[k], self.state(v1[k])])
        return x

    def validate(self, x: TDScene) -> bool:
        v1 = x.data["scene"]["v1"]
        for k in self.overrides:
            if torch.all(v1[k][3:] == 0):
                return False
        return True

    def sample(self) -> TDScene:
        x = self.reset()
        while not self.validate(x):
            x = self.reset()
        return x

    def render(self):
        raise NotImplementedError("Render method not implemented yet.")

    def properties(self) -> list[Property.Config]:
        return v1.properties

    def entities(self) -> list[Entity.Config]:
        return v2.entities
