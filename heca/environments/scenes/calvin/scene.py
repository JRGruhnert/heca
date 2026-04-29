import numpy as np
from dataclasses import dataclass, field

import torch
from tapas_gmm_modified.env.calvin import Calvin, CalvinConfig

from heca.environments.scenes.scene import Scene
from heca.environments.scenes.calvin.area import CalvinAreaConfig
from heca.misc.state import State
from heca.misc.td import TDScene


from heca.converters.calvin_heca import CalvinHecaConverter
from heca.converters.calvin_tapas import CalvinTapasConverter
from heca.converters.converter import HecaConverter, LeafConverter
from heca.properties.default.v1.area import AreaProperty
from heca.properties.default.v1.bool import BoolProperty
from heca.properties.default.v1.flip import FlipProperty
from heca.properties.default.v1.position import PositionProperty
from heca.properties.default.v1.range import RangeProperty
from heca.properties.default.v1.rotation import RotationProperty
from heca.properties.evaluators.default import DefaultEvaluator

ee_position = PositionProperty.Config(label="ee_position")
ee_rotation = RotationProperty.Config(label="ee_rotation")
ee_scalar = BoolProperty.Config(label="ee_scalar")
slide_position = PositionProperty.Config(label="slide_position")
slide_rotation = RotationProperty.Config(label="slide_rotation")
slide_scalar = RangeProperty.Config(label="slide_scalar", low=0.0, high=0.28)
drawer_position = PositionProperty.Config(label="drawer_position")
drawer_rotation = RotationProperty.Config(label="drawer_rotation")
drawer_scalar = RangeProperty.Config(label="drawer_scalar", low=0.0, high=0.22)
button_position = PositionProperty.Config(label="button_position")
button_rotation = RotationProperty.Config(label="button_rotation")
button_scalar = FlipProperty.Config(label="button_scalar")
led_position = PositionProperty.Config(label="led_position")
led_rotation = RotationProperty.Config(label="led_rotation")
block_red_position = AreaProperty.Config(label="block_red_position")
block_red_scalar = BoolProperty.Config(label="block_red_scalar")
block_blue_position = AreaProperty.Config(label="block_blue_position")
block_blue_scalar = BoolProperty.Config(label="block_blue_scalar")
block_pink_position = AreaProperty.Config(label="block_pink_position")
block_pink_scalar = BoolProperty.Config(label="block_pink_scalar")
block_red_rotation = RotationProperty.Config(
    label="block_red_rotation",
    evaluator=DefaultEvaluator.Config(),
)
block_blue_rotation = RotationProperty.Config(
    label="block_blue_rotation",
    evaluator=DefaultEvaluator.Config(),
)
block_pink_rotation = RotationProperty.Config(
    label="block_pink_rotation",
    evaluator=DefaultEvaluator.Config(),
)


base = [
    ee_position,
    ee_rotation,
    ee_scalar,
    drawer_position,
    drawer_rotation,
    drawer_scalar,
    button_position,
    button_rotation,
    button_scalar,
    led_position,
    led_rotation,
]
slide_base = [slide_position, slide_rotation, slide_scalar]
red_base = [block_red_position, block_red_rotation, block_red_scalar]
pink_base = [block_pink_position, block_pink_rotation, block_pink_scalar]
blue_base = [block_blue_position, block_blue_rotation, block_blue_scalar]


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
