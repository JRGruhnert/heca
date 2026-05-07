import numpy as np
from dataclasses import dataclass

from tapas_gmm_modified.env.calvin import Calvin, CalvinConfig

from heca.entities.entity import Entity
from heca.environment.extractors.extractor import Extractor
from heca.environment.scenes.scene import Scene
from heca.environment.scenes.calvin import v1, v2
from heca.misc.td import TDScene


from heca.environment.extractors.calvin_heca_gt import CalvinHecaConverter
from heca.environment.extractors.calvin_tapas import CalvinTapasConverter
from heca.properties.property import Property


class CalvinScene(Scene):
    @dataclass(frozen=True, kw_only=True)
    class Query(Scene.Query):
        label: str = "calvin"

    @dataclass(kw_only=True)
    class Config(Scene.Config):
        heca_cv: Extractor.Config = CalvinHecaConverter.Config()
        leaf_cv: Extractor.Config = CalvinTapasConverter.Config()

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

    def close(self):
        self.env.close()

    def reset(self) -> tuple[TDScene, bool]:
        obs = self.env.reset()[0]
        return self.to_observation(obs)

    def step(self, action: np.ndarray) -> TDScene:
        obs = self.env.step(action, render=False)[0]
        return self.to_observation(obs)[0]

    def sample(self) -> TDScene:
        x, valid = self.reset()
        while not valid:
            x, valid = self.reset()
        return x

    def render(self):
        raise NotImplementedError("Render method not implemented yet.")

    def properties(self) -> list[Property.Config]:
        return v1.properties

    def entities(self) -> list[Entity.Config]:
        return v2.entities

    def get_image(self) -> dict[str, np.ndarray]:
        obs = self.env._get_obs()
        return self.converters["image"](obs)
