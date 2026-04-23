from dataclasses import dataclass

import numpy as np

from hoopgn.observation.td_scene import TDScene
from hoopgn.policies.policy import Policy


class LeafPolicy(Policy):
    @dataclass(kw_only=True)
    class Config(Policy.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(self, x: TDScene, y: TDScene) -> np.ndarray | None:
        raise NotImplementedError()
