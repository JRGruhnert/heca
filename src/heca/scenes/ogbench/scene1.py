from dataclasses import dataclass

import numpy as np

from heca.data.data import DCEntity, DCScene
from heca.data.entity import Entity
from heca.scenes.ogbench.scene import OGScene


class OGScene1(OGScene):
    @dataclass(kw_only=True)
    class Config(OGScene.Config):
        tag: str = "scene1"
        vis: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @property
    def entities(self) -> set[Entity]:
        ents = []
        return set([Entity.get(e) for e in ents])
