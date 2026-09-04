from dataclasses import dataclass

from heca.data.entity import Entity
from heca.data.free import FreeEntity
from heca.data.prismatic import PrismaticEntity
from heca.data.revolute import RevoluteEntity
from heca.data.static import StaticEntity
from heca.scenes.ogbench.scene import OGScene


class OGScene7(OGScene):
    @dataclass(kw_only=True)
    class Config(OGScene.Config):
        tag: str = "scene8"
        vis: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @property
    def entities(self) -> dict[str, Entity]:
        ents = {
            "slider0": PrismaticEntity.Config(),
            "slider1": PrismaticEntity.Config(),
            "drawer0": StaticEntity.Config(),
            "cube0": FreeEntity.Config(),
            "button0": StaticEntity.Config(n_states=3),
        }
        return {l: Entity.get(e) for l, e in ents.items()}
