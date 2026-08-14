from dataclasses import dataclass

from heca.data.entity import Entity
from heca.data.free import FreeEntity
from heca.data.prismatic import PrismaticEntity
from heca.data.revolute import RevoluteEntity
from heca.data.static import StaticEntity
from heca.scenes.ogbench.scene import OGScene


class OGScene3(OGScene):
    @dataclass(kw_only=True)
    class Config(OGScene.Config):
        tag: str = "scene3"
        vis: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @property
    def entities(self) -> dict[str, Entity]:
        ents = {
            "button0": StaticEntity.Config(n_states=2),
            "button1": StaticEntity.Config(n_states=2),
            "button2": StaticEntity.Config(n_states=2),
            "shelf0": StaticEntity.Config(),
            "drawer0": PrismaticEntity.Config(),
            "cube0": FreeEntity.Config(),
        }
        return {l: Entity.get(e) for l, e in ents.items()}
