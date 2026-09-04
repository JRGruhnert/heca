from dataclasses import dataclass

from heca.data.entity import Entity
from heca.data.free import FreeEntity
from heca.data.prismatic import PrismaticEntity
from heca.data.revolute import RevoluteEntity
from heca.data.static import StaticEntity
from heca.scenes.ogbench.scene import OGScene


class OGScene6(OGScene):
    @dataclass(kw_only=True)
    class Config(OGScene.Config):
        tag: str = "scene6"
        vis: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @property
    def entities(self) -> dict[str, Entity]:
        ents = {
            "box0": StaticEntity.Config(n_states=2),
            "faucet0": RevoluteEntity.Config(),
            "button0": StaticEntity.Config(n_states=3),
            "button1": StaticEntity.Config(n_states=2),
            "lid0": FreeEntity.Config(),
            "slider0": PrismaticEntity.Config(),
        }
        return {l: Entity.get(e) for l, e in ents.items()}
