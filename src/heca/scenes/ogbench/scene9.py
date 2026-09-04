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
        tag: str = "scene9"
        vis: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @property
    def entities(self) -> dict[str, Entity]:
        ents = {
            "button0": StaticEntity.Config(n_states=2),
            "button1": StaticEntity.Config(n_states=3),
            "button2": StaticEntity.Config(n_states=2),
            "button3": StaticEntity.Config(n_states=3),
            "faucet0": RevoluteEntity.Config(),
            "faucet1": RevoluteEntity.Config(),
        }
        return {l: Entity.get(e) for l, e in ents.items()}
