from dataclasses import dataclass

from heca.data.entity import Entity
from heca.data.free import FreeEntity
from heca.data.prismatic import PrismaticEntity
from heca.data.revolute import RevoluteEntity
from heca.data.static import StaticEntity
from heca.scenes.ogbench.scene import OGScene


class OGScene2(OGScene):
    @dataclass(kw_only=True)
    class Config(OGScene.Config):
        tag: str = "scene2"
        vis: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @property
    def entities(self) -> dict[str, Entity]:
        ents = {
            "box0": StaticEntity.Config(
                states=["open", "closed"],  # 0, 1
            ),
            "faucet0": RevoluteEntity.Config(
                states=["default"],  # 0, 1
            ),
            "button0": StaticEntity.Config(
                states=["free", "locked"],  # 0, 1
            ),
            "button1": StaticEntity.Config(
                states=["free", "locked"],  # 0, 1
            ),
            "lid0": FreeEntity.Config(
                states=["default"],  # 0, 1
            ),
            "peg0": FreeEntity.Config(
                states=["default"],  # 0, 1
            ),
        }
        return {l: Entity.get(e) for l, e in ents.items()}
