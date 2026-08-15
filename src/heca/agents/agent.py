import abc
from dataclasses import dataclass
from functools import cached_property

from heca.misc.base import Persistable
from heca.data.data import DCScene
from heca.data.entity import Entity
from heca.scenes.scene import Scene, SceneFeedback


class Agent(Persistable, abc.ABC):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        scene: Scene.Config
        folder: str = "agents"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @cached_property
    def entities(self) -> dict[str, Entity]:
        raise NotImplementedError
