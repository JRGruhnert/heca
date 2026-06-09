import abc
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from heca.entities.entity import Entity
from heca.entities.precon import Precon
from heca.environment.scenes.scene import Scene
from heca.misc.td import TDEntity, TDScene
from heca.misc.base import Persistable

from torch_geometric.data import HeteroData


class Cursor(Enum):
    IDLE = 1
    ERROR = 2
    ACTIVE = 3


@dataclass(kw_only=True)
class AgentFeedback:
    done: bool
    learn: bool
    state: Cursor
    reward: float
    terminal: bool


class Agent(Persistable, abc.ABC):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abc.abstractmethod
    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(self, x: TDScene, y: TDScene) -> AgentFeedback:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError()

    @abc.abstractmethod
    def gen_pre(self) -> list[tuple[Entity, TDEntity]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def gen_post(self) -> list[tuple[Entity, TDEntity]]:
        raise NotImplementedError()

    @cached_property
    def precons(self) -> list[tuple[Entity, TDEntity]]:
        return self.gen_pre()

    @cached_property
    def postcons(self) -> list[tuple[Entity, TDEntity]]:
        return self.gen_post()

    @abc.abstractmethod
    def required_scenes(self) -> list[Scene.Config]:
        raise NotImplementedError()

    def make_options(self, x: TDScene, y: TDScene, con: Precon) -> list[HeteroData]:
        # take precon
        return [x]
