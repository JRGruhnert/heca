import abc
from dataclasses import dataclass
from functools import cached_property

from heca.conditions.evaluator import Evaluator
from heca.conditions.pair import ConPair
from heca.misc.base import Persistable
from heca.misc.data import DCScene
from heca.misc.entity import Entity


@dataclass(kw_only=True, slots=True)
class AgentFeedback:
    terminal: bool
    reward: float
    truncated: bool


class Agent(Persistable, abc.ABC):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        n_samples: int = 1000
        folder: str = "agents"
        threshold: float = 0.75
        evaluator: Evaluator.Config = Evaluator.Config()

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.evaluator = Evaluator.get(cfg.evaluator).setup(
            self.conditions,
            self.entities,
            self.elabels,
        )

    @abc.abstractmethod
    def act(self, x: DCScene, y: DCScene) -> tuple[DCScene, AgentFeedback]:
        raise NotImplementedError

    @cached_property
    def elabels(self) -> set[str]:
        raise NotImplementedError

    @cached_property
    def entities(self) -> dict[str, Entity]:
        raise NotImplementedError

    @cached_property
    def conditions(self) -> list[ConPair]:
        raise NotImplementedError
