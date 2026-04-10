from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch_geometric.data import Batch


@dataclass(kw_only=True)
class SkillNetworkerConfig:
    pass


class SkillNetworker(ABC):
    def __init__(self, config: SkillNetworkerConfig):
        self.config = config

    @abstractmethod
    def __call__(self, start) -> Batch:
        "TODO: implement"
        raise NotImplementedError()

    @abstractmethod
    def reset(self, goal):
        "TODO: implement"
        raise NotImplementedError()
