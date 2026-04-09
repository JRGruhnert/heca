from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class NodeParameterConfig:
    pass


class NodeParameter(ABC):
    def __init__(self, config: NodeParameterConfig):
        self.config = config

    @abstractmethod
    def __call__(self, start, goal) -> float:
        "TODO: implement"
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        "TODO: implement"
        raise NotImplementedError()
