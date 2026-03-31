from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch_geometric.data import Batch


@dataclass
class NodeNetworkerConfig:
    pass


class NodeNetworker(ABC):
    def __init__(self, config: NodeNetworkerConfig):
        self.config = config

    @abstractmethod
    def __call__(self, start, goal) -> Batch:
        "TODO: implement"
        raise NotImplementedError()

    @abstractmethod
    def reset(self, goal):
        "TODO: implement"
        raise NotImplementedError()
