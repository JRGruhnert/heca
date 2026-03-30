from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.observation.observation import StateValueDict
from src.skills.tree.leafs.leaf import Leaf


@dataclass
class EnvironmentConfig:
    render: bool = False


class Environment(ABC):
    @abstractmethod
    def sample_task(self) -> tuple[StateValueDict, StateValueDict]:
        raise NotImplementedError("Reset method not implemented yet.")

    @abstractmethod
    def step(self, action) -> tuple[StateValueDict, float, bool]:
        raise NotImplementedError("Step method not implemented yet.")

    @abstractmethod
    def close(self):
        raise NotImplementedError("Close method not implemented yet.")
