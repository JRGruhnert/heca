from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.observation.observation import StateValueDict
from torch_geometric.data import Batch, HeteroData


@dataclass
class BranchNetworkerConfig:
    pass


class BranchNetworker(ABC):
    def __init__(self, config: BranchNetworkerConfig):
        self.config = config

    def reset(self, goal: StateValueDict):
        """Prepare the networker for execution. Before each use."""
        raise NotImplementedError("Subclasses must implement method.")

    @abstractmethod
    def __call__(self, start: StateValueDict, goal: StateValueDict) -> Batch:
        """Predict the next action given the current observation."""
        raise NotImplementedError("Subclasses must implement method.")
