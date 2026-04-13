from abc import ABC, abstractmethod
from dataclasses import dataclass

from hoopgn.observation.td_parameters import TDParameters
from torch_geometric.data import Batch, HeteroData


@dataclass(kw_only=True)
class BranchNetworkerConfig:
    pass


class BranchNetworker(ABC):
    def __init__(self, config: BranchNetworkerConfig):
        self.config = config

    def reset(self, goal: TDParameters):
        """Prepare the networker for execution. Before each use."""
        raise NotImplementedError("Subclasses must implement method.")

    @abstractmethod
    def __call__(self, start: TDParameters, goal: TDParameters) -> Batch:
        """Predict the next action given the current observation."""
        raise NotImplementedError("Subclasses must implement method.")
