from abc import abstractmethod
from dataclasses import dataclass
import torch
from torch_geometric.data import HeteroData

from heca.misc.base import Configurable


@dataclass(slots=True)
class BufferData:
    data: HeteroData
    action: torch.Tensor
    logprob: torch.Tensor
    value: torch.Tensor
    reward: float = 0.0
    terminal: bool = False
    truncated: bool = False


class Buffer(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        capacity: int
        gae_lambda: float = 0.95
        gamma: float = 0.99

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.highscore = float("-inf")

    def store_prediction(
        self,
        data: HeteroData,
        action: torch.Tensor,
        logprob: torch.Tensor,
        value: torch.Tensor,
        tag: str,
    ):
        raise NotImplementedError

    def store_feedback(
        self,
        reward: float,
        terminal: bool,
        truncated: bool,
        tag: str,
    ) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def full(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def flat_data(self) -> list[HeteroData]:
        raise NotImplementedError

    @property
    @abstractmethod
    def flat_actions(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def flat_logprobs(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def flat_values(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_advantages(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
