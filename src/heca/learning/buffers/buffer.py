from abc import abstractmethod
from dataclasses import dataclass
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

from collections import defaultdict, deque
from heca.misc.base import Configurable
from heca.misc import logger


@dataclass(slots=True)
class BufferData:
    tag: str
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
        gamma: float = 0.99

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.queue: deque[BufferData] = deque(maxlen=cfg.capacity)
        self.steps_passed = 0
        self.tags: set[str] = set()

    @abstractmethod
    def compute_advantages(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def is_allowed(self, data: BufferData) -> bool:
        raise NotImplementedError

    def add(self, data: BufferData) -> bool:
        if self.is_allowed(data):
            self.queue.append(data)
            self.steps_passed += 1
        return self.full

    @property
    def tag_indices(self) -> dict[str, list[int]]:
        indices_by_tag = defaultdict(list[int])
        for i, item in enumerate(self.queue):
            indices_by_tag[item.tag].append(i)
        return indices_by_tag

    @property
    def full(self) -> bool:
        return self.steps_passed >= self.cfg.capacity

    @property
    def data(self) -> list[HeteroData]:
        return [d.data for d in self.queue]

    @property
    def actions(self) -> torch.Tensor:
        return torch.stack([d.action for d in self.queue])

    @property
    def logprobs(self) -> torch.Tensor:
        return torch.stack([d.logprob for d in self.queue])

    @property
    def values(self) -> torch.Tensor:
        return torch.stack([d.value for d in self.queue])

    @property
    def rewards(self) -> list[float]:
        return [d.reward for d in self.queue]

    @property
    def terminals(self) -> list[bool]:
        return [d.terminal for d in self.queue]

    @property
    def truncates(self) -> list[float]:
        return [d.truncated for d in self.queue]

    def reset(self):
        self.steps_passed = 0

    def save(self, path: Path, label: str):
        logger.info(f"Saving buffer '{label}'")
        serialized = {
            "items": list(self.queue),
        }
        torch.save(serialized, path / f"buffer_{label}.pt")

    def load(self, path: Path, label: str):
        logger.info(f"Loading buffer '{label}'")
        serialized = torch.load(path / f"buffer_{label}.pt")
        self.queue = deque(serialized["items"], maxlen=self.cfg.capacity)
