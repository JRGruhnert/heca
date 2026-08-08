import numpy as np
from dataclasses import dataclass
import torch

from heca.misc.base import Configurable
from heca.data.data import DCEntity


class Entity(Configurable):
    input_feat_dim: int = 64

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        states: list[str] = []
        question: str = ""
        answers: list[str] = []

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def n_states(self) -> int:
        return len(self.cfg.states)

    def make_zeros(self) -> torch.Tensor:
        return torch.zeros(len(self.cfg.states), dtype=torch.float32)

    def make_idx(self, label: str) -> int:
        assert label is not None, "Label cannot be None."
        assert label in self.cfg.states, "Label must be in state values."
        return self.cfg.states.index(label)

    def one_hot_from_idx(self, idx: int) -> torch.Tensor:
        idx = int(idx)
        assert 0 <= idx < len(self.cfg.states), "Index out of bounds."
        one_hot = torch.zeros(len(self.cfg.states), dtype=torch.float32)
        one_hot[idx] = 1.0
        return one_hot

    @classmethod
    def one_hot_from_idx_dc(cls, idx: int, n_states: int) -> np.ndarray:
        assert 0 <= idx < n_states, "Index out of bounds."
        one_hot = np.zeros(n_states)
        one_hot[idx] = 1.0
        return one_hot

    def value_from_gt(self, obs: dict) -> DCEntity:
        raise NotImplementedError

    def value_from_image(self, obs: dict) -> DCEntity:
        raise NotImplementedError

    def gnn_format(self, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def make_agent_key(
        self, label: str, obs: dict[str, list], start: int, end: int
    ) -> str:
        raise NotImplementedError
