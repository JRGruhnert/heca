from typing import Generic, TypeVar

import numpy as np
import torch

from heca.graphs.nodes import GraphNode
from heca.misc.data import DCEntity

T = TypeVar("T", bound=GraphNode)


class NodeSet(Generic[T]):
    def __init__(self, type: str):
        self.items: list[T] = []
        self.index: dict[str, int] = {}
        self.x: torch.Tensor = torch.empty()
        self.type = type

    def add(self, key: str, value: T):
        self.index[key] = len(self.items)
        self.items.append(value)

    def key_update(self, key: str, data: DCEntity):
        idx = self.index[key]
        self.items[idx].data = data
        self.items[idx].changed = True

    def get_by_key(self, key: str) -> T:
        return self.items[self.index[key]]

    def idx_get(self, idx: int) -> T:
        return self.items[idx]

    def get_index(self, key: str) -> int:
        return self.index[key]

    def get_indices(self, keys: list[str]) -> list[int]:
        indices: list[int] = []
        for key in keys:
            indices.append(self.index[key])
        return indices

    def has_key(self, key: str) -> bool:
        return key in self.index

    def build(self):
        x_np = np.stack([node.data.feature for node in self.items], axis=0)
        self.x = torch.from_numpy(x_np).float()
