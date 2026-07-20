from typing import Generic, TypeVar

import numpy as np
import torch

from heca.graphs.node import EntityNode, GraphNode
from heca.misc.data import DCEntity

T = TypeVar("T", bound=GraphNode)


class NodeSet(Generic[T]):
    def __init__(self, type: str):
        self.items: list[T] = []
        self.keys: list[str] = []  # ← add this
        self.index: dict[str, int] = {}
        self.x: torch.Tensor = torch.empty(1)
        self.type = type

    def add(self, key: str, value: T):
        self.index[key] = len(self.items)
        self.items.append(value)
        self.keys.append(key)

    def key_update(self, key: str, data: DCEntity):
        idx = self.index[key]
        self.items[idx].data = data
        self.items[idx].changed = True

    def key_at(self, idx: int) -> str:
        return self.keys[idx]

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
        if self.items and isinstance(self.items[0], EntityNode):
            x_np = np.stack([node.data.feature for node in self.items], axis=0)
        else:
            x_np = np.zeros((len(self.items), 128), dtype=np.float32)
        self.x = torch.from_numpy(x_np).float()

    def __str__(self) -> str:
        lines = [f"NodeSet<{self.type}> ({len(self.items)} nodes):"]
        for i, (key, item) in enumerate(zip(self.keys, self.items)):
            lines.append(f"  [{i}] {key}: {item}")
        return "\n".join(lines)
