from collections import defaultdict

import numpy as np
import torch

from heca.graphs.nodes import GraphNode


class NodeSet:
    def __init__(self, type: str):
        self.items: list[GraphNode] = []
        self.index: dict[str, int] = {}
        self.tags: dict[str, set[int]] = defaultdict(set[int])
        self.x: torch.Tensor = torch.empty()
        self.type = type

    def add(self, key: str, value: GraphNode):
        self.index[key] = len(self.items)
        self.items.append(value)
        self.tags[value.tag].add(len(self.items))

    def update(self, tag: str, x: np.ndarray):
        for idx in self.tags[tag]:
            self.items[idx].data = x
            self.items[idx].changed = True

    def get_by_key(self, key: str) -> GraphNode:
        return self.items[self.index[key]]

    def idx_get(self, idx: int) -> GraphNode:
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
        x_np = np.stack([node.data for node in self.items], axis=0)
        self.x = torch.from_numpy(x_np).float()
