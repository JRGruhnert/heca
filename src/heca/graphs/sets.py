from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

import torch

from heca.graphs.nodes import GraphNode

T = TypeVar("T", bound=GraphNode)


@dataclass
class NodeSet(Generic[T]):
    nodes: List[T]
    static_x: Optional[torch.Tensor] = None

    def build(self):
        if not self.nodes:
            self.static_x = None
            return
        # Use the first node's x shape to infer feature dim
        sample = self.nodes[0].x
        dim = sample.shape[0]
        feats = []
        for node in self.nodes:
            if node.x is not None:
                feats.append(node.x)
            else:
                feats.append(torch.zeros(dim))
        self.static_x = torch.stack(feats)

    def set(self, node: T):
        # Runtime type check: if we already have nodes, the new one must match
        if self.nodes and not isinstance(node, type(self.nodes[0])):
            raise TypeError(
                f"Expected node of type {type(self.nodes[0]).__name__}, "
                f"got {type(node).__name__}"
            )
        # Insert/replace by key
        for i, existing in enumerate(self.nodes):
            if existing.key == node.key:
                self.nodes[i] = node
                return
        self.nodes.append(node)

    def size(self) -> int:
        return len(self.nodes)


@dataclass
class EdgeSet:
    edge_index: torch.Tensor  # (2, num_edges)
    edge_attr: Optional[torch.Tensor] = None

    @classmethod
    def build(
        cls,
        src: NodeSet[T],
        dst: NodeSet[T],
    ) -> "EdgeSet":
        src_map: dict[str, list] = {}
        for i, node in enumerate(src.nodes):
            src_map.setdefault(node.dst, []).append(i)
        dst_map: dict[str, list] = {}
        for j, node in enumerate(dst.nodes):
            dst_map.setdefault(node.src, []).append(j)

        src_idx, dst_idx = [], []
        for key in set(src_map.keys()) & set(dst_map.keys()):
            for i in src_map[key]:
                for j in dst_map[key]:
                    src_idx.append(i)
                    dst_idx.append(j)

        if not src_idx:
            ei = torch.empty((2, 0), dtype=torch.long)
        else:
            ei = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        return cls(edge_index=ei)

    def size(self) -> int:
        return self.edge_index.shape[1]
