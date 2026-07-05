from dataclasses import dataclass, field
from typing import TypeVar

import torch
from torch_geometric.data import HeteroData

from heca.agents.agent import Agent
from heca.graphs.nodes import (
    CompNode,
    EntityNode,
    StepMixNode,
    GraphNode,
    PosCompNode,
    PosNode,
    RotCompNode,
    RotNode,
    StateCompNode,
    StateNode,
)
from heca.graphs.sets import EdgeSet, NodeSet
from heca.misc.td import TDScene

T = TypeVar("T", bound=GraphNode)


@dataclass
class GraphBlueprint:
    nodes: dict[str, NodeSet] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], EdgeSet] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "GraphBlueprint":
        return cls()

    def flush(self) -> "GraphBlueprint":
        self.nodes = {}
        self.edges = {}
        return self

    def size(self) -> dict[str, dict[str, int]]:
        return {
            "nodes": {n: s.size() for n, s in self.nodes.items()},
            "edges": {f"{e[0]}_{e[1]}_{e[2]}": s.size() for e, s in self.edges.items()},
        }

    def graph(self) -> HeteroData:
        data = HeteroData()
        for label, ns in self.nodes.items():
            num = len(ns.nodes)
            if ns.static_x is not None:
                data[label].x = ns.static_x
            else:
                data[label].x = torch.zeros(num, 1)  # fallback
            data[label].num_nodes = num

        for (src, rel, dst), es in self.edges.items():
            data[(src, rel, dst)].edge_index = es.edge_index
            if es.edge_attr is not None:
                data[(src, rel, dst)].edge_attr = es.edge_attr
        return data

    def options(self) -> list[tuple[Agent.Config, TDScene, TDScene]]:
        raise NotImplementedError

    def set_node(self, node: GraphNode, label: str):
        if label not in self.nodes:
            self.nodes[label] = NodeSet(nodes=[])
        self.nodes[label].set(node)

    def set_entity(self, tdscene: TDScene, tag: str):
        for entity_name, tde in tdscene.entities():
            self.set_node(PosNode(entity=entity_name, x=tde.position), label="pose")
            self.set_node(RotNode(entity=entity_name, x=tde.rotation), label="rot")
            self.set_node(StateNode(entity=entity_name, x=tde.state), label="state")
            self.set_node(EntityNode(entity=entity_name, tag=tag), label="entity")
        self.nodes["pos"].build()
        self.nodes["rot"].build()
        self.nodes["state"].build()
        self.nodes["entity"].build()

    def set_gmm(self, tdscene: TDScene, tag: str):
        for entity_name, tde in tdscene.entities():
            self.set_node(PosCompNode(entity=entity_name, x=tde.position), label="pose")
            self.set_node(RotCompNode(entity=entity_name, x=tde.rotation), label="rot")
            self.set_node(StateCompNode(entity=entity_name, x=tde.state), label="state")
            self.set_node(CompNode(entity=entity_name, tag=tag), label="entity")
            self.set_node(StepMixNode(entity=entity_name, tag=tag), label="entity")
        self.nodes["pos"].build()
        self.nodes["rot"].build()
        self.nodes["state"].build()
        self.nodes["entity"].build()

    def build(self, sets: list[str] | None = None):
        pass
