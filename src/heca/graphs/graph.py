from dataclasses import dataclass, field
from typing import TypeVar

import torch
from torch_geometric.data import HeteroData

from heca.agents.agent import Agent
from heca.graphs.nodes import (
    CompNode,
    ConditionNode,
    EntityNode,
    StepMixNode,
    GraphNode,
    PosCompNode,
    PosNode,
    RotCompNode,
    RotNode,
    SteCompNode,
    SteNode,
)
from heca.graphs.sets import EdgeSet, NodeSet
from heca.misc.td import TDScene
from heca.conditions.condition import Condition

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
        for entity, tde in tdscene.entities():
            self.set_node(PosNode(entity=entity, x=tde.pos), label="pos")
            self.set_node(RotNode(entity=entity, x=tde.rot), label="rot")
            self.set_node(SteNode(entity=entity, x=tde.ste), label="ste")
            self.set_node(EntityNode(entity=entity, tag=tag), label="entity")
        self.nodes["pos"].build()
        self.nodes["rot"].build()
        self.nodes["ste"].build()
        self.nodes["entity"].build()

    def set_condition(self, con: Condition, tag: str):
        contag = con.label
        for entity, components in con.model_parameters.items():
            etag = contag + entity
            for idx, ct in enumerate(components):
                ctag = f"{etag}_{idx}"
                self.set_node(
                    PosCompNode(tag=ctag, x=torch.Tensor(ct[0])),
                    label="posc",
                )
                self.set_node(
                    RotCompNode(tag=ctag, x=torch.Tensor(ct[1])),
                    label="rotc",
                )
                self.set_node(
                    SteCompNode(tag=ctag, x=torch.Tensor(ct[2])),
                    label="stec",
                )
                self.set_node(
                    CompNode(tag=ctag, entity=entity),
                    label="comp",
                )
            self.set_node(StepMixNode(tag=ctag), label="stepmix")
        self.set_node(ConditionNode(condition=tag), label=tag)
        self.nodes["pos"].build()
        self.nodes["rot"].build()
        self.nodes["state"].build()
        self.nodes["entity"].build()

    def set_option(self, tdscene: TDScene, tag: str):
        for entity_name, tde in tdscene.entities():
            self.set_node(PosCompNode(entity=entity_name, x=tde.pos), label="pose")
            self.set_node(RotCompNode(entity=entity_name, x=tde.rot), label="rot")
            self.set_node(SteCompNode(entity=entity_name, x=tde.ste), label="state")
            self.set_node(CompNode(entity=entity_name, tag=tag), label="entity")
            self.set_node(StepMixNode(tag=entity_name, tag=tag), label="entity")
        self.nodes["pos"].build()
        self.nodes["rot"].build()
        self.nodes["state"].build()
        self.nodes["entity"].build()

    def build(self, sets: list[str] | None = None):
        pass
