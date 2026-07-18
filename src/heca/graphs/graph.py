from collections import defaultdict
from dataclasses import field
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from heca.agents.agent import Agent
from heca.graphs.nodes import *
from heca.graphs.node_set import NodeSet
from heca.graphs.edge_set import EdgeSet
from heca.misc import hardware, logger
from heca.misc.data import DCScene
from heca.misc.entity import Entity
from heca.conditions.condition import Condition


@dataclass
class Graph:
    entities: dict[str, Entity]
    ns_entity: NodeSet[EntityNode] = NodeSet[EntityNode]("entity")
    ns_option: NodeSet[OptionNode] = NodeSet[OptionNode]("option")
    es_summary: EdgeSet[EntityNode, OptionNode] = EdgeSet[EntityNode, OptionNode](
        ("entity", "summary", "option")
    )
    es_stepmix: EdgeSet[EntityNode, EntityNode] = EdgeSet[EntityNode, EntityNode](
        ("entity", "stepmix", "entity")
    )
    es_tapas: EdgeSet[EntityNode, EntityNode] = EdgeSet[EntityNode, EntityNode](
        ("entity", "tapas", "entity")
    )
    packages: dict[str, tuple[Agent.Config, DCScene, DCScene]] = field(
        default_factory=dict
    )
    start_keys: set[str] = field(default_factory=set)
    goal_keys: set[str] = field(default_factory=set)
    start: DCScene = DCScene.empty()
    goal: DCScene = DCScene.empty()

    def export(self) -> HeteroData:
        data = HeteroData()
        data[self.ns_entity.type].x = self.ns_entity.x
        data[self.ns_option.type].x = self.ns_option.x

        # Entity type IDs for learned type embeddings
        entity_types = [node.entity for node in self.ns_entity.items]
        unique_types = sorted(set(entity_types))
        type_to_id = {t: i for i, t in enumerate(unique_types)}
        data[self.ns_entity.type].type_ids = torch.tensor(
            [type_to_id[t] for t in entity_types], dtype=torch.long
        )
        data[self.es_stepmix.type].edge_attr = self.es_stepmix.edge_attr
        data[self.es_summary.type].edge_attr = self.es_summary.edge_attr
        data[self.es_stepmix.type].edge_index = self.es_stepmix.edge_index
        data[self.es_summary.type].edge_index = self.es_summary.edge_index
        data[self.es_tapas.type].edge_index = self.es_tapas.edge_index
        return data.to(device=hardware.device.type)

    def assemble_subgoal(self, option: OptionNode) -> DCScene:
        subgoal = self.start
        for src in option.sources:
            node = self.ns_entity.get_by_key(src[1])
            assert isinstance(node, EntityNode)
            subgoal.set(node.entity, node.data)
        return subgoal

    def set_start(self, start: DCScene):
        self.start = start
        for key in self.start_keys:
            node = self.ns_entity.get_by_key(key)
            assert isinstance(node, EntityNode)
            self.ns_entity.key_update(key, start[node.entity])

        self.update_subgoals()
        self.rebuild()

    def set_goal(self, goal: DCScene):
        self.goal = goal
        for node in self.ns_option.items:
            node.data = goal
        self.update_subgoals()
        self.rebuild()

    def test_subgoal(self, node: EntityNode, x: DCScene) -> bool:
        assert node.con is not None
        return node.con.score_single(x[node.entity].value, node.entity)[1]

    def create_subgoal(self, node: EntityNode, x: DCScene | None = None) -> DCEntity:
        assert node.con is not None
        if x is not None:
            dc_entity = x[node.entity]
        else:
            value = node.con.models[node.entity].sample(1)[0]
            feat = value
            dc_entity = DCEntity(value=value, feature=feat)
        return dc_entity

    def update_subgoals(self):
        for key in self.goal_keys:
            node = self.ns_entity.get_by_key(key)
            assert isinstance(node, EntityNode)
            if self.test_subgoal(node, self.goal):
                x = self.goal.get(node.entity)
            elif self.test_subgoal(node, self.start):
                x = self.start.get(node.entity)
            else:
                x = self.create_subgoal(node)
            self.ns_entity.key_update(key, x)

    def rebuild(self):
        self.es_stepmix.build(self.ns_entity, self.ns_entity)
        self.es_summary.build(self.ns_entity, self.ns_option)
        self.es_tapas.build(self.ns_entity, self.ns_entity)

    def set_comps(
        self,
        tag: str,
        con: Condition,
    ) -> dict[str, set[tuple[str, str]]]:
        keys: dict[str, set[tuple[str, str]]] = defaultdict(set[tuple[str, str]])
        for entity, comps in con.comp_features(self.entities).items():
            for idx, feat in enumerate(comps):
                key = con.label + entity + tag + f"{idx}"
                keys[entity].add((self.es_stepmix.type[1], key))
                self.ns_entity.add(
                    key,
                    EntityNode(
                        entity=entity,
                        data=DCEntity(value=np.empty(0), feature=feat),
                        static=True,
                        sources=set(),
                    ),
                )
        return keys

    def set_precon(
        self, label: str, con: Condition, comp_sources: dict[str, set[tuple[str, str]]]
    ) -> dict[str, tuple[str, str]]:
        pre_sources: dict[str, tuple[str, str]] = {}
        for entity, sources in comp_sources.items():
            key = "pre" + entity + label
            pre_sources[entity] = (self.es_tapas.type[1], key)
            self.start_keys.add(key)
            self.ns_entity.add(
                key=key,
                value=EntityNode(
                    entity=entity,
                    data=DCEntity.empty(),
                    sources=sources,
                    con=con,
                ),
            )
        return pre_sources

    def set_postcon(
        self,
        label: str,
        con: Condition,
        comp_sources: dict[str, set[tuple[str, str]]],
        pre_sources: dict[str, tuple[str, str]],
    ) -> dict[str, tuple[str, str]]:
        post_sources: dict[str, tuple[str, str]] = {}
        for entity, sources in pre_sources.items():
            key = "post" + entity + label
            sources = comp_sources[entity]
            sources.add(pre_sources[entity])
            self.goal_keys.add(key)
            self.ns_entity.add(
                key,
                EntityNode(
                    entity=entity,
                    data=DCEntity.empty(),
                    sources=sources,
                    con=con,
                ),
            )
            post_sources[entity] = (self.es_summary.type[1], key)
        return post_sources

    def set_subgoal(
        self,
        label: str,
        comp_sources: dict[str, set[tuple[str, str]]],
        pre_sources: dict[str, tuple[str, str]],
        post_sources: dict[str, tuple[str, str]],
        subgoal: dict[str, tuple[float, np.ndarray]],
    ) -> set[tuple[str, str]]:
        temp_sources = post_sources
        for entity, (_, value) in subgoal.items():
            key = entity + label
            sources = comp_sources[entity]
            sources.add(pre_sources[entity])
            feat = Entity.gnn_format(value, len(self.entities[entity].cfg.states))
            self.ns_entity.add(
                key,
                EntityNode(
                    entity=entity,
                    data=DCEntity(value=value, feature=feat),
                    static=True,
                    sources=sources,
                ),
            )
            temp_sources[entity] = (self.es_summary.type[1], key)
        return {src for src in temp_sources.values()}

    @classmethod
    def generate(cls, cfgs: list[Agent.Config], entities: set[Entity]) -> "Graph":
        graph = cls(entities={e.cfg.label: e for e in entities})
        agents = [Agent.get(cfg) for cfg in cfgs]
        for a in agents:
            for ac in a.conditions:
                pre_comp_sources = graph.set_comps(ac.label, ac.pre)
                post_comp_sources = graph.set_comps(ac.label, ac.post)
                pre_sources = graph.set_precon(ac.label, ac.pre, pre_comp_sources)
                post_sources = graph.set_postcon(
                    ac.label,
                    ac.post,
                    post_comp_sources,
                    pre_sources,
                )
                for b in agents:
                    for bc in b.conditions:
                        otag = ac.label + bc.label
                        if ac.label + bc.label:  # pre == post
                            sources = {src for src in post_sources.values()}
                            graph.ns_option.add(
                                otag,
                                OptionNode(
                                    agent=a.cfg,
                                    sources=sources,
                                ),
                            )
                        else:  # pre != post
                            subgoal = ac.post.make_subgoal(bc.pre)
                            if subgoal is not None:
                                sources = graph.set_subgoal(
                                    otag,
                                    post_comp_sources,
                                    pre_sources,
                                    post_sources,
                                    subgoal,
                                )
                                graph.ns_option.add(
                                    otag,
                                    OptionNode(
                                        agent=a.cfg,
                                        sources=sources,
                                    ),
                                )

        graph.es_stepmix.edges_from_sets(graph.ns_entity, graph.ns_entity)
        graph.es_summary.edges_from_sets(graph.ns_entity, graph.ns_option)
        graph.es_tapas.edges_from_sets(graph.ns_entity, graph.ns_entity)
        return graph

    def select(self, index: int) -> tuple[Agent.Config, DCScene]:
        node = self.ns_option.idx_get(index)
        assert isinstance(node, OptionNode)
        subgoal = self.assemble_subgoal(node)
        return node.agent, subgoal

    def plot(self, path: Path, figsize=(12, 8), show_labels=False):
        """Visualize the heterogeneous graph."""
        plot_path = path / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)

        G = nx.MultiDiGraph()  # directed, allows multiple edges

        # Add nodes with their type and a label
        for key in self.ns_entity.index.keys():
            G.add_node(key, type="entity", label=key)
        for key in self.ns_option.index.keys():
            G.add_node(key, type="option", label=key)

        # Add edges with their type
        for edge in self.es_stepmix.edges:
            G.add_edge(edge[0], edge[1], type="stepmix")
        for edge in self.es_summary.edges:
            G.add_edge(edge[0], edge[1], type="summary")
        for edge in self.es_tapas.edges:
            G.add_edge(edge[0], edge[1], type="tapas")

        # Position nodes (spring layout)
        pos = nx.spring_layout(G, seed=42)

        # Separate nodes by type for color coding
        entity_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "entity"]
        option_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "option"]

        plt.figure(figsize=figsize)
        # Draw entity nodes (blue)
        nx.draw_networkx_nodes(
            G, pos, nodelist=entity_nodes, node_color="lightblue", node_size=800
        )
        # Draw option nodes (green)
        nx.draw_networkx_nodes(
            G, pos, nodelist=option_nodes, node_color="lightgreen", node_size=800
        )

        # Draw edges with different colors for each relation
        edge_colors = {"stepmix": "gray", "summary": "orange", "tapas": "red"}
        for etype, color in edge_colors.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == etype]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color=color,
                arrows=True,
                arrowsize=10,
                alpha=0.6,
            )

        # Labels (optional)
        if show_labels:
            labels = {n: d["label"] for n, d in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)

        # Legend
        legend_elements = [
            Patch(facecolor="lightblue", label="Entity"),
            Patch(facecolor="lightgreen", label="Option"),
            Patch(facecolor="gray", label="stepmix"),
            Patch(facecolor="orange", label="summary"),
            Patch(facecolor="red", label="tapas"),
        ]
        plt.legend(handles=legend_elements, loc="upper left")
        plt.title("Graph Structure")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path / f"graph.png", dpi=300, bbox_inches="tight")
        plt.close()

    def log(self):
        """Log graph statistics and key attributes."""
        logger.info("=== Graph Summary ===")
        logger.info(f"Entities: {len(self.entities)}")
        logger.info(f"Entity Nodes: {len(self.ns_entity.items)}")
        logger.info(f"Option Nodes: {len(self.ns_option.items)}")
        logger.info(f"StepMix Edges: {len(self.es_stepmix.edges)}")
        logger.info(f"Summary Edges: {len(self.es_summary.edges)}")
        logger.info(f"Tapas Edges: {len(self.es_tapas.edges)}")

        # Optionally log node details
        for key, idx in self.ns_entity.index.items():
            node = self.ns_entity.items[idx]
            logger.debug(
                f"Entity Node: {key} | entity={node.entity} | static={node.static}"
            )
        for key, idx in self.ns_option.index.items():
            node = self.ns_option.items[idx]
            logger.debug(
                f"Option Node: {key} | agent={node.agent} | sources={node.sources}"
            )

        # Log edge sources (first few)
        logger.info("StepMix edge examples:")
        for src, dst in list(self.es_stepmix.edges)[:3]:
            logger.info(f"  {src} -> {dst}")
        # etc.
