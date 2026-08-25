from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from heca.experts.expert import ExpertModel
from heca.graphs.node import *
from heca.graphs.node_set import NodeSet
from heca.graphs.edge_set import EdgeSet
from heca.misc import hardware, logger
from heca.data.data import DCScene
from heca.data.entity import Entity
from heca.conditions.condition import Condition


class Graph:
    def __init__(self, entities: dict[str, Entity]):
        self.entities: dict[str, Entity] = entities
        self.ns_entity: NodeSet[EntityNode] = NodeSet[EntityNode]("entity")

        self.ns_option: NodeSet[OptionNode] = NodeSet[OptionNode]("option")

        self.es_summary: EdgeSet[EntityNode, OptionNode] = EdgeSet[
            EntityNode, OptionNode
        ](("entity", "summary", "option"))
        self.es_stepmix: EdgeSet[EntityNode, EntityNode] = EdgeSet[
            EntityNode, EntityNode
        ](("entity", "stepmix", "entity"))

        self.es_tapas: EdgeSet[EntityNode, EntityNode] = EdgeSet[
            EntityNode, EntityNode
        ](("entity", "tapas", "entity"))

        self.start_keys: set[str] = set()
        self.goal_keys: set[str] = set()
        self.start: DCScene = DCScene.empty()
        self.goal: DCScene = DCScene.empty()

    def export(self) -> HeteroData:
        data = HeteroData()
        data[self.ns_entity.type].x = self.ns_entity.x
        data[self.ns_option.type].x = self.ns_option.x
        data[self.ns_entity.type].type_ids = self.ns_entity.type_ids

        data[self.es_stepmix.type].edge_attr = self.es_stepmix.edge_attr
        data[self.es_summary.type].edge_attr = self.es_summary.edge_attr
        data[self.es_stepmix.type].edge_index = self.es_stepmix.edge_index
        data[self.es_summary.type].edge_index = self.es_summary.edge_index
        data[self.es_tapas.type].edge_index = self.es_tapas.edge_index
        return data.to(device=hardware.device.type)

    def set_start(self, start: DCScene):
        self.start = start.copy()
        for key in self.start_keys:
            node = self.ns_entity.get_by_key(key)
            assert isinstance(node, EntityNode)
            self.ns_entity.key_update(key, start[node.entity])

        self.update_nodes()
        self.rebuild()

    def set_goal(self, goal: DCScene):
        self.goal = goal.copy()
        for node in self.ns_option.items:
            node.data = goal.copy()

    def test_value(self, node: EntityNode, x: DCScene) -> bool:
        assert node.con is not None
        up = node.con.models[node.entity].get_parameters().copy()
        return node.con.entities[node.entity].score_single(x[node.entity].value, up)[1]

    def create_value(self, node: EntityNode) -> DCEntity:
        assert node.con is not None
        value = node.con.models[node.entity].sample(1)[0]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = value.squeeze()
        value = self.entities[node.entity].model_to_value(value)
        feat = self.entities[node.entity].gnn_format(value)
        return DCEntity(value=value, feature=feat)

    def update_nodes(self):
        for key in self.goal_keys:
            node = self.ns_entity.get_by_key(key)
            logger.debug(f"Update Subgoal {node.entity} {key}")
            assert isinstance(node, EntityNode)
            if (
                node.change_score is not None
                and node.change_score < Entity.ANCHOR_THRESHOLD
            ):
                x = self.start.get(node.entity)
                logger.debug(f"From Start (anchor): {x}")
            elif self.test_value(node, self.goal):
                x = self.goal.get(node.entity)
                logger.debug(f"From Goal:   {x}")
            elif self.test_value(node, self.start):
                x = self.start.get(node.entity)
                logger.debug(f"From Start:  {x}")
            else:
                x = self.create_value(node)
                logger.debug(f"From New:    {x}")
            self.ns_entity.key_update(key, x)

    def assemble_subgoal(self, option: OptionNode) -> DCScene:
        subgoal = self.start.copy()
        for src in option.sources:
            node = self.ns_entity.get_by_key(src[1])
            assert isinstance(node, EntityNode)
            subgoal.set(node.entity, node.data)
        return subgoal

    def __str__(self) -> str:
        lines = ["=== Graph ==="]
        lines.append(f"Entities: {len(self.entities)}")
        lines.append(str(self.ns_entity))
        lines.append(str(self.ns_option))
        lines.append(f"StepMix: {self.es_stepmix}")
        lines.append(f"Summary: {self.es_summary}")
        lines.append(f"Tapas:   {self.es_tapas}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def rebuild(self):
        self.ns_entity.build()
        self.ns_option.build()
        self.es_stepmix.build(self.ns_entity, self.ns_entity)
        self.es_summary.build(self.ns_entity, self.ns_option)
        self.es_tapas.build(self.ns_entity, self.ns_entity)

    def set_comps(self, tag: str, con: Condition) -> dict[str, set[tuple[str, str]]]:
        keys: dict[str, set[tuple[str, str]]] = defaultdict(set[tuple[str, str]])
        for entity, comps in con.comp_features().items():
            for idx, feat in enumerate(comps):
                key = con.label + entity + tag + f"{idx}"
                keys[entity].add((self.es_stepmix.type[1], key))
                self.ns_entity.add(
                    key,
                    EntityNode(
                        entity=entity,
                        type_id=self.entities[entity].cfg.type_id,
                        n_states=self.entities[entity].cfg.n_states,
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
            key = "pre_" + entity + label
            pre_sources[entity] = (self.es_tapas.type[1], key)
            self.start_keys.add(key)
            self.ns_entity.add(
                key=key,
                value=EntityNode(
                    entity=entity,
                    type_id=self.entities[entity].cfg.type_id,
                    n_states=self.entities[entity].cfg.n_states,
                    data=DCEntity.empty(),
                    sources=set(sources),
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
        change_scores: dict[str, float] | None = None,
    ) -> dict[str, tuple[str, str]]:
        post_sources: dict[str, tuple[str, str]] = {}
        for entity, sources in pre_sources.items():
            key = "post_" + entity + label
            sources = set(comp_sources[entity])
            sources.add(pre_sources[entity])
            self.goal_keys.add(key)
            self.ns_entity.add(
                key,
                EntityNode(
                    entity=entity,
                    type_id=self.entities[entity].cfg.type_id,
                    n_states=self.entities[entity].cfg.n_states,
                    data=DCEntity.empty(),
                    sources=sources,
                    con=con,
                    change_score=(change_scores.get(entity) if change_scores else None),
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
            key = "sub_" + entity + label
            sources = set(comp_sources[entity])
            sources.add(pre_sources[entity])
            feat = self.entities[entity].gnn_format(value)
            self.ns_entity.add(
                key,
                EntityNode(
                    entity=entity,
                    type_id=self.entities[entity].cfg.type_id,
                    n_states=self.entities[entity].cfg.n_states,
                    data=DCEntity(value=value, feature=feat),
                    static=True,
                    sources=sources,
                ),
            )
            temp_sources[entity] = (self.es_summary.type[1], key)
        return {src for src in temp_sources.values()}

    @classmethod
    def generate(cls, cfgs: list[ExpertModel.Config], add_subgoals: bool) -> "Graph":
        entities = {}
        for cfg in cfgs:
            entities.update(ExpertModel.get(cfg).entities)
        graph = cls(entities=entities)
        agents = [ExpertModel.get(cfg) for cfg in cfgs]
        # Track expert-condition connections for visualization.
        _connections: dict[tuple[str, str], dict[str, float]] = {}
        _pair_scores: dict[tuple[str, str], dict[str, float]] = {}
        for a in agents:
            ac = a.conditions
            pre_comp_sources = graph.set_comps(ac.label, ac.pre)
            post_comp_sources = graph.set_comps(ac.label, ac.post)
            pre_sources = graph.set_precon(ac.label, ac.pre, pre_comp_sources)
            post_sources = graph.set_postcon(
                ac.label,
                ac.post,
                post_comp_sources,
                pre_sources,
                change_scores=ac.change_scores,
            )
            for b in agents:
                bc = b.conditions
                if ac.label == bc.label:  # pre == post
                    sources = {src for src in post_sources.values()}
                    graph.ns_option.add(
                        ac.label,
                        OptionNode(
                            model=a.cfg,
                            sources=sources,
                        ),
                    )
                else:  # pre != post
                    if add_subgoals:
                        _pair_scores[(a.cfg.tag, b.cfg.tag)] = bc.pre.scores(ac.post)
                        subgoal = bc.pre.make_subgoal(ac.post)
                        if subgoal is not None:
                            _connections[(a.cfg.tag, b.cfg.tag)] = {
                                key: float(score) for key, (score, _) in subgoal.items()
                            }
                            sources = graph.set_subgoal(
                                ac.label + bc.label,
                                post_comp_sources,
                                pre_sources,
                                post_sources,
                                subgoal,
                            )
                            graph.ns_option.add(
                                ac.label + bc.label,
                                OptionNode(
                                    model=a.cfg,
                                    sources=sources,
                                ),
                            )

        graph.es_stepmix.edges_from_sets(graph.ns_entity, graph.ns_entity)
        graph.es_summary.edges_from_sets(graph.ns_entity, graph.ns_option)
        graph.es_tapas.edges_from_sets(graph.ns_entity, graph.ns_entity)

        # Persist the connection metadata so plot_connections can render it.
        graph._agent_tags = [a.cfg.tag for a in agents]
        graph._connections = _connections
        graph._pair_scores = _pair_scores
        return graph

    def select(self, index: int) -> tuple[ExpertModel.Config, DCScene]:
        node = self.ns_option.idx_get(index)
        assert isinstance(node, OptionNode)
        subgoal = self.assemble_subgoal(node)
        logger.debug(f"Selected Option: {self.ns_option.key_at(index)}")
        return node.model, subgoal

    def plot(self, path: Path, figsize=(12, 8), show_labels=True):
        """Visualize the heterogeneous graph."""
        plot_path = path / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)

        G = nx.MultiDiGraph()  # directed, allows multiple edges

        # Build key lookup: index → key (insertion order matches edge indices)
        entity_keys = self.ns_entity.keys
        option_keys = self.ns_option.keys

        # Add nodes with their type and a label
        for key in entity_keys:
            G.add_node(key, type="entity", label=key)
        for key in option_keys:
            G.add_node(key, type="option", label=key)

        # Add edges with their type (resolve positional indices → keys)
        for src, dst in self.es_stepmix.edges:
            G.add_edge(entity_keys[src], entity_keys[dst], type="stepmix")
        for src, dst in self.es_summary.edges:
            G.add_edge(entity_keys[src], option_keys[dst], type="summary")
        for src, dst in self.es_tapas.edges:
            G.add_edge(entity_keys[src], entity_keys[dst], type="tapas")

        # Separate nodes by type for color coding
        entity_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "entity"]
        option_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "option"]

        # Position nodes (spring layout)
        # pos = nx.spring_layout(G, seed=42, k=2.0)
        shells = [option_nodes, entity_nodes]
        pos = nx.shell_layout(G, nlist=shells, scale=3.0)

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
        plt.savefig(plot_path / f"graph.png", dpi=300, bbox_inches="tight")
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
        entity_lines = []
        for key, idx in self.ns_entity.index.items():
            node = self.ns_entity.items[idx]
            entity_lines.append(
                f"{idx}:\tstatic={int(node.static)}\t{node.entity}\t{key}"
            )
        logger.debug(f"Entity Nodes:\n" + "\n".join(entity_lines))

        option_lines = []
        for key, idx in self.ns_option.index.items():
            node = self.ns_option.items[idx]
            option_lines.append(f"{idx}:\tagent={node.model.tag}\t\t{key}")
        logger.debug(f"Option Nodes:\n" + "\n".join(option_lines))

        stepmix_lines = []
        for src, dst in list(self.es_stepmix.edges):
            stepmix_lines.append(f"({src}->{dst})")
        logger.info("StepMix edges:\n" + ", ".join(stepmix_lines))

        tapas_lines = []
        for src, dst in list(self.es_tapas.edges):
            tapas_lines.append(f"({src}->{dst})")
        logger.info("Tapas edges:\n" + ", ".join(tapas_lines))

        summary_lines = []
        for src, dst in list(self.es_summary.edges):
            summary_lines.append(f"({src}->{dst})")
        logger.info("Summary edges:\n" + ", ".join(summary_lines))

    def plot_connections(self, path: Path, figsize=(10, 8)):
        """Plot expert-condition connections computed during generation.

        Cell ``(i, j)`` is the minimum containment score over shared entities
        for the pair ``post(model i) -> pre(model j)``, regardless of whether
        it passed the thresholds. Empty cells mean the two models share no
        entity. Diagonal entries are self-connections (direct options).
        """
        tags = getattr(self, "_agent_tags", [])
        connections = getattr(self, "_connections", {})
        pair_scores = getattr(self, "_pair_scores", {})
        if not tags:
            return

        n = len(tags)
        mat = np.full((n, n), np.nan)
        for (src, dst), entity_scores in pair_scores.items():
            i = tags.index(src)
            j = tags.index(dst)
            values = entity_scores.values()
            mat[i, j] = min(values) if values else np.nan

        plot_path = path / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgray")
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tags, fontsize=8)
        ax.set_xlabel("pre-condition (target model)")
        ax.set_ylabel("post-condition (source model)")

        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(
                        j,
                        i,
                        "self",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )
                elif not np.isnan(mat[i, j]):
                    val = float(mat[i, j])
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val < 0.5 else "black",
                    )
                else:
                    ax.text(
                        j, i, "—", ha="center", va="center", fontsize=7, color="black"
                    )

        fig.colorbar(im, ax=ax, label="min containment score")
        ax.set_title("Condition connections (min entity containment score)")
        fig.tight_layout()
        fig.savefig(plot_path / "connections.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Per-pair per-entity detail plots (including failed pairs, so the
        # matrix's low-score cells can be understood).
        detail_dir = plot_path / "connections"
        detail_dir.mkdir(parents=True, exist_ok=True)
        for (src, dst), entity_scores in sorted(pair_scores.items()):
            labels = list(entity_scores.keys())
            if not labels:
                continue
            values = [entity_scores[k][0] for k in labels]
            thresholds = [entity_scores[k][1] for k in labels]

            fig, ax = plt.subplots(figsize=(max(4.0, len(labels) * 1.3), 3.2))
            x = np.arange(len(labels))
            colors = [
                "tab:green" if v >= t else "tab:red" for v, t in zip(values, thresholds)
            ]
            ax.bar(x, values, color=colors, alpha=0.85)
            for i, (v, t) in enumerate(zip(values, thresholds)):
                ax.hlines(
                    t,
                    x[i] - 0.4,
                    x[i] + 0.4,
                    color="gray",
                    linestyle="--",
                    linewidth=1.0,
                )
                ax.text(i, max(v, t) + 0.02, f"{v:.2f}", ha="center", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylim(0.0, 1.1)
            ax.set_ylabel("containment score")
            ax.set_xlabel("entity")
            ax.set_title(f"{src} -> {dst} (post -> pre)")
            fig.tight_layout()
            fig.savefig(detail_dir / f"{src}__{dst}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Per-entity text summary.
        for (src, dst), scores in sorted(connections.items()):
            parts = ", ".join(f"{k}={v:.3f}" for k, v in sorted(scores.items()))
            logger.info(f"connection {src} -> {dst}: {parts}")
