from collections import defaultdict
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from heca.experts.expert import ExpertModel

from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.edge_set import DEFAULT_TERMS
from heca.graphs.edges.scene_edges import SceneEdges
from heca.graphs.edges.summary_edges import SummaryEdges
from heca.graphs.edges.translation_edges import TranslationEdges
from heca.graphs.nodes.comp_nodes import CompNodes
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import *
from heca.graphs.nodes.option_nodes import OptionNodes
from heca.graphs.nodes.state_nodes import StateNodes
from heca.graphs.data import HecaData
from heca.graphs.roles import ENMode

from heca.data.data import DCEntity, DCScene
from heca.data.entity import Entity
from heca.data.condition import Condition
from heca.data.pair import ConPair

from heca.misc import hardware, logger


class SubgoalMode(Enum):
    BOTH = "both"
    GOAL = "goal"
    CHAIN = "chain"

    def __str__(self):
        return self.value


JITTER_SCOPES: tuple[str, ...] = ("none", "entity", "scene", "both")


class Graph:
    def __init__(
        self,
        entities: dict[str, Entity],
        use_rotation: bool,
        position_jitter: float,
        edge_terms: tuple[str, ...] = DEFAULT_TERMS,
        goal_residual: bool = False,
        jitter_scope: str = "entity",
    ):
        if jitter_scope not in JITTER_SCOPES:
            raise ValueError(
                f"unknown jitter_scope {jitter_scope!r}; "
                f"known: {', '.join(JITTER_SCOPES)}"
            )
        self.entities: dict[str, Entity] = entities
        self.edge_terms: tuple[str, ...] = edge_terms
        self.use_rotation: bool = use_rotation
        self.position_jitter: float = position_jitter
        self.goal_residual: bool = goal_residual
        self.jitter_scope: str = jitter_scope

        self.ns_comp: CompNodes = CompNodes()
        self.ns_entity: EntityNodes = EntityNodes()
        self.ns_option: OptionNodes = OptionNodes()
        self.ns_state: StateNodes = StateNodes()

        self.es_scene: SceneEdges = SceneEdges()
        self.es_summary: SummaryEdges = SummaryEdges()
        self.es_condition: ConditionEdges = ConditionEdges()
        self.es_translation: TranslationEdges = TranslationEdges()

        self.start: DCScene = DCScene.empty()
        self.goal: DCScene = DCScene.empty()

    def build(self, start: DCScene, goal: DCScene, budget: float) -> HecaData:
        self.start = start.copy()
        self.goal = goal.copy()
        self.ns_comp.build(self.use_rotation)
        self.ns_state.build(budget)
        self.ns_entity.build(self.start, self.goal, self.use_rotation)
        self.ns_option.build(self.ns_entity, self.use_rotation)
        self.es_condition.build(
            self.ns_comp,
            self.ns_entity,
            self.use_rotation,
            self.edge_terms,
            self.goal_residual,
        )

        self._jitter_positions()
        self.es_summary.build(self.ns_entity, self.ns_option)
        self.es_translation.build(self.ns_entity, self.ns_entity)
        self.es_scene.build(self.ns_option, self.ns_state)

        data = HecaData()
        # Nodes
        data[self.ns_entity.type].x = self.ns_entity.x
        data[self.ns_entity.type].type_ids = self.ns_entity.type_ids
        data[self.ns_entity.type].entity_ids = self.ns_entity.entity_ids
        data[self.ns_entity.type].role_ids = self.ns_entity.role_ids
        data[self.ns_comp.type].x = self.ns_comp.x
        data[self.ns_comp.type].type_ids = self.ns_comp.type_ids
        data[self.ns_comp.type].weight = self.ns_comp.weights
        data[self.ns_option.type].x = self.ns_option.x
        data[self.ns_option.type].gated = self.ns_option.gated
        data[self.ns_state.type].x = self.ns_state.x
        data[self.ns_state.type].type_ids = self.ns_state.type_ids

        # Edges
        data[self.es_scene.type].edge_index = self.es_scene.edge_index
        data[self.es_condition.type].edge_index = self.es_condition.edge_index
        data[self.es_condition.type].edge_attr = self.es_condition.edge_attr
        data[self.es_summary.type].edge_index = self.es_summary.edge_index
        data[self.es_translation.type].edge_index = self.es_translation.edge_index

        self._validate_export(data)
        # the full device, not just `.type`: "cuda" without an index silently means
        # "whatever torch.cuda.current_device() is", which ignores the per-rank pin
        return data.to(device=hardware.device)

    def jitter_offsets(self) -> dict[str, np.ndarray]:
        limit = self.position_jitter
        if limit <= 0.0 or self.jitter_scope == "none":
            return {label: np.zeros(3, dtype=np.float32) for label in self.entities}
        shared = (
            np.random.uniform(-limit, limit, size=3)
            if self.jitter_scope in ("scene", "both")
            else np.zeros(3, dtype=np.float32)
        )
        if self.jitter_scope == "scene":
            return {label: shared for label in self.entities}
        if self.jitter_scope == "entity":
            return {
                label: np.random.uniform(-limit, limit, size=3)
                for label in self.entities
            }
        return {
            label: shared + np.random.uniform(-limit, limit, size=3)
            for label in self.entities
        }

    def _jitter_positions(self) -> None:
        """Shift every representation of each entity by its offset."""
        if self.position_jitter <= 0.0 or self.jitter_scope == "none":
            return
        offsets = self.jitter_offsets()
        zeros = np.zeros(3, dtype=np.float32)
        self.ns_entity.x = Entity.jitter_positions(
            self.ns_entity.x,
            np.stack(
                [offsets.get(node.entity, zeros) for node in self.ns_entity.items]
            ),
            Entity.layout(self.use_rotation, logstd=False),
        )
        self.ns_comp.x = Entity.jitter_positions(
            self.ns_comp.x,
            np.stack([offsets.get(node.entity, zeros) for node in self.ns_comp.items]),
            Entity.layout(self.use_rotation, logstd=True),
        )

    @property
    def dead_end(self) -> bool:
        gated = getattr(self.ns_option, "gated", None)
        if gated is None:
            raise RuntimeError("dead_end needs a build() first")
        return bool(gated.all()) if gated.numel() else True

    def set_goal_rows(self):
        for label, entity in self.entities.items():
            for role, vmode in (
                (ENRole.START, ENMode.START),
                (ENRole.GOAL, ENMode.GOAL),
            ):
                self.ns_entity.add(
                    vmode.value + label,
                    EntityNode(
                        entity=label,
                        type_id=entity.cfg.type_id,
                        n_states=entity.cfg.n_states,
                        data=DCEntity.empty(),
                        mode=vmode,
                        role=role,
                        sources={CompNodes.type: set(), EntityNodes.type: set()},
                    ),
                )

    def set_comps(self, tag: str, con: Condition) -> dict[str, set[str]]:
        keys: dict[str, set[str]] = defaultdict(set[str])
        for entity, comps in con.comp_features().items():
            for idx, (comp, weight) in enumerate(comps):
                key = con.label + entity + tag + f"{idx}"
                keys[entity].add(key)
                self.ns_comp.add(
                    key,
                    CompNode(
                        entity=entity,
                        type_id=self.entities[entity].cfg.type_id,
                        n_states=self.entities[entity].cfg.n_states,
                        data=DCEntity(value=np.empty(0), feature=comp),
                        weight=weight,
                    ),
                )
        return keys

    def set_precon(self, label: str, con: Condition) -> dict[str, str]:
        comp_sources = self.set_comps(label, con)
        pre_sources: dict[str, str] = {}
        for entity, sources in comp_sources.items():
            key = "pre_" + entity + label
            pre_sources[entity] = key
            self.ns_entity.add(
                key=key,
                value=EntityNode(
                    entity=entity,
                    type_id=self.entities[entity].cfg.type_id,
                    n_states=self.entities[entity].cfg.n_states,
                    data=DCEntity.empty(),
                    sources={CompNodes.type: set(sources)},
                    con=con,
                    mode=ENMode.START,
                    role=ENRole.PRE,
                ),
            )
        return pre_sources

    def set_postcon(
        self,
        label: str,
        con: Condition,
        comp_sources: dict[str, set[str]],
        pre_sources: dict[str, str],
        entities: list[str],
        mode: ENMode,
    ) -> dict[str, str]:
        post_sources: dict[str, str] = {}
        for entity in entities:
            key = mode.value + entity + label
            self.ns_entity.add(
                key,
                EntityNode(
                    entity=entity,
                    type_id=self.entities[entity].cfg.type_id,
                    n_states=self.entities[entity].cfg.n_states,
                    data=DCEntity.empty(),
                    sources={
                        CompNodes.type: set(comp_sources[entity]),
                        EntityNodes.type: {pre_sources[entity]},
                    },
                    con=con,
                    mode=mode,
                    role=ENRole.POST,
                ),
            )
            post_sources[entity] = key
        return post_sources

    def set_subgoal(
        self,
        label: str,
        con: Condition,
        comp_sources: dict[str, set[str]],
        pre_sources: dict[str, str],
        post_sources: dict[str, str],
        subgoal: dict[str, np.ndarray],
    ) -> set[str]:
        temp_sources = dict(post_sources)
        for entity, value in subgoal.items():
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
                    sources={
                        CompNodes.type: set(comp_sources[entity]),
                        EntityNodes.type: {pre_sources[entity]},
                    },
                    con=con,
                    mode=ENMode.SUBGOAL,
                    role=ENRole.POST,
                ),
            )
            temp_sources[entity] = key
        return set(temp_sources.values())

    @classmethod
    def generate(
        cls,
        cfgs: list[ExpertModel.Config],
        smode: SubgoalMode,
        use_rotation: bool,
        position_jitter: float,
        edge_terms: tuple[str, ...] = DEFAULT_TERMS,
        goal_residual: bool = False,
        jitter_scope: str = "entity",
    ) -> "Graph":
        entities = {}
        for cfg in cfgs:
            entities.update(ExpertModel.get(cfg).entities)
        graph = cls(
            entities=entities,
            use_rotation=use_rotation,
            position_jitter=position_jitter,
            edge_terms=edge_terms,
            goal_residual=goal_residual,
            jitter_scope=jitter_scope,
        )
        graph.set_goal_rows()
        agents = [ExpertModel.get(cfg) for cfg in cfgs]

        for a in agents:
            ac = a.conditions
            pre_sources = graph.set_precon(ac.label, ac.pre)
            post_comp_sources = graph.set_comps(ac.label, ac.post)
            post_start_sources = graph.set_postcon(
                ac.label,
                ac.post,
                post_comp_sources,
                pre_sources,
                entities=ac.anchor_entities,
                mode=ENMode.START,
            )
            post_goal_sources = graph.set_postcon(
                ac.label,
                ac.post,
                post_comp_sources,
                pre_sources,
                entities=ac.target_entities,
                mode=ENMode.GOAL,
            )
            post_sources = post_start_sources | post_goal_sources
            use_sample_variant = smode in (SubgoalMode.GOAL, SubgoalMode.BOTH)
            if use_sample_variant:
                post_sample_sources = graph.set_postcon(
                    ac.label,
                    ac.post,
                    post_comp_sources,
                    pre_sources,
                    entities=ac.target_entities,
                    mode=ENMode.SAMPLE,
                )
                post_sources_alt = post_start_sources | post_sample_sources
            for b in agents:
                bc = b.conditions
                if ac.label == bc.label:
                    graph.ns_option.add(
                        ac.label,
                        OptionNode(
                            model=a.cfg,
                            sources={EntityNodes.type: set(post_sources.values())},
                        ),
                    )
                    if use_sample_variant:
                        graph.ns_option.add(
                            ac.label + "s",
                            OptionNode(
                                model=a.cfg,
                                sources={
                                    EntityNodes.type: set(post_sources_alt.values())
                                },
                            ),
                        )
                if smode in (SubgoalMode.CHAIN, SubgoalMode.BOTH):
                    subgoal = bc.pre.make_subgoal(ac.post)
                    if subgoal:
                        sources = graph.set_subgoal(
                            ac.label + "--" + bc.label,
                            ac.post,
                            post_comp_sources,
                            pre_sources,
                            post_sources,
                            subgoal,
                        )
                        graph.ns_option.add(
                            ac.label + "--" + bc.label,
                            OptionNode(
                                model=a.cfg,
                                sources={EntityNodes.type: sources},
                            ),
                        )
        return graph

    def assemble_subgoal(self, option: OptionNode) -> DCScene:
        subgoal = self.start.copy()
        for key in option.sources[EntityNodes.type]:
            node = self.ns_entity.get_by_key(key)
            assert isinstance(node, EntityNode)
            subgoal.set(node.entity, node.data.copy())
        return subgoal

    def select(self, idx: int) -> tuple[ExpertModel.Config, DCScene]:
        node = self.ns_option.idx_get(idx)
        assert isinstance(node, OptionNode)
        subgoal = self.assemble_subgoal(node)
        logger.debug(f"Selected Option: {self.ns_option.key_at(idx)}")
        return node.model, subgoal

    def plot(self, path: Path, figsize=(12, 8), show_labels=True):
        """Visualize the heterogeneous graph."""
        plot_path = path / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)

        G = nx.MultiDiGraph()  # directed, allows multiple edges

        # Build key lookup: index → key (insertion order matches edge indices)
        entity_keys = self.ns_entity.keys
        comp_keys = self.ns_comp.keys
        option_keys = self.ns_option.keys

        # Add nodes with their type and a label
        for key in entity_keys:
            G.add_node(key, type="entity", label=key)
        for key in comp_keys:
            G.add_node(key, type="comp", label=key)
        for key in option_keys:
            G.add_node(key, type="option", label=key)

        # Add edges with their type (resolve positional indices → keys)
        for src, dst in self.es_condition.edges:
            G.add_edge(comp_keys[src], entity_keys[dst], type="stepmix")
        for src, dst in self.es_summary.edges:
            G.add_edge(entity_keys[src], option_keys[dst], type="summary")
        for src, dst in self.es_translation.edges:
            G.add_edge(entity_keys[src], entity_keys[dst], type="tapas")

        # Separate nodes by type for color coding
        entity_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "entity"]
        comp_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "comp"]
        option_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "option"]

        shells = [option_nodes, comp_nodes, entity_nodes]
        pos = nx.shell_layout(G, nlist=shells, scale=3.0)

        plt.figure(figsize=figsize)
        # Draw entity nodes (blue)
        nx.draw_networkx_nodes(
            G, pos, nodelist=entity_nodes, node_color="lightblue", node_size=800
        )
        # Draw component nodes (grey, smaller)
        nx.draw_networkx_nodes(
            G, pos, nodelist=comp_nodes, node_color="lightgrey", node_size=200
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

    def _validate_export(self, data: HecaData) -> None:
        ent = data[self.ns_entity.type]
        assert ent.role_ids.shape[0] == ent.x.shape[0] == ent.type_ids.shape[0]
        assert ent.entity_ids.shape[0] == ent.x.shape[0]
        # one current and one goal row per entity, added in the same order
        cur = (ent.role_ids == ENRole.START.value).nonzero().flatten()
        goal = (ent.role_ids == ENRole.GOAL.value).nonzero().flatten()
        assert cur.numel() == goal.numel() == len(self.entities) > 0
        # current and goal row of the same entity must share its entity id, and
        # entity ids must be injective over the entities
        assert torch.equal(ent.entity_ids[cur], ent.entity_ids[goal])
        assert len(set(ent.entity_ids[cur].tolist())) == len(self.entities)

        opt = data[self.ns_option.type]
        assert opt.gated.shape == (opt.x.shape[0],)
        assert bool(((opt.gated == 0) | (opt.gated == 1)).all())

        slots_op = data[self.ns_state.type]
        gate = data[self.es_scene.type].edge_index
        assert slots_op.x.shape[0] == 1
        assert gate.shape[1] == data[self.ns_option.type].x.shape[0]
        assert int(gate[0].max()) < data[self.ns_option.type].x.shape[0]
        assert int(gate[1].max()) == 0 and int(gate[1].min()) == 0
        assert "edge_attr" not in data[self.es_scene.type]

        comp = data[self.ns_comp.type]
        cond = data[self.es_condition.type].edge_index
        assert comp.x.shape[0] == comp.type_ids.shape[0] == comp.weight.shape[0] > 0
        assert cond.shape[1] > 0
        assert int(cond[0].max()) < comp.x.shape[0]  # src: components
        assert int(cond[1].max()) < ent.x.shape[0]  # dst: value rows
