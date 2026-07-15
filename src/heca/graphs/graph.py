from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import HeteroData

from heca.agents.agent import Agent
from heca.graphs.nodes import *
from heca.graphs.node_set import NodeSet
from heca.graphs.edge_set import EdgeSet
from heca.misc import hardware
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
    packages: dict[str, tuple[Agent.Config, DCScene, DCScene]] = {}
    start_keys: set[str] = set()
    goal_keys: set[str] = set()
    start: DCScene = DCScene.empty()
    goal: DCScene = DCScene.empty()
    goal_ref: dict[str, np.ndarray] = {}

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
            subgoal.set(node.entity, node.data.env)
        return subgoal

    def set_start(self, start: DCScene):
        self.start = start
        for key in self.start_keys:
            node = self.ns_entity.get_by_key(key)
            assert isinstance(node, EntityNode)
            x = start[node.entity]
            gx = self.entities[node.entity].gnn_format(x)
            self.ns_entity.key_update(key, NodeData(env=x, gnn=gx))

        self.update_subgoals()
        self.rebuild()

    def set_goal(self, goal: DCScene):
        self.goal = goal
        self.update_subgoals()
        for k, v in self.goal.entities():
            self.goal_ref[k] = self.entities[k].gnn_format(v)
        self.rebuild()

    def test_subgoal(self, node: EntityNode, x: DCScene) -> bool:
        assert node.con is not None
        return node.con.score_single(x[node.entity], node.entity)[1]

    def create_subgoal(
        self, node: EntityNode, x: DCScene | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        assert node.con is not None
        if x is not None:
            v = x[node.entity]
        else:
            v = node.con.models[node.entity].sample(1)[0]
        return (self.entities[node.entity].gnn_format(v), v)

    def update_subgoals(self):
        for key in self.goal_keys:
            node = self.ns_entity.get_by_key(key)
            assert isinstance(node, EntityNode)
            if self.test_subgoal(node, self.goal):
                gx, x = self.create_subgoal(node, self.goal)
            elif self.test_subgoal(node, self.start):
                gx, x = self.create_subgoal(node, self.start)
            else:
                gx, x = self.create_subgoal(node)
            self.ns_entity.key_update(key, NodeData(env=x, gnn=gx))

    def rebuild(self):
        self.es_stepmix.build(self.ns_entity, self.ns_entity, {})
        self.es_summary.build(self.ns_entity, self.ns_option, self.goal_ref)
        self.es_tapas.build(self.ns_entity, self.ns_entity, {})

    def set_comps(
        self,
        tag: str,
        con: Condition,
    ) -> dict[str, set[tuple[str, str]]]:
        keys: dict[str, set[tuple[str, str]]] = defaultdict(set[tuple[str, str]])
        for entity, comps in con.parameters.items():
            for idx, data in enumerate(comps):
                key = con.label + entity + tag + f"{idx}"
                keys[entity].add((self.es_stepmix.type[1], key))
                self.ns_entity.add(
                    key,
                    EntityNode(
                        entity=entity,
                        data=NodeData(gnn=data),
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
                    data=NodeData(),
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
                    data=NodeData(),
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
        for entity, value in subgoal.items():
            key = entity + label
            sources = comp_sources[entity]
            sources.add(pre_sources[entity])
            data = self.entities[entity].gnn_format(value[1])
            self.ns_entity.add(
                key,
                EntityNode(
                    entity=entity,
                    data=NodeData(gnn=data, env=value[1]),
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
