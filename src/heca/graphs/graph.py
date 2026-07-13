from collections import defaultdict

import numpy as np
from torch_geometric.data import HeteroData

from heca.agents.agent import Agent
from heca.graphs.nodes import *
from heca.graphs.node_set import NodeSet
from heca.graphs.edge_set import EdgeSet
from heca.misc import hardware
from heca.misc.data import DCEntity, DCScene
from heca.misc.entity import Entity
from heca.conditions.condition import Condition
from heca.conditions.analyzer import ConditionAnalyzer


class GraphBlueprint:
    def __init__(self, threshold: float):
        self.analyzer = ConditionAnalyzer(threshold)

        self.ns_entity: NodeSet = NodeSet("entity")
        self.ns_option: NodeSet = NodeSet("option")
        self.es_summary: EdgeSet = EdgeSet(("entity", "summary", "option"))
        self.es_stepmix: EdgeSet = EdgeSet(("entity", "stepmix", "entity"))
        self.es_tapas: EdgeSet = EdgeSet(("entity", "tapas", "entity"))

        self.packages: dict[str, tuple[Agent.Config, DCScene, DCScene]]

        self.start: DCScene = DCScene.empty()
        self.goal: DCScene = DCScene.empty()
        self.entities: dict[str, Entity] = {}

        self.stempix_edge: tuple[str, str, str] = ("entity", "stepmix", "entity")
        self.summary_edge: tuple[str, str, str] = ("entity", "summary", "option")
        self.tapas_edge: tuple[str, str, str] = ("entity", "tapas", "entity")

    def graph(self) -> HeteroData:
        data = HeteroData()
        data["entity"].x = self.ns_entity.x
        data["option"].x = self.ns_option.x
        data[self.stempix_edge].edge_attr = self.es_stepmix.edge_attr
        data[self.summary_edge].edge_attr = self.es_summary.edge_attr
        data[self.stempix_edge].edge_index = self.es_stepmix.edge_index
        data[self.summary_edge].edge_index = self.es_summary.edge_index
        data[self.tapas_edge].edge_index = self.es_tapas.edge_index
        return data.to(device=hardware.device.type)

    def assemble_subgoal(self, option: OptionNode) -> DCScene:
        raise NotImplementedError

    def pre_tag(self, entity: str):
        return f"{entity}-pre"

    def post_tag(self, entity: str):
        return f"{entity}-post"

    def preprocess_entity(self, x: DCEntity) -> np.ndarray:
        raise NotImplementedError
        Entity.gnn_format()

    def set_start(self, start: DCScene):
        self.start = start
        for label, entity in start.entities():
            x = self.preprocess_entity(entity)
            self.ns_entity.update(tag=self.pre_tag(label), x=x)

        self.es_stepmix.build(self.ns_entity, self.ns_entity)
        self.es_summary.build(self.ns_entity, self.ns_option)
        self.es_tapas.build(self.ns_entity, self.ns_entity)

    def set_goal(self, goal: DCScene):
        self.goal = goal
        for label, entity in goal.entities():
            x = self.preprocess_entity(entity)
            self.ns_entity.update(tag=self.post_tag(label), x=x)

        self.es_stepmix.build(self.ns_entity, self.ns_entity)
        self.es_summary.build(self.ns_entity, self.ns_option)
        self.es_tapas.build(self.ns_entity, self.ns_entity)

    def set_comps(
        self,
        tag: str,
        con: Condition,
    ) -> dict[str, set[tuple[str, str]]]:
        keys: dict[str, set[tuple[str, str]]] = defaultdict(set[tuple[str, str]])
        for entity, comps in con.parameters.items():
            for idx, data in enumerate(comps):
                key = con.label + entity + tag + f"{idx}"
                keys[entity].add((self.stempix_edge[1], key))
                self.ns_entity.add(
                    key,
                    EntityNode(
                        entity=entity,
                        tag="",
                        data=data,
                        sources=set(),
                    ),
                )
        return keys

    def set_precon(
        self, label: str, comp_sources: dict[str, set[tuple[str, str]]]
    ) -> dict[str, tuple[str, str]]:
        pre_sources: dict[str, tuple[str, str]] = {}
        for entity, sources in comp_sources.items():
            key = "pre" + entity + label
            pre_sources[entity] = (self.tapas_edge[1], key)
            self.ns_entity.add(
                key=key,
                value=EntityNode(
                    entity=entity,
                    tag=self.pre_tag(entity),
                    sources=sources,
                ),
            )
        return pre_sources

    def set_postcon(
        self,
        label: str,
        comp_sources: dict[str, set[tuple[str, str]]],
        pre_sources: dict[str, tuple[str, str]],
    ) -> dict[str, tuple[str, str]]:
        post_sources: dict[str, tuple[str, str]] = {}
        for entity, sources in pre_sources.items():
            key = "post" + entity + label
            sources = comp_sources[entity]
            sources.add(pre_sources[entity])
            self.ns_entity.add(
                key,
                EntityNode(
                    entity=entity,
                    tag=self.post_tag(entity),
                    sources=sources,
                ),
            )
            post_sources[entity] = (self.summary_edge[1], key)
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
            self.ns_entity.add(
                key,
                EntityNode(
                    entity=entity,
                    tag="",
                    data=value[1],
                    static=True,
                    sources=sources,
                ),
            )
            temp_sources[entity] = (self.summary_edge[1], key)
        return {src for src in temp_sources.values()}

    def generate(
        self, cfgs: list[Agent.Config], entities: set[Entity]
    ) -> "GraphBlueprint":
        self.entities = {e.cfg.label: e for e in entities}
        agents = [Agent.get(cfg) for cfg in cfgs]
        for a in agents:
            for ac in a.conditions:
                pre_comp_sources = self.set_comps(ac.label, ac.pre)
                post_comp_sources = self.set_comps(ac.label, ac.post)
                pre_sources = self.set_precon(ac.label, pre_comp_sources)
                post_sources = self.set_postcon(
                    ac.label,
                    post_comp_sources,
                    pre_sources,
                )
                for b in agents:
                    for bc in b.conditions:
                        otag = ac.label + bc.label
                        if ac.label + bc.label:  # pre == post
                            sources = {src for src in post_sources.values()}
                            self.ns_option.add(
                                otag,
                                OptionNode(
                                    tag=otag,
                                    agent=a.cfg,
                                    sources=sources,
                                ),
                            )
                        else:  # pre != post
                            subgoal = self.analyzer.make_subgoal(ac.post, bc.pre)
                            if self.analyzer.is_subgoal(subgoal):
                                sources = self.set_subgoal(
                                    otag,
                                    post_comp_sources,
                                    pre_sources,
                                    post_sources,
                                    subgoal,
                                )
                                self.ns_option.add(
                                    otag,
                                    OptionNode(
                                        tag=otag,
                                        agent=a.cfg,
                                        sources=sources,
                                    ),
                                )

        nsets = {nset.type: nset for nset in [self.ns_entity, self.ns_option]}
        for eset in [self.es_stepmix, self.es_summary, self.es_tapas]:
            etype = eset.type[1]
            snset = nsets[eset.type[0]]
            tnset = nsets[eset.type[2]]
            for i, node in enumerate(snset.items):
                for e, key in node.sources:
                    if e == etype:
                        j = tnset.get_index(key)
                        eset.add(i, j)
        return self

    def options(self) -> list[tuple[Agent.Config, DCScene]]:
        values: list[tuple[Agent.Config, DCScene]] = []
        for option in self.ns_option.items:
            assert isinstance(option, OptionNode)
            subgoal = self.assemble_subgoal(option)
            values.append((option.agent, subgoal))
        return values
