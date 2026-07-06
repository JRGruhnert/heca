from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence, TypeVar
import uuid
from heca.conditions.analyzer import ConditionAnalyzer
from heca.conditions.condition import Condition
from heca.conditions.pair import ConditionPair
from heca.misc.td import TDEntity
from itertools import permutations
import torch
from torch_geometric.data import HeteroData

from heca.agents.agent import Agent
from heca.graphs.nodes import *
from heca.graphs.sets import EdgeSet, NodeSet
from heca.misc.td import TDScene
from heca.conditions.condition import Condition

T = TypeVar("T", bound=GraphNode)


class Goal(Enum):
    TRAIN = "train"
    EXPLAIN = "explain"
    EVAL = "evaluate"


class GraphBlueprint:
    def __init__(self, threshold: float):
        self._analyzer = ConditionAnalyzer(threshold)
        self._nodes: dict[str, NodeSet] = field(default_factory=dict)
        self._edges: dict[tuple[str, str, str], EdgeSet] = field(default_factory=dict)
        self._node_staged: set[str] = set()
        self._edge_staged: set[str] = set()
        self._start: TDScene = TDScene({})
        self._goal: TDScene = TDScene({})

    def flush(self) -> "GraphBlueprint":
        self._nodes = {}
        self._edges = {}
        return self

    def build(self):
        for key in self._node_staged:
            self._nodes[key].build()
        for key in self._edge_staged:
            pass

    def size(self) -> dict[str, dict[str, int]]:
        return {
            "nodes": {n: s.size() for n, s in self._nodes.items()},
            "edges": {
                f"{e[0]}_{e[1]}_{e[2]}": s.size() for e, s in self._edges.items()
            },
        }

    def graph(self) -> HeteroData:
        data = HeteroData()
        for label, ns in self._nodes.items():
            num = len(ns.nodes)
            if ns.static_x is not None:
                data[label].x = ns.static_x
            else:
                data[label].x = torch.zeros(num, 1)  # fallback
            data[label].num_nodes = num

        for (src, rel, dst), es in self._edges.items():
            data[(src, rel, dst)].edge_index = es.edge_index
            if es.edge_attr is not None:
                data[(src, rel, dst)].edge_attr = es.edge_attr
        return data

    def nset(self, node: GraphNode) -> str:
        if node.node not in self._nodes:
            self._nodes[node.node] = NodeSet(nodes=[])
        edge_rebuild = self._nodes[node.node].set(node)
        self._node_staged.add(node.node)
        if edge_rebuild:
            self._edge_staged.add(node.node)
        return node.key

    def update_edges(self):
        # 1. Score against goal
        for option in self._nodes[OptionNode.node].nodes:
            goal_score = self._compute_score(goal_cond, tdentity, etag)
            if goal_score >= threshold:
                chosen_dtag = (
                    f"{dtag}_goal"  # or whatever tag you assigned to goal's StepMixNode
                )
            else:
                # 2. Score against start
                start_score = self._compute_score(start_cond, tdentity, etag)
                if start_score >= threshold:
                    chosen_dtag = f"{dtag}_start"
                else:
                    # 3. Sample a new component from the goal model
                    chosen_dtag = self._create_sampled_nodes(goal_cond, etag)

            # Now add entity nodes with the chosen target
            self.nset(PosNode(dtag=chosen_dtag, etag=etag, x=tdentity.pos))
            self.nset(RotNode(dtag=chosen_dtag, etag=etag, x=tdentity.rot))
            self.nset(SteNode(dtag=chosen_dtag, etag=etag, x=tdentity.ste))
            self.nset(EntityNode(dtag=chosen_dtag, etag=etag))

    def _compute_score(self, cond: Condition, tdentity: TDEntity, etag: str) -> float:
        """Use the condition's score method; assume it returns a dict[etag -> float]."""
        scores = cond.score(tdentity)  # you'll need to adapt this to a single entity
        return scores.get(etag, 0.0)

    def _create_sampled_nodes(self, cond: Condition, etag: str) -> str:
        """Sample a new latent representation from cond, create nodes, return new tag."""
        unique_tag = f"sample_{uuid.uuid4().hex[:8]}"
        # Sample: use cond.models[etag].sample(1) or something similar
        sample_data = cond.models[etag].sample(1)[0]  # adjust to your shape
        # Extract pose, rot, state features from sample_data
        # (You may need to adapt your existing Component creation logic)
        # Create the component nodes for this new tag
        # ... (similar to what you do in set_condition)
        self.nset(PreMixNode(atag=..., etag=etag, dtag=unique_tag))
        # Also create PosCompNode, RotCompNode, SteCompNode for each component?
        return unique_tag

    def set_start(self, x: TDScene):
        self.set_entities(x, "start")

    def set_goal(self, x: TDScene):
        self.set_entities(x, "goal")

    def set_entities(self, x: TDScene, tag: str):
        for etag, tde in x.entities():
            ekey = self.nset(EntityNode(dst=etag, tag=tag))
            self.nset(PosNode(dst=ekey, x=tde.pos))
            self.nset(RotNode(dst=ekey, x=tde.rot))
            self.nset(SteNode(dst=ekey, x=tde.ste))

    def set_precondition(self, con: Condition, ptag: str):
        for etag, components in con.model_parameters.items():
            self.set_premix(components, etag, ptag)

    def set_premix(self, components: list[tuple], etag: str, ptag: str):
        mkey = self.nset(PreMixNode(dst=ptag, etag=etag))
        for idx, ct in enumerate(components):
            ckey = self.nset(CompNode(dst=mkey, idx=idx))
            self.nset(PosCompNode(dst=ckey, x=torch.Tensor(ct[0])))
            self.nset(RotCompNode(dst=ckey, x=torch.Tensor(ct[1])))
            self.nset(SteCompNode(dst=ckey, x=torch.Tensor(ct[2])))

    def set_postmix(self, components: list[tuple], etag: str, ptag: str):
        mkey = self.nset(PostMixNode(dst=ptag, etag=etag))
        for idx, ct in enumerate(components):
            ckey = self.nset(CompNode(dst=mkey, idx=idx))
            self.nset(PosCompNode(dst=ckey, x=torch.Tensor(ct[0])))
            self.nset(RotCompNode(dst=ckey, x=torch.Tensor(ct[1])))
            self.nset(SteCompNode(dst=ckey, x=torch.Tensor(ct[2])))

    def set_option(
        self,
        post: Condition,
        pre: Condition,
        agent: Agent.Config,
        ptag: str,
    ):
        values, remaining = self._analyzer.calculate_subgoal(post, pre)
        if self._analyzer.is_subgoal(values):

            for etag, components in post.model_parameters.items():
                self.set_postmix(components, etag, ptag)

    def options(self) -> list[tuple[Agent.Config, TDScene, TDScene]]:
        raise NotImplementedError

    def assemble_subgoal(self) -> TDScene:
        raise NotImplementedError

    def generate(self, cfgs: Sequence[Agent.Config]) -> "GraphBlueprint":
        self.flush()
        agents = [Agent.get(cfg) for cfg in cfgs]
        for a in agents:
            for pair in a.conditions:
                self.set_precondition(pair.pre, pair.label)
                self.set_option(pair.post, pair.post, a.cfg, pair.label)

        for a, b in permutations(agents, 2):
            for ac in a.conditions:
                for bc in b.conditions:
                    self.set_option(ac.post, bc.pre, a.cfg, ac.label)

        return self


# Entity Nodes need to be updated
## Order: Goal, Current, Sample
