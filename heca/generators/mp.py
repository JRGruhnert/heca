from dataclasses import dataclass
from functools import cached_property

import torch

from heca.agents.agent import Agent
from heca.agents.leafs.leaf import LeafAgent
from heca.entities.entity import Entity
from heca.entities.real import RealEntity
from heca.generators.generator import HecaGenerator
from torch_geometric.data import HeteroData

from heca.misc.td import TDScene


class MPGenerator(HecaGenerator):
    @dataclass(kw_only=True)
    class Config(HecaGenerator.Config):
        agents: set[LeafAgent.Query]
        entity: RealEntity.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.agents = [LeafAgent.search(query) for query in self.cfg.agents]
        self.meta = Entity.create(self.cfg.entity)

    def __call__(
        self, x: TDScene, y: TDScene, z: TDScene
    ) -> tuple[list[tuple[LeafAgent.Query, Entity]], HeteroData]:
        options = []
        for agent in self.cfg.agents:
            options.append((agent, self.meta))
        return options, self.to_data(x, y, z)

    def to_data(self, x: TDScene, y: TDScene, z: TDScene) -> HeteroData:
        data = HeteroData()
        data["x"].x = x.v1
        data["y"].x = y.v1
        data["optn"].x = torch.zeros(self.dim_agent, 32)
        data["actor"].x = torch.zeros(self.dim_agent, 1)
        data["critic"].x = torch.zeros(1, 1)

        data[("y", "y-x", "x")].edge_index = self.state_state_sparse
        data[("x", "x-optn", "optn")].edge_index = self.state_skill_full
        # data[("goal", "goal-obs", "obs")].edge_attr = self.cnv.state_state_attr
        data[("x", "x-optn", "optn")].edge_attr = self.s_s_attr(z)
        data[("optn", "optn-actor", "actor")].edge_index = self.skill_skill_sparse
        data[("optn", "optn-critic", "critic")].edge_index = self.skill_to_single
        return data

    def skill_state_distances(
        self, x: TDScene, pad: bool = False, sparse: bool = False
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        for cfg in self.cfg.agents:
            assert isinstance(
                cfg, LeafAgent.Config
            ), "Only LeafAgents are supported in MPGenerator."
            distances = self.distances(x, cfg=cfg, pad=pad, sparse=sparse)
            features.append(distances)
        return torch.stack(features, dim=0).float()

    def distances(
        self, x: TDScene, cfg: LeafAgent.Config, pad: bool, sparse: bool
    ) -> torch.Tensor:
        agent = LeafAgent.search(cfg.query)
        task_features: list[torch.Tensor] = []
        for key, prop in self.meta.properties.items():
            pre = agent.ppre.get(key, None)
            if pre:
                value = torch.tensor([pre, 0.0]) if pad else torch.tensor([pre])
            else:
                nv = -1.0 if sparse else 0.0  # For Identification in filtering
                value = torch.tensor([nv, 1.0]) if pad else torch.tensor([nv])
            task_features.append(value)
        return torch.stack(task_features, dim=0)

    def s_s_attr(self, x: TDScene) -> torch.Tensor:
        dist_matrix = self.skill_state_distances(x, pad=True, sparse=False)
        edge_attr = dist_matrix[self.state_skill_full[1], self.state_skill_full[0]]
        valid_mask = ~(edge_attr == -1).any(dim=1)
        edge_attr = edge_attr[valid_mask]
        return edge_attr

    @property
    def dim_property(self) -> int:
        raise NotImplementedError()

    @property
    def dim_agent(self) -> int:
        raise NotImplementedError()

    @cached_property
    def state_state_full(self) -> torch.Tensor:
        src = (
            torch.arange(self.dim_property)
            .unsqueeze(1)
            .repeat(1, self.dim_property)
            .flatten()
        )
        dst = torch.arange(self.dim_property).repeat(self.dim_property)
        return torch.stack([src, dst], dim=0)

    @cached_property
    def state_skill_full(self) -> torch.Tensor:
        src = (
            torch.arange(self.dim_property)
            .unsqueeze(1)
            .repeat(1, self.dim_agent)
            .flatten()
        )
        dst = torch.arange(self.dim_agent).repeat(self.dim_property)
        return torch.stack([src, dst], dim=0)

    @cached_property
    def state_state_sparse(self) -> torch.Tensor:
        indices = torch.arange(self.dim_property)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def skill_skill_sparse(self) -> torch.Tensor:
        indices = torch.arange(self.dim_agent)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def state_skill_sparse(self) -> torch.Tensor:

        edge_list = []
        for a_idx, agent in enumerate(self.agents):
            for s_idx, property in enumerate(self.meta.properties.keys()):
                if property in agent.ppre.keys():
                    edge_list.append((s_idx, a_idx))
        return torch.tensor(edge_list, dtype=torch.long).t()

    @cached_property
    def skill_to_single(self) -> torch.Tensor:
        indices = torch.arange(self.dim_agent)
        return torch.stack([indices, torch.zeros_like(indices)], dim=0)

    @cached_property
    def single_to_skill(self) -> torch.Tensor:
        indices = torch.arange(self.dim_agent)
        return torch.stack([torch.zeros_like(indices), indices], dim=0)

    @cached_property
    def state_to_single(self) -> torch.Tensor:
        indices = torch.arange(self.dim_property)
        return torch.stack([indices, torch.zeros_like(indices)], dim=0)

    @cached_property
    def single_to_state(self) -> torch.Tensor:
        indices = torch.arange(self.dim_property)
        return torch.stack([torch.zeros_like(indices), indices], dim=0)

    @cached_property
    def state_state_01_attr(self) -> torch.Tensor:
        return (
            (self.state_state_full[0] == self.state_state_full[1])
            .to(torch.float)
            .unsqueeze(-1)
        )

    @cached_property
    def state_skill_01_attr(self) -> torch.Tensor:
        sparse = (
            self.state_skill_sparse[0] * self.dim_agent + self.state_skill_sparse[1]
        )
        full = self.state_skill_full[0] * self.dim_agent + self.state_skill_full[1]
        return torch.isin(full, sparse).float().unsqueeze(-1)
