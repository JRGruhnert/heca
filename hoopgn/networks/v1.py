from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Union, cast
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GINConv, GINEConv
from hoopgn.explainer import HoopgnExplainer
from hoopgn.observation.td_properties import TDProperties
from hoopgn.networks.layers.mlp import GinStandardMLP, UnactivatedMLP
from hoopgn.networks.network import Network, NetworkConfig
from hoopgn.agents.agent import Agent
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyCondition,
)
from hoopgn.environments.properties.property import Property
from torch_geometric.data import HeteroData
from torch_geometric.explain import CaptumExplainer, HeteroExplanation


class ReadoutNetwork(nn.Module):
    def __init__(
        self,
        dim_features: int,
    ):
        super().__init__()
        self.state_state_gin = GINConv(
            nn=GinStandardMLP(
                in_dim=dim_features,
                out_dim=dim_features,
                hidden_dim=dim_features,
            ),
            # edge_dim=1,
        )

        self.state_skill_gin = GINEConv(
            nn=GinStandardMLP(
                in_dim=dim_features,
                out_dim=dim_features,
                hidden_dim=dim_features,
            ),
            edge_dim=2,
        )
        self.readout_net = GINConv(
            nn=UnactivatedMLP(dim_features, 1),
        )

    @abstractmethod
    def readout(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict
    ) -> torch.Tensor:
        raise NotImplementedError("Readout method must be implemented by subclass.")

    def set_edge_attr_dict(self, edge_attr_dict):
        self.edge_attr_dict = edge_attr_dict

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # for a in args:
        #    print(type(a))
        # for k, v in kwargs.items():
        #    print(k, type(v))

        if len(args) == 1 and isinstance(args[0], Batch):
            batch = args[0]
            x_dict = batch.x_dict  # type: ignore
            edge_index_dict = batch.edge_index_dict  # type: ignore
            edge_attr_dict = batch.edge_attr_dict  # type: ignore
        elif len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            x_dict = args[0]
            edge_index_dict = args[1]
            edge_attr_dict = self.edge_attr_dict
        elif (
            len(args) == 3
            and isinstance(args[0], dict)
            and isinstance(args[1], dict)
            and isinstance(args[2], dict)  # This is likely the mask from Captum
        ):
            x_dict = args[0]
            edge_index_dict = args[1]
            edge_attr_dict = args[2]
            # args[2] is the mask, which you may need to apply to x_dict or ignore
        elif len(args) == 3 and isinstance(args[0], dict) and isinstance(args[1], dict):
            x_dict = args[0]
            edge_index_dict = args[1]
            edge_attr_dict = self.edge_attr_dict
        else:
            raise ValueError(
                f"Invalid arguments {len(args)}. Expected (Batch), (x_dict, edge_index_dict), (x_dict, edge_index_dict, mask), or (x_dict, edge_index_dict, edge_attr_dict)."
            )

        x1 = self.state_state_gin(
            x=(x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
            # edge_attr=edge_attr_dict[("goal", "goal-obs", "obs")],
        )
        x2 = self.state_skill_gin(
            x=(x1, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
            edge_attr=edge_attr_dict[("obs", "obs-task", "task")],
        )

        return self.readout(x2, x_dict, edge_index_dict)


class ActorReadoutNetwork(ReadoutNetwork):
    def readout(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict
    ) -> torch.Tensor:
        logits: torch.Tensor = self.readout_net(
            x=(x, x_dict["actor"]),
            edge_index=edge_index_dict[("task", "task-actor", "actor")],
        )
        return logits.view(1, -1)


class CriticReadoutNetwork(ReadoutNetwork):
    def readout(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict
    ) -> torch.Tensor:
        value: torch.Tensor = self.readout_net(
            x=(x, x_dict["critic"]),
            edge_index=edge_index_dict[("task", "task-critic", "critic")],
        )
        return value.squeeze(-1)


@dataclass(kw_only=True)
class HoopgnV1Config(NetworkConfig):
    explain_mode: bool = False
    label: str = "gnn"


class HoopgnV1Network(Network):

    def __init__(
        self,
        config: HoopgnV1Config,
    ):
        super().__init__(config)

        self.actor = ActorReadoutNetwork(dim_features=self.config.registry.dim_encoder)
        self.critic = CriticReadoutNetwork(
            dim_features=self.config.registry.dim_encoder
        )
        self.actor_explainer = HoopgnExplainer(
            self.actor,
            algorithm=CaptumExplainer("Saliency"),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="probs",
            ),
        )

        self.critic_explainer = HoopgnExplainer(
            self.critic,
            algorithm=CaptumExplainer("Saliency"),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="probs",
            ),
        )

    @cached_property
    def state_state_full(self) -> torch.Tensor:
        src = (
            torch.arange(self.config.dim_state)
            .unsqueeze(1)
            .repeat(1, self.config.dim_state)
            .flatten()
        )
        dst = torch.arange(self.config.dim_state).repeat(self.config.dim_state)
        return torch.stack([src, dst], dim=0)

    @cached_property
    def state_skill_full(self) -> torch.Tensor:
        src = (
            torch.arange(self.config.dim_state)
            .unsqueeze(1)
            .repeat(1, self.config.dim_skill)
            .flatten()
        )
        dst = torch.arange(self.config.dim_skill).repeat(self.config.dim_state)
        return torch.stack([src, dst], dim=0)

    @cached_property
    def state_state_sparse(self) -> torch.Tensor:
        indices = torch.arange(self.config.dim_state)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def skill_skill_sparse(self) -> torch.Tensor:
        indices = torch.arange(self.config.dim_skill)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def state_skill_sparse(self) -> torch.Tensor:
        edge_list = []
        for task_idx, skill in enumerate(self.skills):
            for state_idx, state in enumerate(self.states):
                if state.cfg.label in skill.precons.keys():
                    edge_list.append((state_idx, task_idx))
        return torch.tensor(edge_list, dtype=torch.long).t()

    @cached_property
    def skill_to_single(self) -> torch.Tensor:
        indices = torch.arange(self.config.dim_skill)
        return torch.stack([indices, torch.zeros_like(indices)], dim=0)

    @cached_property
    def single_to_skill(self) -> torch.Tensor:
        indices = torch.arange(self.config.dim_skill)
        return torch.stack([torch.zeros_like(indices), indices], dim=0)

    @cached_property
    def state_to_single(self) -> torch.Tensor:
        indices = torch.arange(self.config.dim_state)
        return torch.stack([indices, torch.zeros_like(indices)], dim=0)

    @cached_property
    def single_to_state(self) -> torch.Tensor:
        indices = torch.arange(self.config.dim_state)
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
            self.state_skill_sparse[0] * self.config.dim_skill
            + self.state_skill_sparse[1]
        )
        full = (
            self.state_skill_full[0] * self.config.dim_skill + self.state_skill_full[1]
        )
        return torch.isin(full, sparse).float().unsqueeze(-1)

    def forward(
        self,
        batch: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(batch), self.critic(batch)  # Logits, Value

    def explain(
        self,
        current: TDProperties,
        goal: TDProperties,
        skill: Agent,
    ) -> tuple[Union[HeteroExplanation, None], Union[HeteroExplanation, None]]:
        if isinstance(current, TDProperties):
            x = [current]
        else:
            x = current
        if isinstance(goal, TDProperties):
            y = [goal]
        else:
            y = goal
        batch = self.to_batch(x, y)
        d: HeteroData = batch.get_example(0)  # type: ignore
        # print(batch)
        actor_explanation = self.actor_explainer(
            d.x_dict,
            d.edge_index_dict,
            d.edge_attr_dict,
            index=torch.tensor([0]),
        )
        critic_explanation = self.critic_explainer(
            d.x_dict,
            d.edge_index_dict,
            d.edge_attr_dict,
            index=torch.tensor([0]),
        )
        assert isinstance(actor_explanation, HeteroExplanation)
        assert isinstance(critic_explanation, HeteroExplanation)
        return actor_explanation, critic_explanation

    def _to_batch(
        self, current: list[Tensor], goal: list[Tensor], obs: list[TDProperties]
    ) -> Batch:
        batch_data = []
        for c, g, o in zip(current, goal, obs):
            data = HeteroData()
            data["goal"].x = g
            data["obs"].x = c
            data["task"].x = torch.zeros(
                self.config.dim_skill, self.config.registry.dim_encoder
            )
            data["actor"].x = torch.zeros(self.config.dim_skill, 1)
            data["critic"].x = torch.zeros(1, 1)

            data[("goal", "goal-obs", "obs")].edge_index = self.state_state_sparse
            data[("obs", "obs-task", "task")].edge_index = self.state_skill_full
            # data[("goal", "goal-obs", "obs")].edge_attr = self.cnv.state_state_attr
            data[("obs", "obs-task", "task")].edge_attr = self.s_s_attr(o)
            data[("task", "task-actor", "actor")].edge_index = self.skill_skill_sparse
            data[("task", "task-critic", "critic")].edge_index = self.skill_to_single
            batch_data.append(data)
        return cast(Batch, Batch.from_data_list(batch_data))

    def skill_state_distances(
        self,
        obs: TDProperties,
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        for skill in self.skills:
            distances = self.distances(
                obs, precons=skill.precons, pad=pad, sparse=sparse
            )
            features.append(distances)
        return torch.stack(features, dim=0).float()

    def distances(
        self,
        obs: TDProperties,
        precons: dict[str, PropertyCondition],
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        task_features: list[torch.Tensor] = []
        for state in obs.keys():  # type: ignore
            cnd = precons.get(str(state), None)
            if cnd:
                value = cnd(obs[state])
                value = torch.tensor([value, 0.0]) if pad else torch.tensor([value])
            else:
                nv = -1.0 if sparse else 0.0  # For Identification in filtering
                value = torch.tensor([nv, 1.0]) if pad else torch.tensor([nv])
            task_features.append(value)
        return torch.stack(task_features, dim=0)

    def s_s_attr(
        self,
        current: TDProperties,
        pad: bool = True,
        sparse: bool = False,
    ) -> torch.Tensor:
        dist_matrix = self.skill_state_distances(current, pad, sparse)  # [T, S, 1 or 2]
        # Now safely get edge attributes for (task, state) pairs: [E, 2]
        edge_attr = dist_matrix[
            self.state_skill_full[1], self.state_skill_full[0]
        ]  # [E, 2]
        if sparse:
            # Create mask for valid edges (no -1 in any attribute)
            valid_mask = ~(edge_attr == -1).any(dim=1)
            # Filter edges and attributes
            edge_attr = edge_attr[valid_mask]
        return edge_attr

    def state_skill_attr_weighted_sparse(
        self,
        current: TDProperties,
        pad: bool = True,
    ) -> torch.Tensor:
        dist_matrix = self.skill_state_distances(current, pad)  # [T, S, 2]
        # Now safely get edge attributes for (task, state) pairs: [E, 2]
        edge_attr = dist_matrix[
            self.state_skill_full[1], self.state_skill_full[0]
        ]  # [E, 2]
        return edge_attr

    def _load(self, checkpoint: Any, skills: list[Agent], states: list[Property]):
        self.skills = skills
        self.states = states
        self.load_state_dict(checkpoint["model_state"], strict=False)
