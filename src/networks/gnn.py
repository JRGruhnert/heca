from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Union
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GINConv, GINEConv
from src.observation.observation import StateValueDict
from src.skills.skill import Skill
from src.networks.layers.mlp import GinStandardMLP, UnactivatedMLP
from src.networks.network import Network, NetworkConfig
from src.states.state import State
from torch_geometric.data import HeteroData
from torch_geometric.explain import (
    Explainer,
    CaptumExplainer,
    Explanation,
    HeteroExplanation,
)


class ReadoutNetwork(nn.Module):
    def __init__(
        self,
        dim_features: int,
        dim_skill: int,
        dim_state: int,
    ):
        super().__init__()
        self.dim_skills = dim_skill
        self.dim_state = dim_state
        self.dim_features = dim_features

        self.state_state_gin = GINConv(
            nn=GinStandardMLP(
                in_dim=self.dim_features,
                out_dim=self.dim_features,
                hidden_dim=self.dim_features,
            ),
            # edge_dim=1,
        )

        self.state_skill_gin = GINEConv(
            nn=GinStandardMLP(
                in_dim=self.dim_features,
                out_dim=self.dim_features,
                hidden_dim=self.dim_features,
            ),
            edge_dim=2,
        )
        self.readout_net = GINConv(
            nn=UnactivatedMLP(self.dim_features, 1),
        )

    @abstractmethod
    def readout(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict
    ) -> torch.Tensor:
        raise NotImplementedError("Readout method must be implemented by subclass.")

    def forward(self, batch: Batch) -> torch.Tensor:
        x_dict = batch.x_dict  # type: ignore
        edge_index_dict = batch.edge_index_dict  # type: ignore
        edge_attr_dict = batch.edge_attr_dict  # type: ignore

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
        logits = self.readout_net(
            x=(x, x_dict["actor"]),
            edge_index=edge_index_dict[("task", "task-actor", "actor")],
        )
        return logits.view(-1, self.dim_skills)


class CriticReadoutNetwork(ReadoutNetwork):
    def readout(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict
    ) -> torch.Tensor:
        value = self.readout_net(
            x=(x, x_dict["critic"]),
            edge_index=edge_index_dict[("task", "task-critic", "critic")],
        )
        return value.squeeze(-1)


@dataclass
class GraphNetworkConfig(NetworkConfig):
    # Network cant be trained in explain mode, only used for generating explanations with a trained model.
    explain_mode: bool = False
    name: str = "gnn"


class GraphNetwork(Network):

    def __init__(
        self,
        config: GraphNetworkConfig,
        states: list[State],
        skills: list[Skill],
    ):
        super().__init__(config, states, skills)  # type: ignore
        self.config = config

        self.actor = ActorReadoutNetwork(
            dim_features=self.dim_encoder,
            dim_state=self.dim_states,
            dim_skill=self.dim_skills,
        )
        self.critic = CriticReadoutNetwork(
            dim_features=self.dim_encoder,
            dim_state=self.dim_states,
            dim_skill=self.dim_skills,
        )
        self.indices = torch.arange(self.dim_skills)
        self.actor_explainer = Explainer(
            self.actor,  # It is assumed that model outputs a single tensor.
            algorithm=CaptumExplainer("IntegratedGradients"),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="probs",  # Model returns probabilities.
            ),
        )

        self.critic_explainer = Explainer(
            self.critic,  # It is assumed that model outputs a single tensor.
            algorithm=CaptumExplainer("IntegratedGradients"),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="probs",  # Model returns probabilities.
            ),
        )

    def forward(
        self,
        batch: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(batch), self.critic(batch)  # Logits, Value

    def explain(self, batch: Batch) -> tuple[HeteroExplanation, HeteroExplanation]:
        actor_explanation = self.actor_explainer(
            batch.get_example(0).x_dict,
            batch.get_example(0).edge_index_dict,
            index=self.indices,
        )
        critic_explanation = self.critic_explainer(
            batch.get_example(0).x_dict,
            batch.get_example(0).edge_index_dict,
            index=self.indices,
        )
        assert isinstance(actor_explanation, HeteroExplanation)
        assert isinstance(critic_explanation, HeteroExplanation)
        return actor_explanation, critic_explanation

    def _to_batch(
        self, current: list[Tensor], goal: list[Tensor], obs: list[StateValueDict]
    ) -> Batch:
        batch_data = []
        for c, g, o in zip(current, goal, obs):
            data = HeteroData()
            data["goal"].x = g
            data["obs"].x = c
            data["task"].x = torch.zeros(self.dim_skills, self.dim_encoder)
            data["actor"].x = torch.zeros(self.dim_skills, 1)
            data["critic"].x = torch.zeros(1, 1)

            data[("goal", "goal-obs", "obs")].edge_index = self.state_state_sparse
            data[("obs", "obs-task", "task")].edge_index = self.state_skill_full
            # data[("goal", "goal-obs", "obs")].edge_attr = self.cnv.state_state_attr
            data[("obs", "obs-task", "task")].edge_attr = self.s_s_attr(o)
            data[("task", "task-actor", "actor")].edge_index = self.skill_skill_sparse
            data[("task", "task-critic", "critic")].edge_index = self.skill_to_single
            batch_data.append(data)
        return Batch.from_data_list(batch_data)

    def skill_state_distances(
        self,
        obs: StateValueDict,
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        for skill in self.skills:
            distances = skill.distances(obs, self.states, pad, sparse)
            features.append(distances)
        return torch.stack(features, dim=0).float()

    @cached_property
    def state_state_full(self) -> torch.Tensor:
        src = (
            torch.arange(self.dim_states)
            .unsqueeze(1)
            .repeat(1, self.dim_states)
            .flatten()
        )
        dst = torch.arange(self.dim_states).repeat(self.dim_states)
        return torch.stack([src, dst], dim=0)

    @cached_property
    def state_skill_full(self) -> torch.Tensor:
        src = (
            torch.arange(self.dim_states)
            .unsqueeze(1)
            .repeat(1, self.dim_skills)
            .flatten()
        )
        dst = torch.arange(self.dim_skills).repeat(self.dim_states)
        return torch.stack([src, dst], dim=0)

    @cached_property
    def state_state_sparse(self) -> torch.Tensor:
        indices = torch.arange(self.dim_states)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def skill_skill_sparse(self) -> torch.Tensor:
        indices = torch.arange(self.dim_skills)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def state_skill_sparse(self) -> torch.Tensor:
        edge_list = []
        for task_idx, skill in enumerate(self.skills):
            for state_idx, state in enumerate(self.states):
                if state.name in skill.precons.keys():
                    edge_list.append((state_idx, task_idx))
        return torch.tensor(edge_list, dtype=torch.long).t()

    @cached_property
    def skill_to_single(self) -> torch.Tensor:
        indices = torch.arange(self.dim_skills)
        return torch.stack([indices, torch.zeros_like(indices)], dim=0)

    @cached_property
    def single_to_skill(self) -> torch.Tensor:
        indices = torch.arange(self.dim_skills)
        return torch.stack([torch.zeros_like(indices), indices], dim=0)

    @cached_property
    def state_to_single(self) -> torch.Tensor:
        indices = torch.arange(self.dim_states)
        return torch.stack([indices, torch.zeros_like(indices)], dim=0)

    @cached_property
    def single_to_state(self) -> torch.Tensor:
        indices = torch.arange(self.dim_states)
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
            self.state_skill_sparse[0] * self.dim_skills + self.state_skill_sparse[1]
        )
        full = self.state_skill_full[0] * self.dim_skills + self.state_skill_full[1]
        return torch.isin(full, sparse).float().unsqueeze(-1)

    def s_s_attr(
        self,
        current: StateValueDict,
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
        current: StateValueDict,
        pad: bool = True,
    ) -> torch.Tensor:
        dist_matrix = self.skill_state_distances(current, pad)  # [T, S, 2]
        # Now safely get edge attributes for (task, state) pairs: [E, 2]
        edge_attr = dist_matrix[
            self.state_skill_full[1], self.state_skill_full[0]
        ]  # [E, 2]
        return edge_attr

    def _load(self, checkpoint: Any):
        self.policy_old.load_state_dict(checkpoint["model_state"], strict=False)
        self.policy_new.load_state_dict(checkpoint["model_state"], strict=False)
