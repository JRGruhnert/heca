import torch
from torch import nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GINConv, GINEConv
from torch_geometric.explain import (
    Explainer,
    CaptumExplainer,
    Explanation,
    HeteroExplanation,
)
from src.observation.observation import StateValueDict
from src.networks.actor_critic import GnnBase, PPOType
from src.networks.layers.mlp import GinStandardMLP, UnactivatedMLP
from src.hardware import device


class GinReadoutNetwork(nn.Module):
    def __init__(
        self,
        dim_features: int,
        dim_skill: int,
        dim_state: int,
        ppo_type: PPOType,
    ):
        super().__init__()
        self.ppo_type = ppo_type
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

        self.actor_gin = GINConv(
            nn=UnactivatedMLP(self.dim_features, 1),
        )

        self.critic_gin = GINConv(
            nn=UnactivatedMLP(self.dim_features, 1),
        )

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

        if self.ppo_type is PPOType.ACTOR:

            logits = self.actor_gin(
                x=(x2, x_dict["actor"]),
                edge_index=edge_index_dict[("task", "task-actor", "actor")],
            )
            return logits.view(-1, self.dim_skills)
        else:
            value = self.critic_gin(
                x=(x2, x_dict["critic"]),
                edge_index=edge_index_dict[("task", "task-critic", "critic")],
            )
            return value.squeeze(-1)


class _ExplainerWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._edge_attr_dict: dict = {}

    def set_edge_attrs(self, edge_attr_dict: dict):
        self._edge_attr_dict = edge_attr_dict

    def forward(self, x_dict: dict, edge_index_dict: dict, index: int):
        data = HeteroData()
        for key, x in x_dict.items():
            data[key].x = x
        for key, edge_index in edge_index_dict.items():
            data[key].edge_index = edge_index
        for key, edge_attr in self._edge_attr_dict.items():
            data[key].edge_attr = edge_attr
        batch = Batch.from_data_list([data])
        logits = self.model(batch)
        # critic returns a 1-D value tensor; actor returns [1, dim_skills]
        if logits.dim() == 1:
            return logits
        return logits[:, index]


class Gnn(GnnBase):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.actor = GinReadoutNetwork(
            dim_features=self.dim_encoder,
            dim_state=self.dim_states,
            dim_skill=self.dim_skills,
            ppo_type=PPOType.ACTOR,
        )
        self.critic = GinReadoutNetwork(
            dim_features=self.dim_encoder,
            dim_state=self.dim_states,
            dim_skill=self.dim_skills,
            ppo_type=PPOType.CRITIC,
        )

        self.actor_explainer = Explainer(
            _ExplainerWrapper(self.actor),
            algorithm=CaptumExplainer("IntegratedGradients"),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",  # Model returns raw logits.
            ),
        )

        self.critic_explainer = Explainer(
            _ExplainerWrapper(self.critic),
            algorithm=CaptumExplainer("IntegratedGradients"),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="regression",  # Critic estimates a scalar value, not a class.
                task_level="node",
                return_type="raw",  # Model returns a raw scalar.
            ),
        )

    def explain(
        self,
        obs: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Explanation | HeteroExplanation,
        Explanation | HeteroExplanation,
    ]:
        # Resolve the action first (same logic as act())
        action, logprob, value = self.act(obs, goal)

        # Build the graph for the current observation
        data = self.to_data(obs, goal)

        # Edge attributes are not perturbed by the explainer; store them in the wrappers
        self.actor_explainer.model.set_edge_attrs(data.edge_attr_dict)
        self.critic_explainer.model.set_edge_attrs(data.edge_attr_dict)

        actor_explanation = self.actor_explainer(
            data.x_dict,
            data.edge_index_dict,
            index=int(action.item()),  # explain the chosen skill
        )
        critic_explanation = self.critic_explainer(
            data.x_dict,
            data.edge_index_dict,
            index=0,  # critic has a single scalar output
        )
        return action, logprob, value, actor_explanation, critic_explanation

    def forward(
        self,
        obs: list[StateValueDict],
        goal: list[StateValueDict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch: Batch = self.to_batch(obs, goal)
        logits = self.actor(batch)
        value = self.critic(batch)
        return logits, value

    def to_data(self, obs: StateValueDict, goal: StateValueDict) -> HeteroData:
        obs_tensor = self.encode_states(obs)
        goal_tensor = self.encode_states(goal)
        data = HeteroData()
        data["goal"].x = goal_tensor
        data["obs"].x = obs_tensor
        data["task"].x = torch.zeros(self.dim_skills, self.dim_encoder)
        data["actor"].x = torch.zeros(self.dim_skills, 1)
        data["critic"].x = torch.zeros(1, 1)

        data[("goal", "goal-obs", "obs")].edge_index = self.state_state_sparse
        data[("obs", "obs-task", "task")].edge_index = self.state_skill_full
        # data[("goal", "goal-obs", "obs")].edge_attr = self.cnv.state_state_attr
        data[("obs", "obs-task", "task")].edge_attr = self.state_skill_attr_weighted(
            obs
        )
        data[("task", "task-actor", "actor")].edge_index = self.skill_skill_sparse
        data[("task", "task-critic", "critic")].edge_index = self.skill_to_single
        return data.to(device)  # type: ignore
