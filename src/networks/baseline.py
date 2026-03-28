from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
from src.networks.network import Network, NetworkConfig
from src.observation.observation import StateValueDict
from collections import defaultdict

from src.states.state import State


@dataclass
class BaselineNetworkConfig(NetworkConfig):
    name: str = "baseline"


class BaselineNetwork(Network):

    def __init__(
        self,
        config: BaselineNetworkConfig,
    ):
        super().__init__(config)

        self.combined_feature_dim = (
            self.config.dim_encoder * self.config.state_count * 2
        )

        h_dim1 = self.combined_feature_dim // 2
        h_dim2 = h_dim1 // 2
        self.actor = nn.Sequential(
            nn.Linear(self.combined_feature_dim, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, self.config.skill_count),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.combined_feature_dim, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, 1),
        )

    def forward(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interleaved = torch.stack([current, goal], dim=2)  # [B, S, 2, D]
        x = interleaved.flatten(start_dim=1)  # [B, S * 2 * D]
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)  # shape: [B]
        return logits, value

    def state_type_dict_values(
        self,
        x: StateValueDict,
        states: list[State],
    ) -> dict[str, torch.Tensor]:
        """Group state values by their type strings."""
        grouped = defaultdict(list)
        for state in states:
            value = state.make_input(x[state.name])
            grouped[state.type].append(value)
        return {k: torch.stack(v).float() for k, v in grouped.items()}

    def _to_batch(
        self,
        current: list[torch.Tensor],
        goal: list[torch.Tensor],
        obs: list[StateValueDict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.stack(current, dim=0), torch.stack(goal, dim=0)

    def _load(self, checkpoint: Any):
        old_state_dict: dict[str, torch.Tensor] = (
            checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        )

        # Get current model state dict
        new_state_dict = self.state_dict()

        # Copy compatible weights
        for name, old_param in old_state_dict.items():
            if name in new_state_dict:
                new_param = new_state_dict[name]

                # Check if dimensions match
                if old_param.shape == new_param.shape:
                    # Direct copy
                    new_state_dict[name] = old_param
                elif len(old_param.shape) == len(new_param.shape):
                    # Same number of dimensions, try partial copy
                    new_state_dict[name] = self._expand_tensor_dims(
                        old_param, new_param.shape
                    )
                else:
                    print(
                        f"Skipping {name}: incompatible shapes {old_param.shape} -> {new_param.shape}"
                    )

        # Load the modified state dict
        self.load_state_dict(new_state_dict, strict=False)

    def _expand_tensor_dims(self, old_tensor, target_shape):
        """Expand tensor dimensions by copying/padding"""
        old_shape = old_tensor.shape
        new_tensor = torch.zeros(target_shape, dtype=old_tensor.dtype)

        # Copy the overlapping dimensions
        slices = []
        for i, (old_dim, new_dim) in enumerate(zip(old_shape, target_shape)):
            if old_dim <= new_dim:
                slices.append(slice(0, old_dim))
            else:
                slices.append(slice(0, new_dim))

        new_tensor[tuple(slices)] = old_tensor[tuple(slices)]
        return new_tensor

    def explain(self, batch) -> Any:
        raise NotImplementedError("This network does not support explanations.")
