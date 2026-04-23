from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
from hoopgn.misc import hardware
from hoopgn.misc import logger
from hoopgn.networks.mp_final import MPNetwork
from hoopgn.observation.td_properties import TDProperties
from collections import defaultdict

from hoopgn.properties.property import Property


class MPBaseline(MPNetwork):
    @dataclass(kw_only=True)
    class Config(MPNetwork.Config):
        label: str = "baseline"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

        self.combined_feature_dim = self.cfg.dim_encoder * self.dim_property * 2

        h_dim1 = self.combined_feature_dim // 2
        h_dim2 = h_dim1 // 2
        self.actor = nn.Sequential(
            nn.Linear(self.combined_feature_dim, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, self.dim_agent),
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

    def _to_batch(
        self,
        current: list[torch.Tensor],
        goal: list[torch.Tensor],
        obs: list[TDProperties],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.stack(current, dim=0), torch.stack(goal, dim=0)

    def load(self):
        logger.info(f"Loading MP Baseline Network: {self.cfg.label}.")
        assert (
            self.cfg.checkpoint_path is not None
        ), "Checkpoint path must be provided to load the model."
        checkpoint = torch.load(self.cfg.checkpoint_path, map_location=hardware.device)

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
