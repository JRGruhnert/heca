from dataclasses import dataclass

import torch
from torch import nn

from heca.graphs.data import HecaData
from heca.heca_gnn.modules.encoders.entity_encoder import EntityRowEncoder
from heca.heca_gnn.modules.film import FiLMStack, IdentityStack

from heca.heca_gnn.modules.state_critic import StateCritic

from heca.misc.base import Configurable


class CriticNetwork(Configurable, nn.Module):

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        feature_dim: int = 256
        use_budget: bool = True
        use_film: bool = True

    @property
    def condenser_names(self) -> tuple[str, ...]:
        """Conditioning inputs, in the order they modulate a site."""
        return ("goal",)

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.encoder = EntityRowEncoder(cfg.feature_dim)
        self.films = (
            FiLMStack(cfg.feature_dim, self.condenser_names)
            if cfg.use_film
            else IdentityStack()
        )
        self.state_critic = StateCritic(cfg.feature_dim, use_budget=cfg.use_budget)

    def forward(self, data: HecaData) -> torch.Tensor:
        canonical_x = self.encoder.encode("canonical", data)
        conds = {"goal": self.encoder.goal_slot(canonical_x, data)}

        return self.state_critic(
            canonical_x,
            data.canonical.cur_idx,
            data.canonical.goal_idx,
            self.films,
            conds,
            budget=data.state.budget,
        )
