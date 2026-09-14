import torch
from torch import nn

from heca.heca_gnn.modules.film import FiLM, Identity
from heca.heca_gnn.trunc import TruncOutput


class ActorNetwork(nn.Module):

    def __init__(self, feature_dim: int, use_memory: bool):
        nn.Module.__init__(self)

        if use_memory:
            self.film = FiLM(feature_dim)
        else:
            self.film = Identity()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, data: TruncOutput) -> torch.Tensor:
        option_x = self.film(data.option, data.memory)
        actor_out = self.mlp(option_x)
        return actor_out.view(1, -1)
