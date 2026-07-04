from dataclasses import dataclass
import torch
from torch import nn
from torch_geometric.data import HeteroData

from heca.misc.base import Persistable
from heca.heca_gnn.actor import ActorReadoutNetwork
from heca.heca_gnn.critic import CriticReadoutNetwork


class Network(Persistable, nn.Module):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        feature_dim: int = 32

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.linear = nn.ModuleDict(
            {
                "position": nn.Linear(
                    in_features=3,
                    out_features=self.cfg.feature_dim,
                ),
                "rotation": nn.Linear(
                    in_features=3,
                    out_features=self.cfg.feature_dim,
                ),
                "state": nn.Linear(
                    in_features=self.cfg.feature_dim,
                    out_features=self.cfg.feature_dim,
                ),
            }
        )
        self.actor_net = ActorReadoutNetwork(feature_dim=self.cfg.feature_dim)
        self.critic_net = CriticReadoutNetwork(feature_dim=self.cfg.feature_dim)

    def forward(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict
    ) -> torch.Tensor:
        return self.actor(
            HeteroData(x=x, x_dict=x_dict, edge_index_dict=edge_index_dict)
        )

    def actor(self, data: HeteroData) -> torch.Tensor:
        return self.actor_net(data)

    def critic(self, data: HeteroData) -> torch.Tensor:
        return self.critic_net(data)

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)
