from dataclasses import dataclass
import torch
from torch import nn
from torch_geometric.data import HeteroData

from heca.misc.base import Configurable
from heca.heca_gnn.actor import ActorReadoutNetwork
from heca.heca_gnn.critic import CriticReadoutNetwork


class Network(Configurable, nn.Module):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
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

    def forwarddd(self, raw_observations):
        # raw_observations: [N, 7] (pos+rot) + state as int separately
        # StepMix expects: [pos, rot, state_onehot] or [pos, rot, state_int]

        # 1. Get probabilities
        probs = self.stepmix.predict_proba(raw_observations)  # [N, n_comp]

        # 2. Get log-likelihood
        log_liks = self.stepmix.score_samples(raw_observations)  # [N]

        # 3. Get hard assignments (for one-hot)
        assignments = self.stepmix.predict(raw_observations)  # [N]
        assignment_onehot = F.one_hot(assignments, num_classes=n_comp)  # [N, n_comp]

        # 4. Concatenate everything
        features = torch.cat(
            [
                raw_observations[:, :7],  # pos + rot (keep raw!)
                probs,  # soft memberships
                log_liks.unsqueeze(-1),  # confidence
                assignment_onehot,  # hard assignment
            ],
            dim=-1,
        )

        return features  # [N, 7 + 3*n_comp + 1]

    def forward(self, data: HeteroData) -> torch.Tensor:
        return self.actor(data)

    def actor(self, data: HeteroData) -> torch.Tensor:
        return self.actor_net(data)

    def critic(self, data: HeteroData) -> torch.Tensor:
        return self.critic_net(data)

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)
