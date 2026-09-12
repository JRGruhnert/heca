from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from heca.graphs.data import HecaData
from heca.heca_gnn.actor import ActorNetwork
from heca.heca_gnn.critic import CriticNetwork

from heca.heca_gnn.trunc import TruncNetwork
from heca.misc import hardware
from heca.misc.base import Configurable


class Network(Configurable, nn.Module):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        trunc: TruncNetwork.Config = TruncNetwork.Config()
        actor: ActorNetwork.Config = ActorNetwork.Config()
        critic: CriticNetwork.Config = CriticNetwork.Config()
        seperate_trunc: bool = False

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg
        if cfg.seperate_trunc:
            self.actor_trunc = TruncNetwork(cfg.trunc)
            self.critic_trunc = TruncNetwork(cfg.trunc)
        else:
            self.trunc = TruncNetwork(cfg.trunc)

        self.actor_net = ActorNetwork(cfg.actor)
        self.critic_net = CriticNetwork(cfg.critic)

    def actor(self, data: HecaData) -> torch.Tensor:
        if self.cfg.seperate_trunc:
            x = self.actor_trunc(data)
        else:
            x = self.trunc(data)
        return self.actor_net(x)

    def critic(self, data: HecaData) -> torch.Tensor:
        if self.cfg.seperate_trunc:
            x = self.critic_trunc(data)
        else:
            x = self.trunc(data)
        return self.critic_net(x)

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)

    def evaluate(
        self, data_list: Sequence[HecaData], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logprobs = []
        state_values = []
        entropies = []

        for i, data in enumerate(data_list):
            logits, value = self(data)
            dist = Categorical(logits=logits)

            action = actions[i : i + 1]
            logprob = dist.log_prob(action)
            entropy = dist.entropy()

            logprobs.append(logprob)
            state_values.append(value)
            entropies.append(entropy)

        return (
            torch.cat(logprobs).to(hardware.device),
            torch.cat(state_values).to(hardware.device),
            torch.cat(entropies).to(hardware.device),
        )

    def forward(
        self,
        data: HecaData,
        carried_memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.seperate_trunc:
            xa = self.actor_trunc(data)
            xc = self.critic_trunc(data)
            logits = self.actor_net(xa, carried_memory=carried_memory)
            value = self.critic_net(xc)
        else:
            x = self.trunc(data)
            logits = self.actor_net(x, carried_memory=carried_memory)
            value = self.critic_net(x)
        return logits, value
