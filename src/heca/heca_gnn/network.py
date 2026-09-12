from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from heca.graphs.data import HecaData
from heca.heca_gnn.actor import ActorNetwork
from heca.heca_gnn.critic import CriticNetwork

from heca.misc import hardware
from heca.misc.base import Configurable


class Network(Configurable, nn.Module):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        actor: ActorNetwork.Config = ActorNetwork.Config()
        critic: CriticNetwork.Config = CriticNetwork.Config()

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.actor_net = ActorNetwork(cfg.actor)
        self.critic_net = CriticNetwork(cfg.critic)

    def actor(self, data: HecaData) -> torch.Tensor:
        return self.actor_net(data)

    def critic(self, data: HecaData) -> torch.Tensor:
        return self.critic_net(data)

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)

    def evaluate(
        self, data_list: Sequence[HecaData], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logprobs = []
        state_values = []
        entropies = []

        for i, data in enumerate(data_list):
            logits, value = self.forward(data)
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
        logits = self.actor_net(data, carried_memory=carried_memory)
        value = self.critic_net(data)
        return logits, value
