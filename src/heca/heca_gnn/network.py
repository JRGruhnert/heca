from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from heca.data.entity import Entity
from heca.graphs.data import HecaData, TrunkMemory
from heca.graphs.roles import ROLE_CURRENT, ROLE_GOAL
from heca.heca_gnn.actor import ActorNetwork
from heca.heca_gnn.critic import CriticNetwork

from heca.heca_gnn.trunc import TruncNetwork
from heca.misc import hardware
from heca.misc.base import Configurable


class NetworkOutput(NamedTuple):
    logits: torch.Tensor  # (1, K) option scores
    value: torch.Tensor  # (1,) state value
    memory: TrunkMemory  # per trunk: the recurrence for the *next* decision


class Network(Configurable, nn.Module):

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        trunc: TruncNetwork.Config = TruncNetwork.Config()
        seperate_trunc: bool = False

    SHARED = "trunc"
    ROLES = ("actor", "critic")

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        self.trunks = nn.ModuleDict(
            {role: TruncNetwork(cfg.trunc, role) for role in self.ROLES}
            if cfg.seperate_trunc
            else {self.SHARED: TruncNetwork(cfg.trunc, self.SHARED)}
        )
        self.actor_net = ActorNetwork(Entity.FEATURE_DIM)
        self.critic_net = CriticNetwork(Entity.FEATURE_DIM)

    def key_for(self, role: str) -> str:
        return role if role in self.trunks else self.SHARED

    def trunk_for(self, role: str) -> nn.Module:
        return self.trunks[self.key_for(role)]

    @property
    def memory_keys(self) -> tuple[str, ...]:
        return tuple(self.trunks)

    @property
    def uses_memory(self) -> bool:
        return self.cfg.trunc.use_memory

    def _trunk_rows(self, data: HecaData) -> dict[str, torch.Tensor]:
        return {name: trunk(data) for name, trunk in self.trunks.items()}

    def goal_pair(self, data: HecaData) -> tuple[torch.Tensor, torch.Tensor]:
        roles = data.entity.role_ids
        cur = (roles == ROLE_CURRENT).nonzero().flatten()
        goal = (roles == ROLE_GOAL).nonzero().flatten()
        return cur, goal

    def actor(self, data: HecaData) -> torch.Tensor:
        return self.actor_net(self.trunk_for("actor")(data).option)

    def critic(self, data: HecaData) -> torch.Tensor:
        rows = self.trunk_for("critic")(data)
        cur, goal = self.goal_pair(data)
        return self.critic_net(data.entity.x, cur, goal, rows.state)

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)

    def evaluate(
        self, data_list: Sequence[HecaData], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logprobs = []
        state_values = []
        entropies = []

        for i, data in enumerate(data_list):
            out = self(data)
            dist = Categorical(logits=out.logits)

            action = actions[i : i + 1]
            logprob = dist.log_prob(action)
            entropy = dist.entropy()

            logprobs.append(logprob)
            state_values.append(out.value)
            entropies.append(entropy)

        return (
            torch.cat(logprobs).to(hardware.device),
            torch.cat(state_values).to(hardware.device),
            torch.cat(entropies).to(hardware.device),
        )

    def forward(self, data: HecaData) -> NetworkOutput:
        rows = {name: trunk(data) for name, trunk in self.trunks.items()}
        actor_rows = rows[self.key_for("actor")]
        critic_rows = rows[self.key_for("critic")]

        cur, goal = self.goal_pair(data)
        logits = self.actor_net(actor_rows.option)
        value = self.critic_net(data.entity.x, cur, goal, critic_rows.state)
        memory = {
            name: row.memory for name, row in rows.items() if row.memory is not None
        }
        return NetworkOutput(logits, value, memory)
