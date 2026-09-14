from dataclasses import dataclass
from fnmatch import fnmatch
from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from heca.misc import hardware
from heca.misc.base import Configurable
from heca.graphs.data import HecaData, TrunkMemory
from heca.heca_gnn.actor import ActorNetwork
from heca.heca_gnn.critic import CriticNetwork
from heca.heca_gnn.root import RootNetwork
from heca.heca_gnn.trunc import TruncNetwork, TruncOutput


class NetworkOutput(NamedTuple):
    logits: torch.Tensor
    value: torch.Tensor
    memory: TrunkMemory


class Network(Configurable, nn.Module):

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        feature_dim: int = 256
        seperate_root: bool = False
        seperate_trunc: bool = False
        use_option_transformer: bool = False
        use_condition_gat: bool = False
        use_summary_gcn: bool = False
        use_memory: bool = False
        use_hyperedge: bool = False
        use_option_effects: bool = False

    SHARED = "shared"
    ROLES = ("actor", "critic")

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        if cfg.seperate_root and not cfg.seperate_trunc and cfg.use_memory:
            raise ValueError(
                "seperate_root with a shared trunc and use_memory is ambiguous: the "
                "shared recurrence would be stepped once per root, with different "
                "entity rows, so the memory to carry on is undefined"
            )

        self.roots = nn.ModuleDict(
            {
                role: RootNetwork(
                    role,
                    feature_dim=cfg.feature_dim,
                    condition_gat=cfg.use_condition_gat,
                    hyperedge=cfg.use_hyperedge,
                    effects=cfg.use_option_effects,
                )
                for role in self.ROLES
            }
            if cfg.seperate_root
            else {
                self.SHARED: RootNetwork(
                    self.SHARED,
                    feature_dim=cfg.feature_dim,
                    condition_gat=cfg.use_condition_gat,
                    hyperedge=cfg.use_hyperedge,
                    effects=cfg.use_option_effects,
                )
            }
        )
        self.truncs = nn.ModuleDict(
            {
                role: TruncNetwork(
                    role,
                    feature_dim=cfg.feature_dim,
                    option_transformer=cfg.use_option_transformer,
                    summary_gcn=cfg.use_summary_gcn,
                    memory=cfg.use_memory,
                )
                for role in self.ROLES
            }
            if cfg.seperate_trunc
            else {
                self.SHARED: TruncNetwork(
                    self.SHARED,
                    feature_dim=cfg.feature_dim,
                    option_transformer=cfg.use_option_transformer,
                    summary_gcn=cfg.use_summary_gcn,
                    memory=cfg.use_memory,
                )
            }
        )
        self.actor_net = ActorNetwork(cfg.feature_dim, cfg.use_memory)
        self.critic_net = CriticNetwork(cfg.feature_dim, cfg.use_memory)

    def key_for(self, role: str, segment: str) -> str:
        modules = self.roots if segment == "root" else self.truncs
        return role if role in modules else self.SHARED

    def root_for(self, role: str) -> nn.Module:
        return self.roots[self.key_for(role, "root")]

    def trunc_for(self, role: str) -> nn.Module:
        return self.truncs[self.key_for(role, "trunc")]

    @property
    def a_keys(self) -> tuple[str, str]:
        return self.key_for("actor", "root"), self.key_for("actor", "trunc")

    @property
    def c_keys(self) -> tuple[str, str]:
        return self.key_for("critic", "root"), self.key_for("critic", "trunc")

    @property
    def memory_keys(self) -> tuple[str, ...]:
        return tuple(self.truncs)

    def _rows(self, data: HecaData) -> dict[tuple[str, str], TruncOutput]:
        rows: dict[tuple[str, str], TruncOutput] = {}
        for keys in (self.a_keys, self.c_keys):
            if keys in rows:
                continue
            root_key, trunc_key = keys
            rows[keys] = self.truncs[trunc_key](data, self.roots[root_key](data))
        return rows

    @property
    def layer_names(self) -> dict[str, str]:
        names: dict[str, str] = {}
        for segment, modules in (("root", self.roots), ("trunc", self.truncs)):
            for key, block in modules.items():
                base = segment if key == self.SHARED else f"{segment}.{key}"
                names[base] = f"{segment}s.{key}"
                for layer_name in block.layers:  # type: ignore[attr-defined]
                    names[f"{base}.{layer_name}"] = (
                        f"{segment}s.{key}.layers.{layer_name}"
                    )
        names["actor_head"] = "actor_net"
        names["critic_head"] = "critic_net"
        return names

    def _keys_of(self, logical: Sequence[str]) -> set[str]:
        names = self.layer_names
        modules = tuple(names[name] for name in logical)
        return {
            key
            for key in self.state_dict()
            if any(key == m or key.startswith(m + ".") for m in modules)
        }

    def sync_keys(self, sync: Sequence[str] = ()) -> set[str]:
        if not sync:
            return set(self.state_dict())

        names = self.layer_names
        include = [p for p in sync if not p.startswith("!")]
        exclude = [p[1:] for p in sync if p.startswith("!")]

        def matched(patterns: Sequence[str]) -> list[str]:
            return [name for name in names if any(fnmatch(name, p) for p in patterns)]

        included = matched(include)
        if not included:
            raise ValueError(
                f"sync patterns {list(sync)} matched no layer of {sorted(names)}"
            )
        keys = self._keys_of(included) - self._keys_of(matched(exclude))
        if not keys:
            raise ValueError(
                f"sync selection {list(sync)} owns no tensors; is the selected "
                f"layer disabled in the network config?"
            )
        return keys

    @staticmethod
    def mask_gated(logits: torch.Tensor, data: HecaData) -> torch.Tensor:
        if not data.gating:
            return logits
        gated = data.option.gated.bool()
        if not bool(gated.any()):
            return logits
        if bool(gated.all()):
            raise ValueError("every option is gated off, no option is selectable")
        return logits.masked_fill(gated.unsqueeze(0), float("-inf"))

    def actor_rows(self, data: HecaData) -> TruncOutput:
        return self.trunc_for("actor")(data, self.root_for("actor")(data))

    def critic_rows(self, data: HecaData) -> TruncOutput:
        return self.trunc_for("critic")(data, self.root_for("critic")(data))

    def actor(self, data: HecaData) -> torch.Tensor:
        return self.mask_gated(self.actor_net(self.actor_rows(data)), data)

    def critic(self, data: HecaData) -> torch.Tensor:
        return self.critic_net(self.critic_rows(data))

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)

    def evaluate(
        self, data_list: Sequence[HecaData], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logprobs = []
        state_values = []
        entropies = []

        for i, data in enumerate(data_list):
            out: NetworkOutput = self(data)
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
        rows = self._rows(data)

        actor_rows = rows[self.a_keys]
        logits = self.mask_gated(self.actor_net(actor_rows), data)
        value = self.critic_net(rows[self.c_keys])
        memory = {
            keys[1]: row.memory for keys, row in rows.items() if row.memory is not None
        }
        return NetworkOutput(logits, value, memory)
