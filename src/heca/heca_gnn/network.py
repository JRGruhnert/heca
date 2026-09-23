from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Mapping, NamedTuple, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from heca.misc import hardware, logger
from heca.misc.base import Configurable
from heca.graphs.data import HecaData, TrunkMemory
from heca.graphs.edges.edge_set import DEFAULT_TERMS
from heca.heca_gnn.actor import ActorNetwork
from heca.heca_gnn.critic import CriticNetwork
from heca.heca_gnn.modules.encoders.encoder import EncodedRows
from heca.heca_gnn.root import RootNetwork
from heca.heca_gnn.trunc import TruncNetwork, TruncOutput


class NetworkOutput(NamedTuple):
    logits: torch.Tensor
    value: torch.Tensor
    memory: TrunkMemory


GOAL_CONDITIONING: tuple[str, ...] = ("hyperedge", "residual")


class Network(Configurable, nn.Module):

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        feature_dim: int = 128
        seperate_root: bool = False
        seperate_trunc: bool = False
        use_option_transformer: bool = False
        use_condition_gat: bool = False
        use_summary_gcn: bool = False
        use_memory: bool = False
        goal_conditioning: str = "residual"  # "hyperedge" | "residual"
        use_statistics: bool = False
        pair_norm: bool = False
        use_rotation: bool = False
        position_jitter: float = 1.0  # values around
        jitter_scope: str = "none"  # "none" | "entity" | "scene" | "both"
        edge_terms: tuple[str, ...] = DEFAULT_TERMS
        gating: bool = True
        sync: tuple[str, ...] = ()

    SHARED = "shared"
    ROLES = ("actor", "critic")

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        # one mutually exclusive channel, so a run cannot accidentally carry both
        if cfg.goal_conditioning not in GOAL_CONDITIONING:
            raise ValueError(
                f"unknown goal_conditioning {cfg.goal_conditioning!r}; "
                f"known: {', '.join(GOAL_CONDITIONING)}"
            )
        hyperedge = cfg.goal_conditioning == "hyperedge"
        goal_residual = cfg.goal_conditioning == "residual"

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
                    hyperedge=hyperedge,
                    statistics=cfg.use_statistics,
                    pair_norm=cfg.pair_norm,
                    rotation=cfg.use_rotation,
                    edge_terms=cfg.edge_terms,
                    goal_residual=goal_residual,
                )
                for role in self.ROLES
            }
            if cfg.seperate_root
            else {
                self.SHARED: RootNetwork(
                    self.SHARED,
                    feature_dim=cfg.feature_dim,
                    condition_gat=cfg.use_condition_gat,
                    hyperedge=hyperedge,
                    statistics=cfg.use_statistics,
                    pair_norm=cfg.pair_norm,
                    rotation=cfg.use_rotation,
                    edge_terms=cfg.edge_terms,
                    goal_residual=goal_residual,
                )
            }
        )
        self.truncs = nn.ModuleDict(
            {
                role: TruncNetwork(
                    role,
                    feature_dim=cfg.feature_dim,
                    option_transformer=cfg.use_option_transformer,
                    summary_sage=cfg.use_summary_gcn,
                    pair_norm=cfg.pair_norm,
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
                    summary_sage=cfg.use_summary_gcn,
                    pair_norm=cfg.pair_norm,
                    memory=cfg.use_memory,
                )
            }
        )
        self.actor_net = ActorNetwork(cfg.feature_dim, cfg.use_memory)
        self.critic_net = CriticNetwork(cfg.feature_dim, cfg.use_memory)

        self.to(hardware.device)

    def key_for(self, role: str, segment: str) -> str:
        if segment == "root":
            modules = self.roots
        elif segment == "trunc":
            modules = self.truncs
        else:
            raise ValueError(f"unknown segment {segment!r}; expected 'root' or 'trunc'")
        if role not in (*self.ROLES, self.SHARED):
            raise ValueError(
                f"unknown role {role!r}; expected one of {[*self.ROLES, self.SHARED]}"
            )
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

    @property
    def uses_memory(self) -> bool:
        return self.cfg.use_memory

    def _rows(self, data: HecaData) -> dict[tuple[str, str], TruncOutput]:
        rows: dict[tuple[str, str], TruncOutput] = {}
        encoded: dict[str, EncodedRows] = {}
        for keys in (self.a_keys, self.c_keys):
            if keys in rows:
                continue
            root_key, trunc_key = keys
            if root_key not in encoded:
                # both roles can share one root: encode it once, not once per role
                encoded[root_key] = self.roots[root_key](data)
            rows[keys] = self.truncs[trunc_key](data, encoded[root_key])
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
        if not include:
            # only exclusions given: federate everything except them
            include = ["*"]

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

    def mask_gated(self, logits: torch.Tensor, data: HecaData) -> torch.Tensor:
        if not self.cfg.gating:
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

    def upgrade(self, checkpoint: Mapping[str, torch.Tensor]):
        current = set(self.state_dict())
        given = set(checkpoint)
        shared = current & given
        if not shared:
            raise ValueError(
                f"checkpoint shares no parameter name with this network "
                f"({len(given)} tensors in the checkpoint, {len(current)} in the "
                f"network, e.g. {sorted(given)[:2]} vs {sorted(current)[:2]}); it "
                "was probably written with different layer names, so remap it "
                "before loading"
            )
        result = self.load_state_dict(checkpoint, strict=False)
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                f"upgrade: restored {len(shared)}/{len(current)} tensors "
                f"({len(result.missing_keys)} missing, "
                f"{len(result.unexpected_keys)} unexpected)"
            )
        return list(result.missing_keys), list(result.unexpected_keys)

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
