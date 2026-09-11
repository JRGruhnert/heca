from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.data import HeteroData
from heca.heca_gnn.modules.condition import ConditionBlock
from heca.heca_gnn.modules.encoder import EntityRowEncoder, OptionEncoder
from heca.heca_gnn.modules.film import FiLMStack, IdentityStack
from heca.heca_gnn.modules.interaction import OptionInteraction
from heca.heca_gnn.modules.readout import OptionReadout
from heca.heca_gnn.modules.summary import SummaryBlock
from heca.heca_gnn.modules.timeline import TimelineMemory
from heca.heca_gnn.modules.translation import TranslationBlock
from heca.misc.base import Configurable


class ActorNetwork(Configurable, nn.Module):
    _last_option_x: torch.Tensor
    _last_mem: torch.Tensor

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        feature_dim: int = 256
        attn_heads: int = 4
        use_option_effects: bool = True
        use_option_interaction: bool = False
        use_timeline_memory: bool = False
        use_film: bool = True

    @property
    def condenser_names(self) -> tuple[str, ...]:
        """Conditioning inputs, in the order they modulate a site."""
        return ("goal", "memory") if self.cfg.use_timeline_memory else ("goal",)

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        self.encoder = EntityRowEncoder(cfg.feature_dim)
        self.option_encoder = OptionEncoder(cfg.feature_dim)
        self.condition_layer = ConditionBlock(cfg.feature_dim)
        self.translation_layer = TranslationBlock(cfg.feature_dim)
        self.summary_layer = SummaryBlock(cfg.feature_dim)

        if cfg.use_option_interaction:
            self.interaction_layer = OptionInteraction(cfg.feature_dim, cfg.attn_heads)
        else:
            self.interaction_layer = None

        if cfg.use_timeline_memory:
            self.timeline_layer = TimelineMemory(cfg.feature_dim)
        else:
            self.timeline_layer = None

        n_cond = len(self.condenser_names)
        self.films = (
            FiLMStack(cfg.feature_dim, self.condenser_names, ("actor",))
            if cfg.use_film
            else IdentityStack()
        )
        # Concatenating the conditioning widens the readout input instead.
        self.option_readout = OptionReadout(
            cfg.feature_dim * (1 + n_cond) if not cfg.use_film else cfg.feature_dim
        )

    def _resolve_memory(
        self,
        data: HeteroData,
        ref: torch.Tensor,
        memory: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.timeline_layer is None:
            self._last_mem = ref.new_zeros(1, self.cfg.feature_dim)
            return None
        if memory is None:
            step = getattr(data, "mem_step", None)
            memory = (
                ref.new_zeros(1, self.cfg.feature_dim)
                if step is None
                else self.timeline_layer(*step)
            )
        self._last_mem = memory
        return memory

    def forward(
        self,
        data: HeteroData,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        comp_x = self.encoder.encode("comp", data)
        entity_x = self.encoder.encode("entity", data)
        stepmix = data[("comp", "condition", "entity")]
        entity_x = self.condition_layer(
            comp_x, entity_x, stepmix.edge_index, stepmix.edge_attr
        )

        tapas_idx = data[("entity", "translation", "entity")].edge_index
        entity_x = self.translation_layer(entity_x, tapas_idx)

        canonical_x = self.encoder.encode("canonical", data)
        h_goal = self.encoder.goal_slot(canonical_x, data)

        effects = data["option"].x
        if not self.cfg.use_option_effects:
            effects = torch.zeros_like(effects)
        option_x = self.option_encoder(effects)
        option_x = self.summary_layer(
            entity_x, option_x, data[("entity", "summary", "option")].edge_index
        )
        if self.interaction_layer is not None:
            option_x = self.interaction_layer(option_x)

        self._last_option_x = option_x

        memory = self._resolve_memory(data, option_x, memory)
        conds = {"goal": h_goal}
        if memory is not None:
            conds["memory"] = memory

        if not self.cfg.use_film:
            option_x = torch.cat(
                [option_x]
                + [conds[name].expand(option_x.shape[0], -1) for name in conds],
                dim=-1,
            )

        return self.option_readout(option_x, self.films, conds)
