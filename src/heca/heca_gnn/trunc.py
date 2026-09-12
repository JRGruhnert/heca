from dataclasses import dataclass

import torch
from torch import nn
from heca.graphs.data import HecaData
from heca.heca_gnn.modules.condition import ConditionBlock
from heca.heca_gnn.modules.encoders.encoder import EncodedRows, EncoderBlock
from heca.heca_gnn.modules.film import FiLMStack
from heca.heca_gnn.modules.interaction import OptionInteraction
from heca.heca_gnn.modules.overview import OptionSummaryBlock
from heca.heca_gnn.modules.summary import TPSummaryBlock
from heca.heca_gnn.modules.timeline import TimelineMemory
from heca.heca_gnn.modules.translation import TranslationBlock
from heca.misc.base import Configurable


class TruncNetwork(Configurable, nn.Module):
    _last_option_x: torch.Tensor
    _last_mem: torch.Tensor

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        feature_dim: int = 256
        use_option_interaction: bool = False
        use_timeline_memory: bool = False
        use_hyperedge: bool = False
        use_effects: bool = False
        use_film: bool = True

    @property
    def condenser_names(self) -> tuple[str, ...]:
        """Conditioning inputs, in the order they modulate a site."""
        return ("goal", "memory") if self.cfg.use_timeline_memory else ("goal",)

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        self.encoder = EncoderBlock(cfg.feature_dim, cfg.use_effects)
        self.condition_layer = ConditionBlock(cfg.feature_dim)
        self.translation_layer = TranslationBlock(cfg.feature_dim)
        self.tp_summary_layer = TPSummaryBlock(cfg.feature_dim)
        self.option_summary_layer = OptionSummaryBlock(cfg.feature_dim)

        if cfg.use_option_interaction:
            self.interaction_layer = OptionInteraction(cfg.feature_dim)
        else:
            self.interaction_layer = None

        if cfg.use_timeline_memory:
            self.timeline_layer = TimelineMemory(cfg.feature_dim)
        else:
            self.timeline_layer = None

        self.films = FiLMStack(cfg.feature_dim, self.condenser_names)

    def _resolve_memory(
        self,
        data: HecaData,
        ref: torch.Tensor,
        carried: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.timeline_layer is None:
            self._last_mem = ref.new_zeros(1, self.cfg.feature_dim)
            return None
        if carried is None:
            step = data.mem_step
            carried = (
                ref.new_zeros(1, self.cfg.feature_dim)
                if step is None
                else self.timeline_layer(*step)
            )
        self._last_mem = carried  # type: ignore
        return carried

    def forward(
        self,
        data: HecaData,
        carried_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x: EncodedRows = self.encoder(data)

        entity_x = self.condition_layer(
            x.comp,
            x.entity,
            data[("comp", "condition", "entity")].edge_index,
            data[("comp", "condition", "entity")].edge_attr,
        )

        entity_x = self.translation_layer(
            entity_x,
            data[("entity", "translation", "entity")].edge_index,
        )

        option_x = self.tp_summary_layer(
            entity_x,
            x.option,
            data[("entity", "summary", "option")].edge_index,
        )
        if self.interaction_layer is not None:
            option_x = self.interaction_layer(option_x)

        self._last_option_x = option_x

        memory = self._resolve_memory(data, option_x, carried_memory)
        conds = {}
        if memory is not None:
            conds["memory"] = memory

        if not self.cfg.use_film:
            option_x = torch.cat(
                [option_x]
                + [conds[name].expand(option_x.shape[0], -1) for name in conds],
                dim=-1,
            )

        return self.option_readout(option_x, self.films, conds)
