from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from heca.data.entity import Entity
from heca.misc.base import Configurable
from heca.graphs.data import HecaData
from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.scene_edges import SceneEdges
from heca.graphs.edges.summary_edges import SummaryEdges
from heca.graphs.edges.translation_edges import TranslationEdges
from heca.heca_gnn.modules.condition import ConditionBlock
from heca.heca_gnn.modules.encoders.encoder import EncodedRows, EncoderBlock
from heca.heca_gnn.modules.film import FiLMStack
from heca.heca_gnn.modules.overview import SceneGNNBlock
from heca.heca_gnn.modules.summary import SummaryBlock
from heca.heca_gnn.modules.timeline import TimelineMemory
from heca.heca_gnn.modules.translation import TranslationBlock
from heca.heca_gnn.modules.interaction import OptionInteraction


class TruncedRows(NamedTuple):
    option: torch.Tensor
    state: torch.Tensor


class TruncNetwork(Configurable, nn.Module):
    _last_option_x: torch.Tensor
    _last_mem: torch.Tensor

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        use_interaction: bool = False
        use_timeline_memory: bool = False
        use_hyperedge: bool = False
        use_summary_attr: bool = False
        use_effects: bool = False
        use_film: bool = True

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        self.encoder = EncoderBlock(Entity.FEATURE_DIM, cfg.use_effects)
        self.condition_layer = ConditionBlock(Entity.FEATURE_DIM)
        self.translation_layer = TranslationBlock(Entity.FEATURE_DIM)
        self.summary_layer = SummaryBlock(Entity.FEATURE_DIM)
        self.scene_layer = SceneGNNBlock(Entity.FEATURE_DIM)

        if cfg.use_interaction:
            self.interaction_layer = OptionInteraction(Entity.FEATURE_DIM)
        else:
            self.interaction_layer = None

        if cfg.use_timeline_memory:
            self.timeline_layer = TimelineMemory(Entity.FEATURE_DIM)
        else:
            self.timeline_layer = None

        self.films = FiLMStack(Entity.FEATURE_DIM, self.condenser_names)

    # logits gating
    # hyperedges with canonical
    # goal node with canonical
    # film vs concat
    # memory fix
    #
    def _resolve_memory(
        self,
        data: HecaData,
        ref: torch.Tensor,
        carried: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.timeline_layer is None:
            self._last_mem = ref.new_zeros(1, Entity.FEATURE_DIM)
            return None
        if carried is None:
            step = data.mem_step
            carried = (
                ref.new_zeros(1, Entity.FEATURE_DIM)
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
            data[ConditionEdges.type].edge_index,
            data[ConditionEdges.type].edge_attr,
        )

        entity_x = self.translation_layer(
            entity_x,
            data[TranslationEdges.type].edge_index,
        )

        option_x = self.summary_layer(
            entity_x,
            x.option,
            data[SummaryEdges.type].edge_index,
        )

        state_x = self.scene_layer(
            option_x,
            data[SceneEdges.type].x,
            data[SceneEdges.type].edge_index,
            data[SceneEdges.type].edge_attr,
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
