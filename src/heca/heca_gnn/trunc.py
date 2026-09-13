from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from heca.data.entity import Entity
from heca.heca_gnn.modules.identity import IdentityBlock
from heca.heca_gnn.modules.interaction import TransformerBlock
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


class TruncedRows(NamedTuple):
    option: torch.Tensor
    state: torch.Tensor
    memory: torch.Tensor | None


class TruncNetwork(Configurable, nn.Module):
    condenser_names = ("memory",)

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        use_transformer: bool = False
        use_memory: bool = False
        # use_hyperedge: bool = False
        use_effects: bool = False
        use_film: bool = True

    def __init__(self, cfg: Config, name: str):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.name = name
        self.encoder = EncoderBlock(Entity.FEATURE_DIM, cfg.use_effects)
        self.condition_layer = ConditionBlock(Entity.FEATURE_DIM)
        self.translation_layer = TranslationBlock(Entity.FEATURE_DIM)
        self.summary_layer = SummaryBlock(Entity.FEATURE_DIM)
        self.scene_layer = SceneGNNBlock(Entity.FEATURE_DIM)
        self.films = FiLMStack(Entity.FEATURE_DIM, self.condenser_names)

        if cfg.use_transformer:
            self.transformer_layer = TransformerBlock(Entity.FEATURE_DIM)
        else:
            self.transformer_layer = IdentityBlock()

        if cfg.use_memory:
            self.timeline_layer = TimelineMemory(Entity.FEATURE_DIM)
        else:
            self.timeline_layer = IdentityBlock()

    # logits gating
    # hyperedges over the rows of one entity
    # goal rows in the option path (see set_goal_rows)

    @property
    def output_dim(self) -> int:
        if self.cfg.use_film:
            return Entity.FEATURE_DIM
        return Entity.FEATURE_DIM * (1 + len(self.condenser_names))

    def _memory(self, data: HecaData, state: torch.Tensor) -> torch.Tensor | None:
        if not self.cfg.use_memory:
            return None
        return self.timeline_layer(state, data.memory.get(self.name))

    def _condition(
        self, option_x: torch.Tensor, conds: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if not conds:
            return option_x
        if self.cfg.use_film:
            return self.films(option_x, conds)
        return torch.cat(
            [option_x] + [c.expand(option_x.shape[0], -1) for c in conds.values()],
            dim=-1,
        )

    def forward(self, data: HecaData) -> TruncedRows:
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

        option_x = self.transformer_layer(option_x)

        state_x = self.scene_layer(
            option_x,
            x.state,
            data[SceneEdges.type].edge_index,
            data[SceneEdges.type].edge_attr,
        )
        memory = self._memory(data, state_x)

        conds = {} if memory is None else {"memory": memory}
        option_x = self._condition(option_x, conds)

        return TruncedRows(option=option_x, state=state_x, memory=memory)
