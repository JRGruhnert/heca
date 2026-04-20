from tensordict import TensorDict
from typing import cast

from hoopgn.observation.td_entities import TDEntities
from hoopgn.observation.td_properties import TDProperties
from hoopgn.observation import empty_batchsize


class TDScene(TensorDict):
    def __init__(self, v2: TDEntities, v1: TDProperties):
        data = {
            "entities": v2,
            "properties": v1,
        }
        super().__init__(data, batch_size=empty_batchsize)

    @property
    def entities(self) -> TDEntities:
        return cast(TDEntities, self["entities"])

    @property
    def properties(self) -> TDProperties:
        return cast(TDProperties, self["properties"])
