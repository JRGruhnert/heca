from tensordict import TensorDict
from typing import cast

from hoopgn.observation.td_entities import TDEntities
from hoopgn.observation.td_properties import TDProperties
from hoopgn.observation import empty_batchsize


class TDScene(TensorDict):
    def __init__(
        self,
        v1: TDProperties,
        v2: TDEntities,
        leaf: TensorDict,
    ):
        data = {
            "v1": v1,
            "v2": v2,
            "leaf": leaf,
        }
        super().__init__(data, batch_size=empty_batchsize)

    @property
    def v1(self) -> TDProperties:
        return cast(TDProperties, self["v1"])

    @property
    def v2(self) -> TDEntities:
        return cast(TDEntities, self["v2"])

    @property
    def leaf(self) -> TensorDict:
        return cast(TensorDict, self["leaf"])
