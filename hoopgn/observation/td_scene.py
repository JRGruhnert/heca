from tensordict import TensorDict
from typing import cast

from hoopgn.observation.td_entities import TDEntities
from hoopgn.observation.td_parameters import TDParameters
from hoopgn.observation import empty_batchsize


class TDScene(TensorDict):
    def __init__(self, entities, parameters: TDParameters):
        data = {
            "entities": entities,
            "parameters": parameters,
        }
        super().__init__(data, batch_size=empty_batchsize)

    @property
    def entities(self) -> TDEntities:
        return cast(TDEntities, self["entities"])

    @property
    def parameters(self) -> TDParameters:
        return cast(TDParameters, self["parameters"])
