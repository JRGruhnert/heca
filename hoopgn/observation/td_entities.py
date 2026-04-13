from typing import cast

from tensordict import TensorDict

from hoopgn.observation import empty_batchsize
from hoopgn.observation.td_entity import TDEntity


class TDEntities(TensorDict):
    def __init__(self, entities: dict[str, TDEntity]):
        super().__init__(entities, batch_size=empty_batchsize)

    def get_entity(self, key: str) -> TDEntity:
        if key not in self.keys():
            raise KeyError(f"Entity '{key}' not found in TDEntities.")
        return cast(TDEntity, self[key])
