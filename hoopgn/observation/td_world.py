from tensordict import TensorDict
from typing import cast

from hoopgn.observation import empty_batchsize
from hoopgn.observation.td_scene import TDScene


class TDWorld(TensorDict):
    def __init__(self, scenes: dict[str, TDScene]):
        super().__init__(scenes, batch_size=empty_batchsize)

    def __getitem__(self, key: str) -> TDScene:
        return cast(TDScene, super().__getitem__(key))
