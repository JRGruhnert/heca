from typing import cast

import numpy as np
from tensordict import TensorDict
import torch

empty_batchsize = torch.Size([])


class TDEntity(TensorDict):

    def __init__(
        self,
        domain: torch.Tensor,
        position: torch.Tensor,
        rotation: torch.Tensor,
        state: torch.Tensor,
    ):
        data = {
            "domain": domain,
            "position": position,
            "rotation": rotation,
            "state": state,
        }
        super().__init__(data, batch_size=empty_batchsize)

    @property
    def domain(self) -> torch.Tensor:
        return self["domain"]

    @domain.setter
    def domain(self, value: torch.Tensor):
        self["domain"] = value

    @property
    def position(self) -> torch.Tensor:
        return self["position"]

    @position.setter
    def position(self, value: torch.Tensor):
        self["position"] = value

    @property
    def rotation(self) -> torch.Tensor:
        return self["rotation"]

    @rotation.setter
    def rotation(self, value: torch.Tensor):
        self["rotation"] = value

    @property
    def state(self) -> torch.Tensor:
        return self["state"]

    @state.setter
    def state(self, value: torch.Tensor):
        self["state"] = value


class TDProperties(TensorDict):
    def __init__(self, values: dict[str, torch.Tensor]):
        super().__init__(values, batch_size=empty_batchsize)

    @classmethod
    def from_numpy_dict(cls, data: dict[str, np.ndarray]) -> "TDProperties":
        return cls({k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()})

    def same_fields(self, other: "TDProperties") -> bool:
        return set(self.keys()) == set(other.keys())  # type: ignore

    def equal(self, other: "TDProperties") -> bool:
        result = self == other
        if isinstance(result, bool):
            return result
        return bool(result.all())

    def get_by_label(self, label: str) -> torch.Tensor:
        if label not in self.keys():
            raise KeyError()
        return self[label]


class TDEntities(TensorDict):
    def __init__(self, entities: dict[str, TDEntity]):
        super().__init__(entities, batch_size=empty_batchsize)

    def get(self, key: str) -> TDEntity:
        if key not in self.keys():
            raise KeyError(f"Entity '{key}' not found in TDEntities.")
        return cast(TDEntity, self[key])


class TDScene(TensorDict):
    def __init__(
        self,
        heca: TDEntities,
        leaf: TensorDict | None = None,
    ):
        super().__init__(heca, batch_size=empty_batchsize)
        self["entities"] = heca
        if leaf is not None:
            self["leaf"] = leaf

    @property
    def entities(self) -> TDEntities:
        return cast(TDEntities, self["entities"])

    @property
    def leaf(self) -> TensorDict:
        return self.get("leaf")

    @property
    def length(self) -> int:
        return len(self.entities)


class TDWorld(TensorDict):
    def __init__(self, scenes: dict[str, TDScene]):
        super().__init__(scenes, batch_size=empty_batchsize)

    def scenes(self, label: str) -> TDScene:
        if label in self:
            return cast(TDScene, self[label])
        assert False, f"World with label '{label}' not found."

    @property
    def calvin(self) -> TDScene:
        return self.scenes("calvin")

    @property
    def ogbench(self) -> TDScene:
        return self.scenes("ogbench")

    def __len__(self):
        return len(self.keys())
