from functools import cached_property
from typing import cast

import numpy as np
from tensordict import TensorDict
import torch

empty_bs = torch.Size([])


class TDEntity(TensorDict):

    def __init__(
        self,
        # domain: torch.Tensor,
        position: torch.Tensor,
        rotation: torch.Tensor,
        state: torch.Tensor,
    ):
        data = {
            # "domain": domain,
            "position": position,
            "rotation": rotation,
            "state": state,
        }
        super().__init__(data, batch_size=empty_bs)


class TDProperties(TensorDict):
    def __init__(self, values: dict[str, torch.Tensor]):
        super().__init__(values, batch_size=empty_bs)

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
        super().__init__(entities, batch_size=empty_bs)


class TDScene(TensorDict):
    def __init__(self, formats: dict[str, TensorDict]):
        super().__init__(formats, batch_size=empty_bs)

    def heca_format(self) -> TDEntities:
        assert "heca" in self.keys()
        return cast(TDEntities, self["heca"])

    def tapas_format(self) -> TDProperties:
        assert "tapas" in self.keys()
        return cast(TDProperties, self["tapas"])


class TDWorld(TensorDict):
    def __init__(self, scenes: dict[str, TDScene]):
        super().__init__(scenes, batch_size=empty_bs)


class TDStateReferences(TensorDict):
    # Each B here stands for one state
    def __init__(self):
        super().__init__({}, batch_size=empty_bs)

    @property
    def images_raw(self) -> torch.Tensor:  # (B, C, H, W)
        return cast(torch.Tensor, self["images_raw"])

    @property
    def cls_tokens(self) -> torch.Tensor:  # (B, D)
        return cast(torch.Tensor, self["states"])

    def set_preprocessed(self, states: torch.Tensor):
        self["states"] = states


class TDEntityStates(TensorDict):
    def __init__(self):
        super().__init__({}, batch_size=empty_bs)

    @property
    def entities(self) -> dict[str, TDStateReferences]:
        return {
            label: cast(TDStateReferences, self[label])
            for label in self.keys()
            if isinstance(label, str)
        }


class TDPoseReferences(TensorDict):
    def __init__(self):
        super().__init__(
            {},
            batch_size=empty_bs,
        )

    @property
    def points(self) -> torch.Tensor:
        return cast(torch.Tensor, self["points"])

    @property
    def images_raw(self) -> torch.Tensor:
        return cast(torch.Tensor, self["images_raw"])

    @property
    def descriptors(self) -> torch.Tensor:
        return cast(torch.Tensor, self["descriptors"])

    @cached_property
    def kp_refs(self) -> torch.Tensor:
        points = self.points  # (B, 2)
        descriptors = self.descriptors  # (B, D, H, W)
        x = points[:, 0]
        y = points[:, 1]
        batch_idx = torch.arange(descriptors.shape[0])
        return descriptors[batch_idx, :, x, y]  # (B, D)

    def set_preprocessed(self, desc: torch.Tensor):
        self["descriptors"] = desc


class TDCamReferences(TensorDict):
    def __init__(self):
        super().__init__(
            {
                "pose_references": TDPoseReferences(),
                "entity_states": TDEntityStates(),
            },
            batch_size=empty_bs,
        )

    @property
    def pose_refs(self) -> TDPoseReferences:
        return cast(TDPoseReferences, self["pose_references"])

    @property
    def entity_states(self) -> TDEntityStates:
        return cast(TDEntityStates, self["entity_states"])


class TDSceneReferences(TensorDict):
    def __init__(self):
        super().__init__(
            {},
            batch_size=empty_bs,
        )

    @property
    def cams(self) -> dict[str, TDCamReferences]:
        return {
            label: cast(TDCamReferences, self[label])
            for label in self.keys()
            if isinstance(label, str)
        }
