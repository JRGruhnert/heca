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

    def get_state_points(self) -> torch.Tensor:
        if "points" not in self.keys():
            raise KeyError()
        return cast(torch.Tensor, self["points"])

    def get_state_descriptors(self) -> torch.Tensor:
        if "descriptors" not in self.keys():
            raise KeyError()
        return cast(torch.Tensor, self["descriptors"])

    def get_state_images_raw(self) -> torch.Tensor:
        if "images_raw" not in self.keys():
            raise KeyError()
        return cast(torch.Tensor, self["images_raw"])

    def get_state_references(self) -> torch.Tensor:
        raise NotImplementedError()


class TDPoseStates(TensorDict):
    def __init__(self):
        super().__init__({}, batch_size=empty_bs)


class TDPoseReferences(TensorDict):
    def __init__(self):
        super().__init__(
            {},
            batch_size=empty_bs,
        )

    def get_points(self) -> torch.Tensor:
        if "points" not in self.keys():
            raise KeyError()
        return cast(torch.Tensor, self["points"])

    def get_descriptors(self) -> torch.Tensor:
        if "descriptors" not in self.keys():
            raise KeyError()
        return cast(torch.Tensor, self["descriptors"])

    def get_images_raw(self) -> torch.Tensor:
        if "images_raw" not in self.keys():
            raise KeyError()
        return cast(torch.Tensor, self["images_raw"])

    @cached_property
    def get_point_references(self) -> torch.Tensor:
        points = self.get_points()  # (B, 2)
        descriptors = self.get_descriptors()  # (B, D, H, W)
        B = descriptors.shape[0]
        x = points[:, 0]
        y = points[:, 1]
        batch_idx = torch.arange(B)
        return descriptors[batch_idx, :, x, y]  # (B, D)


class TDCamReferences(TensorDict):
    def __init__(self):
        super().__init__(
            {
                "pose_references": TDPoseReferences(),
                "pose_states": TDPoseStates(),
            },
            batch_size=empty_bs,
        )

    def get_pose_references(self) -> TDPoseReferences:
        if "pose_references" not in self.keys():
            raise KeyError()
        return cast(TDPoseReferences, self["pose_references"])

    def get_pose_states(self) -> TDPoseStates:
        if "pose_states" not in self.keys():
            raise KeyError()
        return cast(TDPoseStates, self["pose_states"])


class TDSceneReferences(TensorDict):
    def __init__(self):
        super().__init__(
            {},
            batch_size=empty_bs,
        )

    def get_all_cams(self) -> dict[str, TDCamReferences]:
        for label in self.keys():
            if not isinstance(label, str):
                raise TypeError()
        return {label: cast(TDCamReferences, self[label]) for label in self.keys()}
