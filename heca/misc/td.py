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

    def add_reference(self, img_raw: torch.Tensor):
        if "images_raw" not in self.keys():
            self["images_raw"] = img_raw.unsqueeze(0)
        else:
            self["images_raw"] = torch.cat(
                [self.images_raw, img_raw.unsqueeze(0)], dim=0
            )

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

    def add_reference(self, point: torch.Tensor, img_raw: torch.Tensor):
        if "points" not in self.keys():
            self["points"] = point.unsqueeze(0)
        else:
            self["points"] = torch.cat([self.points, point.unsqueeze(0)], dim=0)

        if "images_raw" not in self.keys():
            self["images_raw"] = img_raw.unsqueeze(0)
        else:
            self["images_raw"] = torch.cat(
                [self.images_raw, img_raw.unsqueeze(0)], dim=0
            )

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

    def add_pose(
        self,
        cam_label: str,
        entity_label: str,
        point: torch.Tensor,
        img_raw: torch.Tensor,
    ):
        if cam_label not in self.keys():
            self[cam_label] = TDCamReferences()
        cam_refs = self.cams[cam_label]

        if entity_label not in cam_refs.pose_refs.keys():
            cam_refs.pose_refs[entity_label] = TDPoseReferences()

        pose_ref = cam_refs.pose_refs[entity_label]
        assert isinstance(pose_ref, TDPoseReferences)
        pose_ref.add_reference(
            point=point,
            img_raw=img_raw,
        )

    def add_state(
        self,
        cam_label: str,
        entity_label: str,
        img_raw: torch.Tensor,
    ):
        if cam_label not in self.keys():
            self[cam_label] = TDCamReferences()
        cam_refs = self.cams[cam_label]

        if entity_label not in cam_refs.entity_states.keys():
            cam_refs.entity_states[entity_label] = TDStateReferences()

        state_ref = cam_refs.entity_states[entity_label]
        assert isinstance(state_ref, TDStateReferences)
        state_ref.add_reference(img_raw)


class TDCamRecord(TensorDict):
    def __init__(
        self,
        rgb: torch.Tensor,
        d: torch.Tensor,
        mask: torch.Tensor,
        extr: torch.Tensor,
        intr: torch.Tensor,
    ):
        super().__init__(
            {
                "rgb": rgb,
                "d": d,
                "mask": mask,
                "extr": extr,
                "intr": intr,
            },
            batch_size=empty_bs,
        )

    @property
    def rgb(self) -> torch.Tensor:
        return cast(torch.Tensor, self["rgb"])

    @property
    def d(self) -> torch.Tensor:
        return cast(torch.Tensor, self["d"])

    @property
    def mask(self) -> torch.Tensor:
        return cast(torch.Tensor, self["mask"])

    @property
    def extr(self) -> torch.Tensor:
        return cast(torch.Tensor, self["extr"])

    @property
    def intr(self) -> torch.Tensor:
        return cast(torch.Tensor, self["intr"])


class TDCamRecordings(TensorDict):
    def __init__(self, records: dict[str, TDCamRecord]):
        super().__init__(
            records,
            batch_size=empty_bs,
        )

    @property
    def records(self) -> dict[str, TDCamRecord]:
        return {
            label: cast(TDCamRecord, self[label])
            for label in self.keys()
            if isinstance(label, str)
        }
