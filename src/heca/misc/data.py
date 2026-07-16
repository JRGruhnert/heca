from dataclasses import dataclass

import torch
import numpy as np
from typing import Iterator, cast
from tensordict import TensorDict


@dataclass(slots=True)
class DCEntity:
    value: np.ndarray
    feature: np.ndarray

    @classmethod
    def empty(cls) -> "DCEntity":
        return cls(np.empty(0), np.empty(0))


class DCScene:
    def __init__(
        self,
        ee: DCEntity,
        entities: dict[str, DCEntity],
        extras: dict[str, np.ndarray] = {},
    ):
        self._ee = ee
        self._entities = entities
        self._extras = extras if extras is not None else {}

    def __getitem__(self, key: str) -> DCEntity:
        return self._entities[key]

    def get(self, key: str) -> DCEntity:
        if key not in self._entities:
            raise KeyError(f"Entity {key} not found in TDScene")
        return self._entities[key]

    def set(self, key: str, value: DCEntity):
        self._entities[key] = value

    @property
    def extras(self) -> dict[str, np.ndarray]:
        return self._extras

    @property
    def ee(self) -> DCEntity:
        return self._ee

    def entities(self) -> Iterator[tuple[str, DCEntity]]:
        for key, value in self._entities.items():
            yield key, value

    @classmethod
    def empty(cls) -> "DCScene":
        return cls(DCEntity.empty(), {})


class TDImage(TensorDict):
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
            batch_size=torch.Size([]),
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


class TDSceneReferences(TensorDict):
    def __init__(self):
        super().__init__({}, batch_size=torch.Size([]))

    def add_scene(
        self, label: str, patch_desc: torch.Tensor, state_coords: torch.Tensor
    ):
        self[label] = patch_desc
        self[f"{label}_state_coords"] = state_coords
        self["patches"] = torch.cat(
            (
                [self["patches"], patch_desc.unsqueeze(0)]
                if "patches" in self.keys()
                else [patch_desc.unsqueeze(0)]
            ),
            dim=0,
        )
        self["state_coords"] = torch.cat(
            (
                [self["state_coords"], state_coords.unsqueeze(0)]
                if "state_coords" in self.keys()
                else [state_coords.unsqueeze(0)]
            ),
            dim=0,
        )

    def get_reference(self, label: str) -> torch.Tensor:
        if label not in self.keys():
            raise KeyError(f"Scene {label} not found in TDSceneReferences")
        return cast(torch.Tensor, self[label])

    def get_state_coord(self, label: str) -> torch.Tensor:
        key = f"{label}_state_coords"
        if key not in self.keys():
            raise KeyError(
                f"State coordinates for scene {label} not found in TDSceneReferences"
            )
        return cast(torch.Tensor, self[key])

    @property
    def patches(self) -> torch.Tensor:
        if "patches" not in self.keys():
            raise KeyError("No patches found in TDSceneReferences")
        return cast(torch.Tensor, self["patches"])

    @property
    def state_coords(self) -> torch.Tensor:
        if "state_coords" not in self.keys():
            raise KeyError("No state coordinates found in TDSceneReferences")
        return cast(torch.Tensor, self["state_coords"])


def relative_quaternion(q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    # q, q_ref: (..., 4) in (w, x, y, z) or (x, y, z, w) format
    # Assume (x, y, z, w) format as is common in PyTorch/robotics
    # Convert to (w, x, y, z) for computation if needed
    # Here, we assume (x, y, z, w)
    def quat_conj(q):
        return torch.tensor([-q[0], -q[1], -q[2], q[3]], device=q.device, dtype=q.dtype)

    def quat_mult(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([x, y, z, w])

    q_ref_conj = quat_conj(q_ref)
    return quat_mult(q, q_ref_conj)
