from typing import cast

from tensordict import TensorDict
import torch

empty_bs = torch.Size([])


class TDEntity(TensorDict):

    def __init__(
        self,
        position: torch.Tensor,
        rotation: torch.Tensor,
        state: torch.Tensor,
    ):
        data = {
            "position": position,
            "rotation": rotation,
            "state": state,
        }
        super().__init__(data, batch_size=empty_bs)

    @property
    def position(self) -> torch.Tensor:
        return self["position"]

    @property
    def rotation(self) -> torch.Tensor:
        return self["rotation"]

    @property
    def state(self) -> torch.Tensor:
        return self["state"]


class TDScene(TensorDict):
    def __init__(
        self, entities: dict[str, TDEntity], extras: dict[str, torch.Tensor] = {}
    ):
        super().__init__({**entities, "extras": extras}, batch_size=empty_bs)

    def __getitem__(self, key: str) -> TDEntity:
        if key not in self.keys():
            raise KeyError(f"Entity {key} not found in TDScene")
        return cast(TDEntity, super().__getitem__(key))

    @property
    def extras(self) -> dict[str, torch.Tensor]:
        return self.get("extras", {})


class TDAgentCon(TensorDict):
    def __init__(self, entities: dict[str, TDEntity]):
        super().__init__(entities, batch_size=empty_bs)

    def __getitem__(self, key: str) -> TDEntity:
        if key not in self.keys():
            raise KeyError(f"Entity {key} not found in TDAgentCon")
        return cast(TDEntity, super().__getitem__(key))


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


class TDSceneReferences(TensorDict):
    def __init__(self):
        super().__init__({}, batch_size=empty_bs)

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


# class TDStateReferences(TensorDict):
#     # Each B here stands for one state
#     def __init__(self):
#         super().__init__({}, batch_size=empty_bs)

#     @property
#     def images_raw(self) -> torch.Tensor:  # (B, C, H, W)
#         return cast(torch.Tensor, self["images_raw"])

#     def add_reference(self, img_raw: torch.Tensor):
#         if "images_raw" not in self.keys():
#             self["images_raw"] = img_raw.unsqueeze(0)
#         else:
#             self["images_raw"] = torch.cat(
#                 [self.images_raw, img_raw.unsqueeze(0)], dim=0
#             )

#     @property
#     def cls_tokens(self) -> torch.Tensor:  # (B, D)
#         return cast(torch.Tensor, self["states"])

#     def set_preprocessed(self, states: torch.Tensor):
#         self["states"] = states


# class TDEntityStates(TensorDict):
#     def __init__(self):
#         super().__init__({}, batch_size=empty_bs)

#     @property
#     def entities(self) -> dict[str, TDStateReferences]:
#         return {
#             label: cast(TDStateReferences, self[label])
#             for label in self.keys()
#             if isinstance(label, str)
#         }


# class TDPoseReferences(TensorDict):
#     def __init__(self):
#         super().__init__(
#             {},
#             batch_size=empty_bs,
#         )

#     @property
#     def points(self) -> torch.Tensor:
#         return cast(torch.Tensor, self["points"])

#     @property
#     def images_raw(self) -> torch.Tensor:
#         return cast(torch.Tensor, self["images_raw"])

#     def add_reference(self, point: torch.Tensor, img_raw: torch.Tensor):
#         if "points" not in self.keys():
#             self["points"] = point.unsqueeze(0)
#         else:
#             self["points"] = torch.cat([self.points, point.unsqueeze(0)], dim=0)

#         if "images_raw" not in self.keys():
#             self["images_raw"] = img_raw.unsqueeze(0)
#         else:
#             self["images_raw"] = torch.cat(
#                 [self.images_raw, img_raw.unsqueeze(0)], dim=0
#             )

#     @property
#     def descriptors(self) -> torch.Tensor:
#         return cast(torch.Tensor, self["descriptors"])

#     @cached_property
#     def kp_refs(self) -> torch.Tensor:
#         points = self.points  # (B, 2)
#         descriptors = self.descriptors  # (B, D, H, W)
#         x = points[:, 0]
#         y = points[:, 1]
#         batch_idx = torch.arange(descriptors.shape[0])
#         return descriptors[batch_idx, :, x, y]  # (B, D)

#     def set_preprocessed(self, desc: torch.Tensor):
#         self["descriptors"] = desc


# class TDCamReferences(TensorDict):
#     def __init__(self):
#         super().__init__(
#             {
#                 "pose_references": TDPoseReferences(),
#                 "entity_states": TDEntityStates(),
#             },
#             batch_size=empty_bs,
#         )

#     @property
#     def pose_refs(self) -> TDPoseReferences:
#         return cast(TDPoseReferences, self["pose_references"])

#     @property
#     def entity_states(self) -> TDEntityStates:
#         return cast(TDEntityStates, self["entity_states"])


# class TDSceneReferences(TensorDict):
#     def __init__(self):
#         super().__init__(
#             {},
#             batch_size=empty_bs,
#         )

#     @property
#     def cams(self) -> dict[str, TDCamReferences]:
#         return {
#             label: cast(TDCamReferences, self[label])
#             for label in self.keys()
#             if isinstance(label, str)
#         }

#     def add_pose(
#         self,
#         cam_label: str,
#         entity_label: str,
#         point: torch.Tensor,
#         img_raw: torch.Tensor,
#     ):
#         if cam_label not in self.keys():
#             self[cam_label] = TDCamReferences()
#         cam_refs = self.cams[cam_label]

#         if entity_label not in cam_refs.pose_refs.keys():
#             cam_refs.pose_refs[entity_label] = TDPoseReferences()

#         pose_ref = cam_refs.pose_refs[entity_label]
#         assert isinstance(pose_ref, TDPoseReferences)
#         print(
#             f"Adding point: {point}, img_raw shape: {img_raw.shape} to cam {cam_label}, entity {entity_label}"
#         )
#         pose_ref.add_reference(
#             point=point,
#             img_raw=img_raw,
#         )

#     def add_state(
#         self,
#         cam_label: str,
#         entity_label: str,
#         img_raw: torch.Tensor,
#     ):
#         if cam_label not in self.keys():
#             self[cam_label] = TDCamReferences()
#         cam_refs = self.cams[cam_label]

#         if entity_label not in cam_refs.entity_states.keys():
#             cam_refs.entity_states[entity_label] = TDStateReferences()

#         state_ref = cam_refs.entity_states[entity_label]
#         assert isinstance(state_ref, TDStateReferences)
#         state_ref.add_reference(img_raw)

# class TDProperties(TensorDict):
#     def __init__(self, values: dict[str, torch.Tensor]):
#         super().__init__(values, batch_size=empty_bs)

#     @classmethod
#     def from_numpy_dict(cls, data: dict[str, np.ndarray]) -> "TDProperties":
#         return cls({k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()})

#     def same_fields(self, other: "TDProperties") -> bool:
#         return set(self.keys()) == set(other.keys())  # type: ignore

#     def equal(self, other: "TDProperties") -> bool:
#         result = self == other
#         if isinstance(result, bool):
#             return result
#         return bool(result.all())

#     def get_by_label(self, label: str) -> torch.Tensor:
#         if label not in self.keys():
#             raise KeyError()
#         return self[label]

# class TDWorld(TensorDict):
#     def __init__(self, scenes: dict[str, TDScene]):
#         super().__init__(scenes, batch_size=empty_bs)


def relative_quaternion(q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    """
    Compute the relative quaternion q_rel such that: q = q_rel * q_ref
    Returns q_rel = q * q_ref_conj
    """

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
    q_rel = quat_mult(q, q_ref_conj)
    return q_rel


def make_abs_and_rel_td_entity(
    position: torch.Tensor,
    rotation: torch.Tensor,
    state: torch.Tensor,
    cursor_pos: torch.Tensor,
    cursor_rot: torch.Tensor,
) -> tuple[TDEntity, TDEntity]:
    td_rel = TDEntity(
        position=position - cursor_pos,
        rotation=relative_quaternion(rotation, cursor_rot),
        state=state,
    )
    td_abs = TDEntity(
        position=position,
        rotation=rotation,
        state=state,
    )
    return td_abs, td_rel
