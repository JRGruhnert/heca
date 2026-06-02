import abc
from pathlib import Path
import re
import torch

from dataclasses import dataclass

from heca.classes.persist import Persistable
from heca.entities.entity import Entity
from heca.misc import logger
from heca.misc.td import TDImage
from PIL import Image


class ImageExtractor(Persistable):
    @dataclass(frozen=True, kw_only=True)
    class Location(Persistable.Location):
        folder: str = "references"

    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        kp_selection_threshold: float = 0.2
        image_size: tuple[int, int] = (256, 256)

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abc.abstractmethod
    def extract_entities(
        self, image: TDImage, entities: list[Entity]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_entity_states(
        self, image: TDImage, entities: list[Entity], kps: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def normalize_coords(
        self, coords: torch.Tensor, size_hw: tuple[int, int]
    ) -> torch.Tensor:
        return torch.stack(
            [
                2.0 * coords[..., 0] / max(size_hw[0] - 1, 1) - 1.0,
                2.0 * coords[..., 1] / max(size_hw[1] - 1, 1) - 1.0,
            ],
            dim=-1,
        )  # (..., 2)

    def transform_coords(
        self, y: int, x: int, origin_h: int, origin_w: int, target_h: int, target_w: int
    ) -> tuple[int, int]:
        # logger.debug(f"get_patch_index_from_img_coords: size: {img_desc.shape}")
        ref_norm_yx = self.normalize_coords(torch.tensor([y, x]), (origin_h, origin_w))
        target_yx = self.scale_normalized_coords(ref_norm_yx, (target_h, target_w))
        return int(target_yx[0].item()), int(target_yx[1].item())

    def scale_normalized_coords(
        self, norm_coords: torch.Tensor, size_hw: tuple[int, int]
    ) -> torch.Tensor:
        height, width = size_hw
        scale_yx = torch.tensor([height - 1, width - 1])
        pixel_coordinates_yx = (norm_coords + 1.0) * 0.5 * scale_yx
        return pixel_coordinates_yx.round().long()

    def kps_2d_to_3d(self, image: TDImage, kps_raw_2d: torch.Tensor) -> torch.Tensor:
        u_norm, v_norm = kps_raw_2d.chunk(2, dim=-1)
        norm_uv = torch.stack((v_norm, u_norm), dim=-1)
        size_hw = (image.rgb.shape[1], image.rgb.shape[2])
        pixel_yx = self.scale_normalized_coords(norm_uv, size_hw)
        y_pixel = pixel_yx[..., 0]  # (B, Nref)
        x_pixel = pixel_yx[..., 1]  # (B, Nref)

        return self.hard_pixels_to_3D_world(
            y_pixel, x_pixel, image.d, image.extr, image.intr
        )

    def hard_pixels_to_3D_world(
        self,
        y_pixel: torch.Tensor,  # B, Nref
        x_pixel: torch.Tensor,  # B, Nref
        depth: torch.Tensor,  # N, H, W
        extr: torch.Tensor,  # N, 4, 4
        intr: torch.Tensor,  # N, 3, 3
    ):

        B, N_kp = x_pixel.shape
        batch_indices = torch.arange(
            B, device=depth.device, dtype=torch.long
        ).repeat_interleave(N_kp)
        z = depth[batch_indices, x_pixel.flatten(), y_pixel.flatten()]
        z = z.reshape(B, N_kp)

        pos = self.batched_pinhole_projection_image_to_world_coordinates_orig(
            y_pixel, x_pixel, z, intr, extr
        )

        return pos.permute(0, 2, 1).reshape((B, -1))

    def batched_pinhole_projection_image_to_camera_coordinates_orig(self, u, v, z, K):
        uv1 = torch.stack((u, v, torch.ones(u.shape, device=u.device)), dim=-1)
        K_inv = K.inverse()

        pos = torch.transpose(torch.matmul(K_inv, torch.transpose(uv1, -1, -2)), -1, -2)

        pos = z.unsqueeze(2).repeat(1, 1, 3) * pos
        return pos

    def batched_pinhole_projection_image_to_world_coordinates_orig(
        self, u, v, z, K, camera_to_world
    ):
        pos_in_camera_frame = (
            self.batched_pinhole_projection_image_to_camera_coordinates_orig(u, v, z, K)
        )
        pos_in_camera_frame_homog = torch.cat(
            (
                pos_in_camera_frame,
                torch.ones(
                    (*pos_in_camera_frame.shape[:-1], 1),
                    device=pos_in_camera_frame.device,
                ),
            ),
            dim=-1,
        )

        pos_in_world_homog = torch.transpose(
            torch.matmul(
                camera_to_world, torch.transpose(pos_in_camera_frame_homog, -1, -2)
            ),
            -1,
            -2,
        )

        return pos_in_world_homog[..., :3]

    @abc.abstractmethod
    def prepare_references(
        self,
        entities: list[Entity],
        kp_references: list[tuple[Image.Image, int, int, int, int]],
        state_references: list[dict[str, list[Image.Image]]],
    ):
        assert len(entities) == len(kp_references) == len(state_references)
        for entity, kp_refs, state_refs in zip(
            entities, kp_references, state_references
        ):
            self.kp_extractor.register(entity.cfg.label, kp_refs)
            for state_label, state_imgs in state_refs.items():
                for state_img in state_imgs:
                    self.state_extractor.register(
                        entity.cfg.label, state_label, state_img
                    )

    def add_entity_reference_sample(
        self, dc_ref_patch_desc: torch.Tensor, state_coords: torch.Tensor
    ):
        if self.kp_patch_descr is None or self.entity_state_coords is None:
            self.kp_patch_descr = dc_ref_patch_desc
            self.entity_state_coords = state_coords
        else:
            self.kp_patch_descr = torch.cat(
                (self.kp_patch_descr, dc_ref_patch_desc), dim=0
            )
            self.entity_state_coords = torch.cat(
                (self.entity_state_coords, state_coords), dim=0
            )

    def add_entity_sample_for_cam(
        self,
        kpt: list[tuple[Image.Image, int, int, int, int]],
        kps: dict[str, list[Image.Image]],
    ):
        assert len(kpt) == self.cfg.sample_per_reference
        assert len(kps) == len(entity.cfg.states)
        assert all(state in kps for state in entity.cfg.states)
        assert all(
            len(kps[state]) == self.cfg.sample_per_reference
            for state in entity.cfg.states
        )
        sample_patch_descs = []
        sample_state_coords = []
        for i, (dc_img, dc_x, dc_y, state_x, state_y) in enumerate(kpt):
            dc_img_desc = self.compute_descriptor(dc_img)  # (1, D, H, W)
            NSample, D, H, W = dc_img_desc.shape
            dc_py, dc_px = self.transform_coords(
                dc_x, dc_y, dc_img.height, dc_img.width, H, W
            )
            state_py, state_px = self.transform_coords(
                state_x, state_y, dc_img.height, dc_img.width, H, W
            )
            sample_patch_descs.append(dc_img_desc[0, :, dc_py, dc_px])  # (D,)

            dc_norm = self.normalize_coords(
                torch.tensor([dc_py, dc_px]), self.image_size
            )
            state_norm = self.normalize_coords(
                torch.tensor([state_py, state_px]), self.image_size
            )
            sample_state_coords.append(state_norm - dc_norm)  # (2,)

        dc_ref_patch_desc = torch.stack(sample_patch_descs, dim=0)  # (NSample, D)
        # Average relative state offset across samples → single (1, 2) per entity
        state_coords = (
            torch.stack(sample_state_coords, dim=0).mean(dim=0).unsqueeze(0)
        )  # (1, 2)

        self.add_entity_reference_sample(dc_ref_patch_desc, state_coords)

        for state, images in kps.items():
            for img in images:
                state_img_desc = self.compute_descriptor(img)  # (1, D, H, W)
                kps_raw_2d, _, _, _ = self.compute_keypoints(
                    state_img_desc, dc_ref_patch_desc
                )  # (1, 2)
                kernel = self.get_state_kernel(
                    state_img_desc,
                    kps_raw_2d,
                    state_coords,
                )  # (1, C, k, k)
                self.entity_state_knn.register(
                    entity.cfg.label,
                    state,
                    kernel,
                )
