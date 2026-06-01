import abc
import torch

from dataclasses import dataclass

from heca.classes.register import Registerable
from heca.entities.entity import Entity
from heca.misc.td import TDImage


class ImageExtractor(Registerable):
    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        kp_selection_threshold: float = 0.2
        image_size: tuple[int, int] = (256, 256)

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abc.abstractmethod
    def extract_entities(
        self,
        td_img: TDImage,
        entities: list[Entity],
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_states(
        self,
        td_img: TDImage,
        entities: list[Entity],
        kps_raw_2d: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def kps_raw_2d_to_image_coords(
        self, kps_raw_2d: torch.Tensor, size_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_norm, v_norm = kps_raw_2d.chunk(2, dim=-1)
        norm_uv = torch.stack((v_norm, u_norm), dim=-1)
        pixel_yx = self.scale_normalized_coords(norm_uv, size_hw)
        y_pixel = pixel_yx[..., 0]  # (B, Nref)
        x_pixel = pixel_yx[..., 1]  # (B, Nref)
        return y_pixel, x_pixel

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
