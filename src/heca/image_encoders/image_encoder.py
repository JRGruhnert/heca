import abc
import numpy as np
import torch

from dataclasses import dataclass

from heca.misc.base import Registerable
from heca.scenes.scene import Scene
from heca.data.data import TDImage
from heca.data.reference import Keypoint
from heca.utils.quaternion import Quaternion


class ImageEncoder(Registerable):
    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        label: str = "default"
        kp_selection_threshold: float = 0.2
        image_size: tuple[int, int] = (256, 256)

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def extract_poses(self, image: TDImage) -> dict[str, Keypoint]:
        """Where each entity's keypoint is in ``image``, keyed by entity label.

        A keypoint carries its world position, the matcher's confidence and the
        pixel it was found at, so a caller can tell a located keypoint from a
        guess and a viewer can draw it.
        """
        raise NotImplementedError()

    def extract_states(self, image: TDImage) -> dict[str, tuple[int, float]]:
        """The state of every entity that has more than one, keyed by label.

        An entity with a single state is constant by construction, so it is not
        asked about and does not appear here.
        """
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

    @staticmethod
    def hard_pixels_to_3D_world(
        y_pixel: torch.Tensor,  # B, N
        x_pixel: torch.Tensor,  # B, N
        depth: torch.Tensor,  # N, H, W
        extr: torch.Tensor,  # N, 4, 4
        intr: torch.Tensor,  # N, 3, 3
    ) -> torch.Tensor:
        """Pixels plus their measured depth -> world points, as (B, N, 3).

        TAPAS returned these flattened per keypoint and followed by an identity
        quaternion, which is their pose convention; nothing here reads a rotation
        off the keypoint encoder, so the quaternion is dropped and the shape stays
        one point per keypoint.
        """
        B, N = x_pixel.shape
        rows = torch.arange(B, device=depth.device, dtype=torch.long).repeat_interleave(N)
        # depth is (H, W), so the row comes first and the projection takes u = x
        z = depth[rows, y_pixel.flatten(), x_pixel.flatten()].reshape(B, N)
        return ImageEncoder.batched_pinhole_projection_image_to_world_coordinates_orig(
            x_pixel, y_pixel, z, intr, extr
        )

    @staticmethod
    def pixels_to_world(
        x: torch.Tensor,  # (B, N)
        y: torch.Tensor,  # (B, N)
        z: torch.Tensor,  # (B, N) camera-frame depth
        extr: torch.Tensor,  # (B, 4, 4)
        intr: torch.Tensor,  # (B, 3, 3)
    ) -> torch.Tensor:
        return ImageEncoder.batched_pinhole_projection_image_to_world_coordinates_orig(
            x, y, z, intr, extr
        )

    @staticmethod
    def world_to_pixels(
        points: torch.Tensor,  # (N, 3)
        extr: torch.Tensor,  # (4, 4) camera to world
        intr: torch.Tensor,  # (3, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        world_to_camera = torch.inverse(extr)
        homogeneous = torch.cat(
            [points, torch.ones_like(points[..., :1])], dim=-1
        )  # (N, 4)
        camera = homogeneous @ world_to_camera.transpose(-1, -2)
        z = camera[..., 2]
        safe = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
        x = intr[..., 0, 0] * camera[..., 0] / safe + intr[..., 0, 2]
        y = intr[..., 1, 1] * camera[..., 1] / safe + intr[..., 1, 2]
        return torch.stack([x, y], dim=-1), z

    @staticmethod
    def pixel_to_world(
        y: int,
        x: int,
        depth: np.ndarray,
        extr: np.ndarray,
        intr: np.ndarray,
    ) -> np.ndarray:
        positions = ImageEncoder.hard_pixels_to_3D_world(
            torch.tensor([[y]], dtype=torch.long),
            torch.tensor([[x]], dtype=torch.long),
            torch.as_tensor(np.asarray(depth)[None], dtype=torch.float32),
            torch.as_tensor(np.asarray(extr)[None], dtype=torch.float32),
            torch.as_tensor(np.asarray(intr)[None], dtype=torch.float32),
        )
        return positions[0, 0].numpy().astype(np.float64)

    @staticmethod
    def batched_pinhole_projection_image_to_camera_coordinates_orig(u, v, z, K):
        uv1 = torch.stack((u, v, torch.ones(u.shape, device=u.device)), dim=-1)
        K_inv = K.inverse()

        pos = torch.transpose(torch.matmul(K_inv, torch.transpose(uv1, -1, -2)), -1, -2)

        pos = z.unsqueeze(2).repeat(1, 1, 3) * pos
        return pos

    @staticmethod
    def batched_pinhole_projection_image_to_world_coordinates_orig(
        u, v, z, K, camera_to_world
    ):
        pos_in_camera_frame = (
            ImageEncoder.batched_pinhole_projection_image_to_camera_coordinates_orig(
                u, v, z, K
            )
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
    def prepare_for_scene(self, scene: Scene.Config):
        raise NotImplementedError()
