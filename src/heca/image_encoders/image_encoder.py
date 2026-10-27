import abc
import numpy as np
import torch

from dataclasses import dataclass

from heca.misc.base import Registerable
from heca.scenes.scene import Scene
from heca.data.data import TDImage
from heca.data.reference import Keypoint
from heca.utils.quaternion import Quaternion


def device_from_spec(spec: str) -> torch.device | None:
    """The device a config string names, or None when it names none.

    A bare number is a CUDA card, so "1" and "cuda:1" are the same device. The
    card variables are plain indices everywhere else -- HECA_MOLMO_DEVICES is
    handed to accelerate's max_memory map, whose keys have to be integers -- so
    they are spelled as plain indices here too.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    try:
        return torch.device(f"cuda:{int(spec)}")
    except ValueError:
        return torch.device(spec)


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

    def patch_index(
        self,
        x: int,
        y: int,
        origin_hw: tuple[int, int],
        target_hw: tuple[int, int],
    ) -> tuple[int, int]:
        return self.transform_coords(
            y=y,
            x=x,
            origin_h=origin_hw[0],
            origin_w=origin_hw[1],
            target_h=target_hw[0],
            target_w=target_hw[1],
        )

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
    def pixels_and_depth_to_world(
        x_pixel: torch.Tensor,  # B, N
        y_pixel: torch.Tensor,  # B, N
        depth: torch.Tensor,  # B, N camera-axis depth
        extr: torch.Tensor,  # 4, 4 or B, 4, 4   world -> camera
        intr: torch.Tensor,  # 3, 3 or B, 3, 3
    ) -> torch.Tensor:
        """Pixels plus their depth -> world points, as (B, N, 3).

        ``extr`` is world -> camera (what ``get_camera_extrinsics`` returns) and the
        depth is MuJoCo's, measured along the viewing direction in a camera frame
        that is x-right, y-up, z-back. The mapping is spelled out here rather than
        routed through the +z-forward pinhole helper, because no single sign flip
        reproduces that frame: reading it as z-forward and y-down mirrors every
        keypoint and puts it about a metre from the object it belongs to.
        """
        batch = x_pixel.shape[0]
        k = intr.reshape(-1, 3, 3)
        if k.shape[0] == 1 and batch > 1:
            k = k.expand(batch, -1, -1)
        u = (x_pixel.float() - k[:, 0, 2:3]) / k[:, 0, 0:1]
        v = (y_pixel.float() - k[:, 1, 2:3]) / k[:, 1, 1:2]
        camera = torch.stack((u * depth, -v * depth, -depth), dim=-1)

        to_world = torch.inverse(extr.reshape(-1, 4, 4))
        if to_world.shape[0] == 1 and batch > 1:
            to_world = to_world.expand(batch, -1, -1)
        rotation = to_world[:, :3, :3]
        translation = to_world[:, :3, 3]
        return camera @ rotation.transpose(-1, -2) + translation.unsqueeze(1)

    @staticmethod
    def hard_pixels_to_3D_world(
        y_pixel: torch.Tensor,  # B, N
        x_pixel: torch.Tensor,  # B, N
        depth: torch.Tensor,  # (H, W) or (N, H, W)
        extr: torch.Tensor,  # 4, 4 or N, 4, 4   world -> camera
        intr: torch.Tensor,  # 3, 3 or N, 3, 3
    ) -> torch.Tensor:
        """The same mapping, reading the depth of each pixel out of a depth image."""
        B, N = x_pixel.shape
        if depth.ndim == 3:
            rows = torch.arange(
                B, device=depth.device, dtype=torch.long
            ).repeat_interleave(N)
            z = depth[rows, y_pixel.flatten(), x_pixel.flatten()].reshape(B, N)
        else:
            z = depth[y_pixel.flatten(), x_pixel.flatten()].reshape(B, N)
        return ImageEncoder.pixels_and_depth_to_world(x_pixel, y_pixel, z, extr, intr)

    @staticmethod
    def pixels_to_world(
        x: torch.Tensor,  # (B, N)
        y: torch.Tensor,  # (B, N)
        z: torch.Tensor,  # (B, N) camera-axis depth
        extr: torch.Tensor,  # (B, 4, 4) world -> camera
        intr: torch.Tensor,  # (B, 3, 3)
    ) -> torch.Tensor:
        return ImageEncoder.pixels_and_depth_to_world(x, y, z, extr, intr)

    @staticmethod
    def world_to_pixels(
        points: torch.Tensor,  # (N, 3)
        extr: torch.Tensor,  # (4, 4) world -> camera
        intr: torch.Tensor,  # (3, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World points -> pixels and their depth, the inverse of the mapping above."""
        homogeneous = torch.cat(
            [points, torch.ones_like(points[..., :1])], dim=-1
        )  # (N, 4)
        camera = homogeneous @ extr.transpose(-1, -2)
        z = -camera[..., 2]
        safe = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
        x = intr[..., 0, 0] * camera[..., 0] / safe + intr[..., 0, 2]
        y = -intr[..., 1, 1] * camera[..., 1] / safe + intr[..., 1, 2]
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
