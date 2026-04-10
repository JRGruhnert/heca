from dataclasses import dataclass
import math

import torch


@dataclass(kw_only=True)
class BinaryConfig:
    pass


class Binary:
    def __init__(self, config: BinaryConfig):
        self.config = config

    def normalize_quat(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize quaternion and ensure positive w component."""
        nx = x / torch.linalg.norm(x)
        if nx[3] < 0:
            return -nx
        return nx

    def is_binary(self, x: torch.Tensor) -> bool:
        """Check if the tensor contains only binary values (0 and 1)."""
        return bool(((x == 0.0) | (x == 1.0)).all())

    def mean(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean quaternion using the eigenvector method.
        quaternions: tensor of shape [N, 4] (x, y, z, w)
        Returns: mean quaternion [4] in (x, y, z, w) format
        """
        # Swap to (w, x, y, z) for computation
        quats = quaternions[:, [3, 0, 1, 2]]
        quats = quats / quats.norm(dim=1, keepdim=True)
        A = quats.t() @ quats
        _, eigenvectors = torch.linalg.eigh(A)
        mean_quat = eigenvectors[:, -1]
        # Ensure positive scalar part
        if mean_quat[0] < 0:
            mean_quat = -mean_quat
        # Swap back to (x, y, z, w)
        mean_quat_xyzw = mean_quat[[1, 2, 3, 0]]
        return self.normalize_quat(mean_quat_xyzw)

    def _quaternion_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> float:
        """Calculate the angular distance between two quaternions."""
        dot_product = torch.abs(torch.dot(q1, q2))
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle = 2 * torch.acos(dot_product) / math.pi
        return angle.item()
