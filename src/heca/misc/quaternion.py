import math
import numpy as np
import torch


class Quaternion:

    @staticmethod
    def normalize_quat(x: torch.Tensor) -> torch.Tensor:
        """Normalize quaternion and ensure positive w component."""
        nx = x / torch.linalg.norm(x)
        if nx[3] < 0:
            return -nx
        return nx

    @staticmethod
    def mean(quaternions: torch.Tensor) -> torch.Tensor:
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
        return Quaternion.normalize_quat(mean_quat_xyzw)

    @staticmethod
    def _quaternion_distance(q1: torch.Tensor, q2: torch.Tensor) -> float:
        """Calculate the angular distance between two quaternions."""
        dot_product = torch.abs(torch.dot(q1, q2))
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle = 2 * torch.acos(dot_product) / math.pi
        return angle.item()

    @staticmethod
    def quat_to_6d(w: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Converts a quaternion (w, x, y, z) to the 6D continuous rotation
        representation (first two columns of the rotation matrix).
        This avoids the q/-q ambiguity and provides a smoother gradient landscape.
        """
        # Rotation matrix from quaternion (w, x, y, z)
        # R00 = 1 - 2*y^2 - 2*z^2
        # R10 = 2*x*y + 2*z*w
        # R20 = 2*x*z - 2*y*w
        # R01 = 2*x*y - 2*z*w
        # R11 = 1 - 2*x^2 - 2*z^2
        # R21 = 2*y*z + 2*x*w

        r00 = 1.0 - 2.0 * y * y - 2.0 * z * z
        r10 = 2.0 * x * y + 2.0 * z * w
        r20 = 2.0 * x * z - 2.0 * y * w

        r01 = 2.0 * x * y - 2.0 * z * w
        r11 = 1.0 - 2.0 * x * x - 2.0 * z * z
        r21 = 2.0 * y * z + 2.0 * x * w

        # Flatten the first two columns: [col0, col1] -> 6 values
        return np.array([r00, r10, r20, r01, r11, r21])
