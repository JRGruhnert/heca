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
    def inv(q):
        """q: [..., 4] unit quaternion. Returns inverse."""
        # For unit quaternion, inverse is just conjugate
        return np.concatenate([q[..., 0:1], -q[..., 1:]], axis=-1)

    @staticmethod
    def mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamiltonian product. q1, q2: [..., 4]."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.stack([w, x, y, z], axis=-1)

    @staticmethod
    def log_map(q: np.ndarray) -> np.ndarray:
        """q: [..., 4] unit quaternion. Returns axis-angle vector [..., 3]."""
        w = np.clip(q[..., 0], -1.0, 1.0)  # Clamp for numerical stability
        xyz = q[..., 1:]
        norm = np.linalg.norm(xyz, axis=-1, keepdims=True)

        # Avoid division by zero
        safe_norm = np.where(norm < 1e-12, 1.0, norm)
        angle = 2 * np.arctan2(safe_norm.squeeze(-1), w)
        axis = xyz / safe_norm

        return angle[..., np.newaxis] * axis  # [..., 3]

    @staticmethod
    def exp(delta_rot: np.ndarray) -> np.ndarray:
        """
        Exponential map from axis-angle (3D) to unit quaternion.
        delta_rot: [..., 3] axis-angle vector.
        Returns: [..., 4] unit quaternion.
        """
        angle = np.linalg.norm(delta_rot, axis=-1, keepdims=True)  # [..., 1]
        # Avoid division by zero
        safe_angle = np.where(angle < 1e-12, 1.0, angle)
        axis = delta_rot / safe_angle

        half_angle = angle / 2.0
        w = np.cos(half_angle).squeeze(-1)  # [..., ]
        xyz = np.sin(half_angle) * axis  # [..., 3]
        return np.concatenate([w[..., np.newaxis], xyz], axis=-1)  # [..., 4]
