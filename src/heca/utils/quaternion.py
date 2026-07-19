import numpy as np


class Quaternion:

    @staticmethod
    def identity() -> np.ndarray:
        """Returns the identity quaternion [1, 0, 0, 0], representing zero rotation."""
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion from [x, y, z, w] order to [w, x, y, z] order.
        Supports batched inputs of shape [..., 4].
        """
        return q[..., [3, 0, 1, 2]]

    @staticmethod
    def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion from [w, x, y, z] order to [x, y, z, w] order.
        Supports batched inputs of shape [..., 4].
        """
        return q[..., [1, 2, 3, 0]]

    @staticmethod
    def normalize(q: np.ndarray, eps: float = 1e-15) -> np.ndarray:
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        nx = q / np.where(norm < eps, 1.0, norm)
        # Ensure w >= 0 by flipping sign of whole quaternion where w < 0
        flip = nx[..., 0:1] < 0  # (..., 1)
        return np.where(flip, -nx, nx)

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
    def _safe_div(x, y, eps=1e-12):
        return x / np.where(y < eps, 1.0, y)

    @staticmethod
    def log_map(q: np.ndarray) -> np.ndarray:
        """q: [..., 4] unit quaternion. Returns axis-angle vector [..., 3]."""
        w = np.clip(q[..., 0], -1.0, 1.0)  # Clamp for numerical stability
        xyz = q[..., 1:]
        norm = np.linalg.norm(xyz, axis=-1, keepdims=True)

        # Avoid division by zero
        safe_norm = np.where(norm < 1e-12, 1.0, norm)
        axis = xyz / safe_norm
        angle = 2 * np.arctan2(safe_norm, w[..., np.newaxis])
        return angle * axis  # (...,1) * (...,3) -> (...,3)

    @staticmethod
    def exp(q: np.ndarray) -> np.ndarray:
        """
        Exponential map from axis-angle (3D) to unit quaternion.
        delta_rot: [..., 3] axis-angle vector.
        Returns: [..., 4] unit quaternion.
        """
        angle = np.linalg.norm(q, axis=-1, keepdims=True)  # [..., 1]
        # Avoid division by zero
        safe_angle = np.where(angle < 1e-12, 1.0, angle)
        axis = q / safe_angle

        half_angle = angle / 2.0
        w = np.cos(half_angle).squeeze(-1)  # [..., ]
        xyz = np.sin(half_angle) * axis  # [..., 3]
        return np.concatenate([w[..., np.newaxis], xyz], axis=-1)  # [..., 4]
