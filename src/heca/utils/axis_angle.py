import numpy as np


class AxisAngle:

    @staticmethod
    def exp(a: np.ndarray) -> np.ndarray:
        """
        Exponential map from axis-angle (3D) to unit quaternion.
        delta_rot: [..., 3] axis-angle vector.
        Returns: [..., 4] unit quaternion.
        """
        angle = np.linalg.norm(a, axis=-1, keepdims=True)  # [..., 1]
        # Avoid division by zero
        safe_angle = np.where(angle < 1e-12, 1.0, angle)
        axis = a / safe_angle

        half_angle = angle / 2.0
        w = np.cos(half_angle).squeeze(-1)  # [..., ]
        xyz = np.sin(half_angle) * axis  # [..., 3]
        return np.concatenate([w[..., np.newaxis], xyz], axis=-1)  # [..., 4]
