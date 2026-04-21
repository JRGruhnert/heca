import math
import torch

from hoopgn.networks.layers.property_encoder import PropertyEncoder
from hoopgn.properties.features.conditions.condition import PropertyCondition
from hoopgn.properties.features.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.features.evaluators.threshold_evaluator import ThresholdEvaluator
from hoopgn.properties.features.extractors.calvin_gt_extractor import CalvinGTExtractor
from hoopgn.properties.features.extractors.extractor import PropertyExtractor
from hoopgn.properties.features.normalizers.quaternion_normalizer import (
    QuaternionNormalizer,
)
from hoopgn.properties.features.parameters.quaternion_parameter import (
    QuaternionParameter,
)
from hoopgn.properties.features.rulers.angular_ruler import AngularRuler


class Quaternion:
    label: str = "Quat"
    ruler: AngularRuler.Config = AngularRuler.Config()
    encoder: PropertyEncoder.Config = PropertyEncoder.Config(dim_input=4)
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config(
        ruler=AngularRuler.Config(),
    )
    condition: PropertyCondition.Config = PropertyCondition.Config(
        ruler=AngularRuler.Config(),
        parameter=QuaternionParameter.Config(),
    )
    normalizer: QuaternionNormalizer.Config = QuaternionNormalizer.Config()
    extractor: PropertyExtractor.Config

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
