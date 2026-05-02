from heca.agents.experts.legacy.parameters.binary import BinaryParameter
from heca.agents.experts.legacy.parameters.euclidean import EuclideanParameter
from heca.agents.experts.legacy.parameters.flip import FlipParameter
from heca.agents.experts.legacy.parameters.parameter import PropertyParameter
from heca.agents.experts.legacy.parameters.quaternion import QuaternionParameter
from heca.agents.experts.legacy.parameters.range import RangeParameter
from heca.properties.normalizers.boundary import BoundaryNormalizer

extractors: dict[str, PropertyParameter.Config] = {
    "ee_position": EuclideanParameter.Config(),
    "ee_rotation": QuaternionParameter.Config(),
    "ee_scalar": BinaryParameter.Config(),
    "slide_position": EuclideanParameter.Config(),
    "slide_rotation": QuaternionParameter.Config(),
    "drawer_position": EuclideanParameter.Config(),
    "drawer_rotation": QuaternionParameter.Config(),
    "button_position": EuclideanParameter.Config(),
    "button_rotation": QuaternionParameter.Config(),
    "button_scalar": FlipParameter.Config(),
    "led_position": EuclideanParameter.Config(),
    "led_rotation": QuaternionParameter.Config(),
    "block_red_position": EuclideanParameter.Config(),
    "block_red_scalar": BinaryParameter.Config(),
    "block_blue_position": EuclideanParameter.Config(),
    "block_blue_scalar": BinaryParameter.Config(),
    "block_pink_position": EuclideanParameter.Config(),
    "block_pink_scalar": BinaryParameter.Config(),
    "block_red_rotation": QuaternionParameter.Config(),
    "block_blue_rotation": QuaternionParameter.Config(),
    "block_pink_rotation": QuaternionParameter.Config(),
    "slide_scalar": RangeParameter.Config(
        normalizer=BoundaryNormalizer.Config(
            lower=[0.0],
            upper=[0.28],
        ),
        threshold=0.05,
    ),
    "drawer_scalar": RangeParameter.Config(
        normalizer=BoundaryNormalizer.Config(
            lower=[0.0],
            upper=[0.22],
        ),
        threshold=0.05,
    ),
}
