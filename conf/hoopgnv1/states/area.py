from dataclasses import dataclass, field

from src.networks.layers.classifiers.state_classifier import StateClassifierConfig
from src.networks.layers.encoder import StateEncoderConfig
from src.objects.properties.area import AreaConfig

from src.objects.properties.handlers.validators.validator import StateValidatorConfig
from src.objects.properties.handlers.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from src.objects.properties.handlers.parameters.euclidean_parameter import (
    EuclideanParameterConfig,
)
from src.objects.properties.handlers.rulers.euclidean_ruler import (
    EuclideanRulerConfig,
)
from src.objects.properties.handlers.normalizers.boundary_normalizer import (
    AreaNormalizerConfig,
)

from src.objects.properties.handlers.normalizers.normalizer import NormalizerConfig
from src.objects.properties.handlers.validators.area_validator import (
    AreaValidatorConfig,
)
from src.objects.properties.handlers.handler import ValueHandlerConfig
from src.objects.properties.handlers.one_hot_handler import OneHotValueConfig
from src.objects.properties.property import PropertyConfig
from src.objects.properties.property_condition import PropertyConditionConfig


@dataclass
class CalvinAreaConfig(AreaConfig):
    label: str = "AreaEuler"
    values: list[str] = field(
        default_factory=lambda: ["table", "drawer_open", "drawer_closed", "drawer"]
    )
    spawn_surfaces: dict = field(
        default_factory=lambda: {
            "table": [[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]],
            "drawer_open": [[0.04, -0.35, 0.38], [0.30, -0.21, 0.38]],
            "drawer_closed": [[0.04, -0.16, 0.38], [0.30, -0.03, 0.38]],
        }
    )
    eval_surfaces: dict = field(
        default_factory=lambda: {
            "table": [[-0.02, -0.17, 0.44], [0.32, -0.01, 0.54]],
            "drawer_open": [[0.02, -0.37, 0.34], [0.32, -0.23, 0.44]],
            "drawer_closed": [[0.02, -0.18, 0.34], [0.32, -0.00, 0.44]],
        }
    )


@dataclass
class CalvinAreaStateConfig(PropertyConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="AreaEuler",
        dim_input=6,
    )
    normalizer: NormalizerConfig = AreaNormalizerConfig()
    ruler: EuclideanRulerConfig = EuclideanRulerConfig()
    evaluator: StateEvaluatorConfig = StateEvaluatorConfig()
    condition: PropertyConditionConfig = PropertyConditionConfig(
        ruler=EuclideanRulerConfig(),
        parameter=EuclideanParameterConfig(),
    )
    validator: StateValidatorConfig = AreaValidatorConfig(
        area=CalvinAreaConfig(
            classifier=StateClassifierConfig(),
        )
    )
    preencoder: ValueHandlerConfig = OneHotValueConfig(
        state=CalvinAreaConfig(
            classifier=StateClassifierConfig(),
        ),
    )
