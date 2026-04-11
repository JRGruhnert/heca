from hoopgn.objects.properties.features.parameters.parameter import StateParameter
from hoopgn.objects.properties.features.parameters.binary_parameter import (
    BinaryParameter,
    BinaryParameterConfig,
)
from hoopgn.objects.properties.features.parameters.euclidean_parameter import (
    EuclideanParameter,
    EuclideanParameterConfig,
)
from hoopgn.objects.properties.features.parameters.flip_parameter import (
    FlipParameter,
    FlipParameterConfig,
)
from hoopgn.objects.properties.features.parameters.ignore_parameter import (
    IgnoreParameter,
    IgnoreParameterConfig,
)
from hoopgn.objects.properties.features.parameters.quaternion_parameter import (
    QuaternionParameter,
    QuaternionParameterConfig,
)
from hoopgn.objects.properties.features.parameters.scalar_parameter import (
    ScalarParameter,
    ScalarParameterConfig,
)


STATE_PARAMETER_BUILDERS = {
    BinaryParameterConfig: lambda config: BinaryParameter(config),
    EuclideanParameterConfig: lambda config: EuclideanParameter(config),
    FlipParameterConfig: lambda config: FlipParameter(config),
    IgnoreParameterConfig: lambda config: IgnoreParameter(config),
    QuaternionParameterConfig: lambda config: QuaternionParameter(config),
    ScalarParameterConfig: lambda config: ScalarParameter(config),
}


def register_state_parameter(config_type, builder):
    STATE_PARAMETER_BUILDERS[config_type] = builder


def select_state_parameter(config) -> StateParameter:
    builder = STATE_PARAMETER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in STATE_PARAMETER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
