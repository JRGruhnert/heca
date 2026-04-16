from hoopgn.environments.properties.features.parameters.parameter import (
    PropertyParameter,
)
from hoopgn.environments.properties.features.parameters.binary_parameter import (
    BinaryParameter,
    BinaryParameterConfig,
)
from hoopgn.environments.properties.features.parameters.euclidean_parameter import (
    EuclideanParameter,
    EuclideanParameterConfig,
)
from hoopgn.environments.properties.features.parameters.flip_parameter import (
    FlipParameter,
    FlipParameterConfig,
)
from hoopgn.environments.properties.features.parameters.ignore_parameter import (
    IgnoreParameter,
    IgnoreParameterConfig,
)
from hoopgn.environments.properties.features.parameters.quaternion_parameter import (
    QuaternionParameter,
    QuaternionParameterConfig,
)
from hoopgn.environments.properties.features.parameters.scalar_parameter import (
    ScalarParameter,
    ScalarParameterConfig,
)


_PROPERTY_PARAMETER_BUILDERS = {
    BinaryParameterConfig: lambda config: BinaryParameter(config),
    EuclideanParameterConfig: lambda config: EuclideanParameter(config),
    FlipParameterConfig: lambda config: FlipParameter(config),
    IgnoreParameterConfig: lambda config: IgnoreParameter(config),
    QuaternionParameterConfig: lambda config: QuaternionParameter(config),
    ScalarParameterConfig: lambda config: ScalarParameter(config),
}


def register_property_parameter(config_type, builder):
    _PROPERTY_PARAMETER_BUILDERS[config_type] = builder


def select_property_parameter(config) -> PropertyParameter:
    builder = _PROPERTY_PARAMETER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in _PROPERTY_PARAMETER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
