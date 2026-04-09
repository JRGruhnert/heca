from src.objects.properties.value_handler.parameters.binary_parameter import (
    BinaryParameter,
    BinaryParameterConfig,
)
from src.objects.properties.value_handler.parameters.euclidean_parameter import (
    EuclideanParameter,
    EuclideanParameterConfig,
)


STATE_RULER_BUILDERS = {
    BinaryParameterConfig: lambda config: BinaryParameter(config),
    EuclideanParameterConfig: lambda config: EuclideanParameter(config),
}


def register_state_parameter(config_type, builder):
    STATE_RULER_BUILDERS[config_type] = builder


def select_state_parameter(config):
    builder = STATE_RULER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in STATE_RULER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
