from hoopgn.observation.converters.converter import Converter
from hoopgn.observation.converters.tapas import (
    TapasConverter,
    TapasConverterConfig,
)


OBSERVATION_CONVERTER_BUILDERS = {
    TapasConverterConfig: lambda config: TapasConverter(config),
}


def register_observation_converter(config_type, builder):
    OBSERVATION_CONVERTER_BUILDERS[config_type] = builder


def select_observation_converter(config) -> Converter:
    builder = OBSERVATION_CONVERTER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in OBSERVATION_CONVERTER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
