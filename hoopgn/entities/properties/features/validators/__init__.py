from hoopgn.entities.properties.features.validators.area_validator import (
    AreaValidator,
    AreaValidatorConfig,
)
from hoopgn.entities.properties.features.validators.skip_validator import (
    SkipValidator,
    SkipValidatorConfig,
)


_PROPERTY_VALIDATOR_BUILDERS = {
    AreaValidatorConfig: lambda config: AreaValidator(config),
    SkipValidatorConfig: lambda config: SkipValidator(config),
}


def register_property_validator(config_type, builder):
    _PROPERTY_VALIDATOR_BUILDERS[config_type] = builder


def select_property_validator(config):
    builder = _PROPERTY_VALIDATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in _PROPERTY_VALIDATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
