from hoopgn.objects.properties.features.validators.area_validator import (
    AreaValidator,
    AreaValidatorConfig,
)
from hoopgn.objects.properties.features.validators.ignore_validator import (
    IgnoreValidator,
    IgnoreValidatorConfig,
)


STATE_VALIDATOR_BUILDERS = {
    AreaValidatorConfig: lambda config: AreaValidator(config),
    IgnoreValidatorConfig: lambda config: IgnoreValidator(config),
}


def register_state_validator(config_type, builder):
    STATE_VALIDATOR_BUILDERS[config_type] = builder


def select_property_validator(config):
    builder = STATE_VALIDATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in STATE_VALIDATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
