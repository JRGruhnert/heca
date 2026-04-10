from src.objects.properties.handlers.validators.area_validator import (
    AreaValidator,
    AreaValidatorConfig,
)
from src.objects.properties.handlers.validators.ignore_validator import (
    IgnoreValidator,
    IgnoreValidatorConfig,
)


STATE_VALIDATOR_BUILDERS = {
    AreaValidatorConfig: lambda config: AreaValidator(config),
    IgnoreValidatorConfig: lambda config: IgnoreValidator(config),
}


def register_state_validator(config_type, builder):
    STATE_VALIDATOR_BUILDERS[config_type] = builder


def select_state_validator(config):
    builder = STATE_VALIDATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in STATE_VALIDATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
