from hoopgn.objects.properties.handlers.ignore_handler import (
    IgnoreValue,
    IgnoreValueConfig,
)
from hoopgn.objects.properties.handlers.one_hot_handler import (
    OneHotValue,
    OneHotValueConfig,
)


STATE_HANDLER_BUILDERS = {
    IgnoreValueConfig: lambda config: IgnoreValue(config),
    OneHotValueConfig: lambda config: OneHotValue(config),
}


def register_state_handler(config_type, builder):
    STATE_HANDLER_BUILDERS[config_type] = builder


def select_state_handler(config):
    builder = STATE_HANDLER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in STATE_HANDLER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
