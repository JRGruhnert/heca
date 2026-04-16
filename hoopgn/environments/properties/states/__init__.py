from hoopgn.environments.properties.states.area_state import AreaState, AreaStateConfig
from hoopgn.environments.properties.states.binary_state import (
    BinaryState,
    BinaryStateConfig,
)
from conf.properties.v2.state import StateConfig, State


_STATE_PROPERTY_BUILDERS = {
    AreaStateConfig: lambda config: AreaState(config),
    BinaryStateConfig: lambda config: BinaryState(config),
}


def register_state_property(config_type, builder):
    _STATE_PROPERTY_BUILDERS[config_type] = builder


def select_state_property(config: StateConfig) -> State:
    builder = _STATE_PROPERTY_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in _STATE_PROPERTY_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
