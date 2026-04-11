from hoopgn.objects.properties.features.modifiers.one_hot_modifier import (
    OneHotModifier,
    OneHotModifierConfig,
)
from hoopgn.objects.properties.features.modifiers.skip_modifier import (
    SkipModifier,
    SkipModifierConfig,
)


_PROPERTY_MODIFIER_BUILDERS = {
    SkipModifierConfig: lambda config: SkipModifier(config),
    OneHotModifierConfig: lambda config: OneHotModifier(config),
}


def register_property_modifier(config_type, builder):
    _PROPERTY_MODIFIER_BUILDERS[config_type] = builder


def select_property_modifier(config):
    builder = _PROPERTY_MODIFIER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in _PROPERTY_MODIFIER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
