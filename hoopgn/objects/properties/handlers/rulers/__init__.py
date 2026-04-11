from hoopgn.objects.properties.handlers.rulers.angular_ruler import (
    AngularRuler,
    AngularRulerConfig,
)
from hoopgn.objects.properties.handlers.rulers.binary_ruler import (
    BinaryRuler,
    BinaryRulerConfig,
)
from hoopgn.objects.properties.handlers.rulers.euclidean_ruler import (
    EuclideanRuler,
    EuclideanRulerConfig,
)
from hoopgn.objects.properties.handlers.rulers.flip_ruler import (
    FlipRuler,
    FlipRulerConfig,
)


STATE_RULER_BUILDERS = {
    BinaryRulerConfig: lambda config: BinaryRuler(config),
    EuclideanRulerConfig: lambda config: EuclideanRuler(config),
    FlipRulerConfig: lambda config: FlipRuler(config),
    AngularRulerConfig: lambda config: AngularRuler(config),
}


def register_state_ruler(config_type, builder):
    STATE_RULER_BUILDERS[config_type] = builder


def select_state_ruler(config):
    builder = STATE_RULER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in STATE_RULER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
