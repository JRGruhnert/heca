from hoopgn.objects.properties.features.rulers.angular_ruler import (
    AngularRuler,
    AngularRulerConfig,
)
from hoopgn.objects.properties.features.rulers.binary_ruler import (
    BinaryRuler,
    BinaryRulerConfig,
)
from hoopgn.objects.properties.features.rulers.euclidean_ruler import (
    EuclideanRuler,
    EuclideanRulerConfig,
)
from hoopgn.objects.properties.features.rulers.flip_ruler import (
    FlipRuler,
    FlipRulerConfig,
)


_PROPERTY_RULER_BUILDERS = {
    BinaryRulerConfig: lambda config: BinaryRuler(config),
    EuclideanRulerConfig: lambda config: EuclideanRuler(config),
    FlipRulerConfig: lambda config: FlipRuler(config),
    AngularRulerConfig: lambda config: AngularRuler(config),
}


def register_property_ruler(config_type, builder):
    _PROPERTY_RULER_BUILDERS[config_type] = builder


def select_property_ruler(config):
    builder = _PROPERTY_RULER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in _PROPERTY_RULER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
