from hoopgn.properties.features.normalizers.boundary_normalizer import (
    BoundaryNormalizer,
    BoundaryNormalizerConfig,
)
from hoopgn.properties.features.normalizers.ignore_normalizer import (
    IgnoreNormalizer,
    IgnoreNormalizerConfig,
)
from hoopgn.properties.features.normalizers.quaternion_normalizer import (
    QuaternionNormalizer,
    QuaternionNormalizerConfig,
)


_PROPERTY_NORMALIZER_BUILDERS = {
    BoundaryNormalizerConfig: lambda config: BoundaryNormalizer(config),
    IgnoreNormalizerConfig: lambda config: IgnoreNormalizer(config),
    QuaternionNormalizerConfig: lambda config: QuaternionNormalizer(config),
}


def register_property_normalizer(config_type, builder):
    _PROPERTY_NORMALIZER_BUILDERS[config_type] = builder


def select_property_normalizer(config):
    builder = _PROPERTY_NORMALIZER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in _PROPERTY_NORMALIZER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
