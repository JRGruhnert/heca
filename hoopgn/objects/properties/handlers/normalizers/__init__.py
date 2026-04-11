from hoopgn.objects.properties.handlers.normalizers.boundary_normalizer import (
    BoundaryNormalizer,
    BoundaryNormalizerConfig,
)
from hoopgn.objects.properties.handlers.normalizers.ignore_normalizer import (
    IgnoreNormalizer,
    IgnoreNormalizerConfig,
)
from hoopgn.objects.properties.handlers.normalizers.quaternion_normalizer import (
    QuaternionNormalizer,
    QuaternionNormalizerConfig,
)


STATE_NORMALIZER_BUILDERS = {
    BoundaryNormalizerConfig: lambda config: BoundaryNormalizer(config),
    IgnoreNormalizerConfig: lambda config: IgnoreNormalizer(config),
    QuaternionNormalizerConfig: lambda config: QuaternionNormalizer(config),
}


def register_state_normalizer(config_type, builder):
    STATE_NORMALIZER_BUILDERS[config_type] = builder


def select_state_normalizer(config):
    builder = STATE_NORMALIZER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in STATE_NORMALIZER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
