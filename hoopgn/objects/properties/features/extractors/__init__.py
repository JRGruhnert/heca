from hoopgn.objects.properties.features.extractors.gt_extractor import (
    GTExtractor,
    GTExtractorConfig,
)
from hoopgn.objects.properties.features.extractors.image_extractor import (
    ImageExtractor,
    ImageExtractorConfig,
)


EXTRACTOR_BUILDERS = {
    GTExtractorConfig: lambda config: GTExtractor(config),
    ImageExtractorConfig: lambda config: ImageExtractor(config),
}


def register_extractor(config_type, builder):
    EXTRACTOR_BUILDERS[config_type] = builder


def select_property_extractor(config):
    builder = EXTRACTOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in EXTRACTOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
