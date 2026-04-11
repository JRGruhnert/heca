from hoopgn.objects.properties.features.extractors.calvin_gt_extractor import (
    CalvinGTExtractor,
    CalvinGTExtractorConfig,
)
from hoopgn.objects.properties.features.extractors.calvin_image_extractor import (
    CalvinImageExtractor,
    CalvinImageExtractorConfig,
)


EXTRACTOR_BUILDERS = {
    CalvinGTExtractorConfig: lambda config: CalvinGTExtractor(config),
    CalvinImageExtractorConfig: lambda config: CalvinImageExtractor(config),
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
