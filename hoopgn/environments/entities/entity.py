from dataclasses import dataclass
from hoopgn.base import ConfigurableClass
from hoopgn.environments.properties.property import Property


class Entity(ConfigurableClass):
    @dataclass(kw_only=True)
    class EntityConfig:
        id: int
        label: str
        # Properties
        # domain: DomainPropertyConfig
        position: Property.Config
        rotation: Property.Config
        state: Property.Config
        # Features
        # ruler: EntityRulerConfig
        # encoder: EntityEncoderConfig
        # condition: EntityConditionConfig
        # evaluator: EntityEvaluatorConfig
        # normalizer: EntityNormalizerConfig
        # extractor: EntityExtractorConfig
        # modifier: EntityModifierConfig
        # validator: EntityValidatorConfig

    def __init__(
        self,
        config: EntityConfig,
    ):
        self.config = config
