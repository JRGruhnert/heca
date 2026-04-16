from dataclasses import dataclass
from hoopgn.environments.properties.property import PropertyConfig


@dataclass(kw_only=True)
class EntityConfig:
    id: int
    label: str
    # Properties
    # domain: DomainPropertyConfig
    position: PropertyConfig
    rotation: PropertyConfig
    state: PropertyConfig
    # Features
    # ruler: EntityRulerConfig
    # encoder: EntityEncoderConfig
    # condition: EntityConditionConfig
    # evaluator: EntityEvaluatorConfig
    # normalizer: EntityNormalizerConfig
    # extractor: EntityExtractorConfig
    # modifier: EntityModifierConfig
    # validator: EntityValidatorConfig


class Entity:
    def __init__(
        self,
        config: EntityConfig,
    ):
        self.config = config
