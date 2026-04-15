from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from conf.entities import properties_to_entities
from hoopgn import logger
from hoopgn.entities.entity import EntityConfig
from hoopgn.properties.property import PropertyConfig
from hoopgn.skills.skill import SkillConfig
from hoopgn.operators.tapas_operator import TapasOperator


@dataclass
class HoopGNRunnerConfig:
    skills: list[SkillConfig]
    properties: list[PropertyConfig]
    entities: list[EntityConfig] = field(init=False)

    def __post_init__(self):
        assert len(self.skills) > 0, "At least one skill must be provided."
        assert len(self.properties) > 0, "At least one property must be provided."
        logger.warning(
            f"HoopGN multiple version support should be removed in the future."
        )
        self.entities = properties_to_entities(properties=self.properties)
        for skill in self.skills:
            if isinstance(skill.operator, TapasOperator):
                logger.warning(
                    f"Skill '{skill.label}' is a Tapas skill. Performing automated property assignment."
                )
                skill.operator.config.properties = self.properties


class HoopGNRunner(ABC):
    def __init__(self, config: HoopGNRunnerConfig):
        self.config = config

    @abstractmethod
    def run(self):
        raise NotImplementedError()
