from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from conf.entities import properties_to_entities
from hoopgn import logger
from hoopgn.environments.entities.entity import EntityConfig
from hoopgn.environments.properties.property import PropertyConfig
from hoopgn.agents.agent import SkillConfig
from hoopgn.policies.tapas_policy import TapasPolicy


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
            if isinstance(skill.policy, TapasPolicy):
                logger.warning(
                    f"Skill '{skill.label}' is a Tapas skill. Performing automated property assignment."
                )
                skill.policy.config.properties = self.properties


class HoopGNRunner(ABC):
    def __init__(self, config: HoopGNRunnerConfig):
        self.config = config

    @abstractmethod
    def run(self):
        raise NotImplementedError()
