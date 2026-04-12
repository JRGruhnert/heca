from abc import ABC, abstractmethod
from dataclasses import dataclass

from hoopgn.entities.entity import Entity, EntityConfig
from hoopgn.properties.property import Property, PropertyConfig
from hoopgn.skills.skill import Skill, SkillConfig


@dataclass
class HoopGNRunnerConfig:
    skills: list[SkillConfig]
    entities: list[EntityConfig]
    properties: list[PropertyConfig]


class HoopGNRunner(ABC):
    def __init__(self, config: HoopGNRunnerConfig):
        self.config = config
        self.skills = [Skill(c) for c in config.skills]
        self.entities = [Entity(c) for c in config.entities]
        self.properties = [Property(c) for c in config.properties]

    @abstractmethod
    def __call__(self):
        raise NotImplementedError()
