from abc import ABC, abstractmethod
from dataclasses import dataclass

from hoopgn.entities.entity import EntityConfig
from hoopgn.properties.property import PropertyConfig
from hoopgn.skills.skill import SkillConfig


@dataclass
class HoopGNRunnerConfig:
    skills: list[SkillConfig]
    entities: list[EntityConfig]
    properties: list[PropertyConfig]


class HoopGNRunner(ABC):
    def __init__(self, config: HoopGNRunnerConfig):
        self.config = config

    @abstractmethod
    def run(self):
        raise NotImplementedError()
