from dataclasses import dataclass

from hoopgn.entities.entity import EntityConfig

from hoopgn.runners.runner import HoopGNRunner, HoopGNRunnerConfig
from hoopgn.skills.skill import SkillConfig


@dataclass
class ExplainRunnerConfig(HoopGNRunnerConfig):
    skills: list[SkillConfig]
    entities: list[EntityConfig]


class ExplainRunner(HoopGNRunner):
    def __init__(self, config: ExplainRunnerConfig):
        super().__init__(config)
        self.config = config

    def run(self):
        raise NotImplementedError()
