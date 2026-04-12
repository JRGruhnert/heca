from abc import abstractmethod
from dataclasses import dataclass

from hoopgn.skills.skill import Skill
from hoopgn.runners.runner import HoopGNRunner, HoopGNRunnerConfig


@dataclass
class SkillRunnerConfig(HoopGNRunnerConfig):
    pass


class SkillRunner(HoopGNRunner):
    def __init__(self, config: SkillRunnerConfig):
        super().__init__(config)
        self.config = config

    def __call__(self):
        for skill in self.config.skills:
            self.run(Skill(skill))

    @abstractmethod
    def run(self, skill: Skill):
        raise NotImplementedError()
