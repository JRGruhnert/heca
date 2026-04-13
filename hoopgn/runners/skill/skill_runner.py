from abc import abstractmethod
from dataclasses import dataclass

from hoopgn.skills.skill import Skill, SkillConfig
from hoopgn.runners.runner import HoopGNRunner, HoopGNRunnerConfig


@dataclass
class SkillRunnerConfig(HoopGNRunnerConfig):
    skill: SkillConfig | None


class SkillRunner(HoopGNRunner):
    def __init__(self, config: SkillRunnerConfig):
        super().__init__(config)
        self.config = config
        self.skills = [Skill(skill_config) for skill_config in config.skills]
        self.skill = Skill(config.skill) if config.skill else None
        self.skills_by_name = {skill.config.label: skill for skill in self.skills}

    def run(self):
        if self.config.skill:
            if self.skill:
                self.skill_run(self.skill)
            else:
                raise ValueError(f"Skill '{self.config.skill}' not found in skills.")
        else:
            for skill in self.skills:
                self.skill_run(skill)

    @abstractmethod
    def skill_run(self, skill: Skill):
        raise NotImplementedError()
