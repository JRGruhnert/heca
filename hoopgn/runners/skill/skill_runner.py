from abc import abstractmethod
from dataclasses import dataclass

from hoopgn import logger
from hoopgn.agents.agent import Skill, SkillConfig
from hoopgn.runners.runner import HoopGNRunner, HoopGNRunnerConfig
from hoopgn.policies.tapas_policy import TapasPolicy


@dataclass
class SkillRunnerConfig(HoopGNRunnerConfig):
    skill: SkillConfig | None

    def __post_init__(self):
        logger.warning(
            f"SkillRunner multiple version support should be removed in the future."
        )
        super().__post_init__()
        if self.skill and isinstance(self.skill.policy, TapasPolicy):
            self.skill.policy.config.properties = self.properties


class SkillRunner(HoopGNRunner):
    def __init__(self, config: SkillRunnerConfig):
        super().__init__(config)
        self.config = config
        self.skills = [Skill(cfg) for cfg in config.skills]

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
