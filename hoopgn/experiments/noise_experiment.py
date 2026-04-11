from collections.abc import Sequence
from dataclasses import dataclass, field
import math
import random

from loguru import logger
from hoopgn.skills.skip.skip_leaf import SkipConfig
from hoopgn.experiments.experiment import Experiment, ExperimentConfig
from hoopgn.skills.skill import Skill, SkillConfig
from hoopgn.storage import select_skills


@dataclass(kw_only=True)
class NoiseExperimentConfig(ExperimentConfig):
    p_empty: float
    p_rand: float
    min_steps: int
    skills: Sequence[SkillConfig] = field(default_factory=list)

    def __post_init__(self):
        assert 0.0 <= self.p_empty <= 1.0, "p_empty must be between 0 and 1"
        assert 0.0 <= self.p_rand <= 1.0, "p_rand must be between 0 and 1"
        assert (
            self.p_empty + self.p_rand <= 1.0
        ), "The sum of p_empty and p_rand must be less than or equal to 1"
        assert self.min_steps > 0, "min_steps must be greater than 0"


class NoiseExperiment(Experiment):
    def __init__(self, config: NoiseExperimentConfig):
        super().__init__(config)
        self.config = config

        self.max_allowed_steps = math.ceil(
            self.config.min_steps
            + self.config.min_steps * self.config.p_empty
            + self.config.min_steps * self.config.p_rand
        )
        self.skills = select_skills(config.skills)
        self.skip_skill = Skill(config=SkipConfig())
        logger.info(
            "Noise Experiment Values:\n"
            f"No. Skills:  {self.config.min_steps}\n"
            f"No. Steps:   {self.max_allowed_steps}\n"
            f"% Skip:      {self.config.p_empty}\n"
            f"% Random:    {self.config.p_rand}\n"
        )

    def modify(self, skill: Skill) -> Skill:
        sample = random.random()
        if sample < self.config.p_empty:
            selected_skill = self.skip_skill
            logger.info("Taking Empty Step")
        elif sample < self.config.p_empty + self.config.p_rand:
            selected_skill = random.choice(self.skills)
            logger.info("Taking Random Step")
        else:
            selected_skill = skill

        selected_skill.reset(self.goal)

        return selected_skill

    def metadata(self) -> dict:
        return {
            "p_empty": self.config.p_empty,
            "p_rand": self.config.p_rand,
            "max_allowed_steps": self.max_allowed_steps,
        }
