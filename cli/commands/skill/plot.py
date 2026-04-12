from dataclasses import dataclass

from hoopgn.skills.skill import SkillConfig


@dataclass
class SkillPlotConfig:
    skills: list[SkillConfig]
