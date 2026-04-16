from dataclasses import dataclass


from hoopgn.agents.agent import SkillConfig


@dataclass(kw_only=True)
class SkipConfig(SkillConfig):
    pass
