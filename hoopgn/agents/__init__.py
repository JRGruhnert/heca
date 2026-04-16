from hoopgn.agents.agent import Skill, SkillConfig

from hoopgn.agents.branches.human_agent import ManualSkill, ManualSkillConfig
from hoopgn.agents.branches.hoopgn_agent import HoopGNSkill, HoopGNSkillConfig


SKILL_BUILDERS = {
    HoopGNSkillConfig: lambda config: HoopGNSkill(config),
    ManualSkillConfig: lambda config: ManualSkill(config),
}


def register_skill(config_type, builder):
    SKILL_BUILDERS[config_type] = builder


def select_skill(config) -> Skill:
    builder = SKILL_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in SKILL_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)


def select_skills(configs: list[SkillConfig]) -> list[Skill]:
    return [select_skill(config) for config in configs]
