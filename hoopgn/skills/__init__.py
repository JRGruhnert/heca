from hoopgn.skills.skill import Skill, SkillConfig
from hoopgn.skills.skill_evaluator import SkillEvaluator, SkillEvaluatorConfig
from hoopgn.skills.skill_networker import SkillNetworker, SkillNetworkerConfig
from hoopgn.operators.skill_operator import SkillOperator, SkillOperatorConfig
from hoopgn.skills.leafs.skip.skip_networker import SkipNetworker, SkipNetworkerConfig
from hoopgn.skills.leafs.skip.skip_operator import SkipOperator, SkipOperatorConfig

from hoopgn.operators.tapas_operator import (
    TapasOperator,
    TapasOperatorConfig,
)


SKILL_OPERATOR_BUILDERS = {
    TapasOperatorConfig: lambda config: TapasOperator(config),
    SkipOperatorConfig: lambda config: SkipOperator(config),
}
SKILL_NETWORKER_BUILDERS = {
    SkipNetworkerConfig: lambda config: SkipNetworker(config),
}
SKILL_EVALUATOR_BUILDERS = {}


def register_skill_operator(config_type, builder):
    SKILL_OPERATOR_BUILDERS[config_type] = builder


def register_skill_networker(config_type, builder):
    SKILL_NETWORKER_BUILDERS[config_type] = builder


def register_skill_evaluator(config_type, builder):
    SKILL_EVALUATOR_BUILDERS[config_type] = builder


def select_skill_operator(config: SkillOperatorConfig) -> SkillOperator:
    builder = SKILL_OPERATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in SKILL_OPERATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)


def select_skill_networker(config: SkillNetworkerConfig) -> SkillNetworker:
    builder = SKILL_NETWORKER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in SKILL_NETWORKER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)


def select_skill_evaluator(config: SkillEvaluatorConfig) -> SkillEvaluator:
    builder = SKILL_EVALUATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in SKILL_EVALUATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)


from hoopgn.skills.branches.manual.human import ManualSkill, ManualSkillConfig
from hoopgn.skills.branches.hoopgn.hoopgn_skill import HoopGNSkill, HoopGNSkillConfig


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
