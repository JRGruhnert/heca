from src.skills.skip.skip_networker import SkipNetworker, SkipNetworkerConfig
from src.skills.skip.skip_operator import SkipOperator, SkipOperatorConfig
from src.skills.tapas.tapas_networker import (
    TapasNetworker,
    TapasNetworkerConfig,
)
from src.skills.tapas.tapas_operator import (
    TapasOperator,
    TapasOperatorConfig,
)


SKILL_OPERATOR_BUILDERS = {
    TapasOperatorConfig: lambda config: TapasOperator(config),
    SkipOperatorConfig: lambda config: SkipOperator(config),
}
SKILL_NETWORKER_BUILDERS = {
    TapasNetworkerConfig: lambda config: TapasNetworker(config),
    SkipNetworkerConfig: lambda config: SkipNetworker(config),
}


def register_skill_operator(config_type, builder):
    SKILL_OPERATOR_BUILDERS[config_type] = builder


def register_skill_networker(config_type, builder):
    SKILL_NETWORKER_BUILDERS[config_type] = builder


def select_skill_operator(config):
    builder = SKILL_OPERATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in SKILL_OPERATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)


def select_skill_networker(config):
    builder = SKILL_NETWORKER_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in SKILL_NETWORKER_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
