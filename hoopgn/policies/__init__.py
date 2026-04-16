from hoopgn.policies.tapas_policy import (
    TapasPolicy,
    TapasOperatorConfig,
)
from hoopgn.policies.branch_policy import BranchPolicy, BranchPolicyConfig
from hoopgn.agents.leafs.skip.skip_operator import SkipOperator, SkipOperatorConfig


SKILL_OPERATOR_BUILDERS = {
    TapasOperatorConfig: lambda config: TapasPolicy(config),
    SkipOperatorConfig: lambda config: SkipOperator(config),
}


def register_skill_operator(config_type, builder):
    SKILL_OPERATOR_BUILDERS[config_type] = builder


def select_operator(config: BranchPolicyConfig) -> BranchPolicy:
    builder = SKILL_OPERATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in SKILL_OPERATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
