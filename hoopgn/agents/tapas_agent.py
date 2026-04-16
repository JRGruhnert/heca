from dataclasses import dataclass, field

from hoopgn.policies.branch_policy import BranchPolicyConfig
from hoopgn.registry import SkillIdent
from hoopgn.agents.leafs.leaf_agent import LeafConfig
from hoopgn.policies.tapas_policy import TapasOperatorConfig


@dataclass(kw_only=True)
class TapasSkillIdent(SkillIdent):
    overrides: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class TapasConfig(LeafConfig):
    ident: TapasSkillIdent
    operator: BranchPolicyConfig = field(init=False)

    def __post_init__(self):
        self.operator = TapasOperatorConfig(
            label=self.ident.label,
            reversed=len(self.ident.overrides) != 0,
            overrides=set(self.ident.overrides),
        )
