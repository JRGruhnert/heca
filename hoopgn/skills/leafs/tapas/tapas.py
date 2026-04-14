from dataclasses import dataclass, field

from hoopgn.skills.skill_evaluator import SkillEvaluatorConfig
from hoopgn.skills.leafs.tapas.tapas_networker import (
    TapasNetworkerConfig,
    TapasNetworkerConfig,
)
from hoopgn.skills.leafs.tapas.tapas_operator import TapasOperatorConfig
from hoopgn.skills.skill_networker import SkillNetworkerConfig
from hoopgn.skills.skill import SkillConfig


@dataclass(kw_only=True)
class TapasConfig(SkillConfig):
    overrides: list[str] = field(default_factory=list)
    childs: set[int] = field(default_factory=set)
    networker: SkillNetworkerConfig = TapasNetworkerConfig()
    evaluator: SkillEvaluatorConfig = SkillEvaluatorConfig()
    operator: TapasOperatorConfig = field(init=False)

    def __post_init__(self):
        self.operator = TapasOperatorConfig(
            label=self.label,
            reversed=self.reversed,
            overrides=set(self.overrides),
        )
        self.reversed = len(self.overrides) != 0
