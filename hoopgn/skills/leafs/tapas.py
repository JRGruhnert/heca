from dataclasses import dataclass, field

from hoopgn.registry import SkillIdent
from hoopgn.skills.leafs.leaf import LeafConfig
from hoopgn.operators.tapas_operator import TapasOperatorConfig


@dataclass(kw_only=True)
class TapasSkillIdent(SkillIdent):
    overrides: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class TapasConfig(LeafConfig):
    ident: TapasSkillIdent
    operator: LeafOperatorConfig = field(init=False)

    def __post_init__(self):
        # TODO: make tapas not init here but rather load stuff here
        self.operator = TapasOperatorConfig(
            label=self.ident.label,
            reversed=len(self.ident.overrides) != 0,
            overrides=set(self.ident.overrides),
        )
