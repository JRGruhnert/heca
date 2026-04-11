from dataclasses import dataclass, field

from hoopgn.skills.tapas.tapas_networker import (
    TapasNetworkerConfig,
    TapasNetworkerConfig,
)
from hoopgn.skills.tapas.tapas_operator import TapasOperatorConfig
from hoopgn.skills.skill_networker import SkillNetworkerConfig
from hoopgn.skills.skill import SkillConfig
from hoopgn.objects.properties.property import PropertyConfig


@dataclass(kw_only=True)
class TapasConfig(SkillConfig):
    states: list[PropertyConfig] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    childs: set[int] = field(default_factory=set)
    networker: SkillNetworkerConfig = TapasNetworkerConfig()
    operator: TapasOperatorConfig = field(init=False)
    reversed: bool = False

    def __post_init__(self):
        self.operator = TapasOperatorConfig(
            conditions={state.label: state.condition for state in self.states},
            label=self.label,
            reversed=self.reversed,
            overrides=set(self.overrides),
        )
        if len(self.overrides) != 0:
            self.reversed = True
