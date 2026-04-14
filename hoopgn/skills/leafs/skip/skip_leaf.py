from dataclasses import dataclass, field

from hoopgn.properties.features.conditions.condition import PropertyCondition
from hoopgn.skills.skill_operator import SkillOperatorConfig
from hoopgn.skills.leafs.skip.skip_networker import (
    SkipNetworkerConfig,
    SkipNetworkerConfig,
)
from hoopgn.skills.leafs.skip.skip_operator import SkipOperatorConfig
from hoopgn.skills.skill_networker import SkillNetworkerConfig
from hoopgn.skills.skill import SkillConfig
from hoopgn.properties.property import PropertyConfig


@dataclass(kw_only=True)
class SkipConfig(SkillConfig):
    label: str = "skip"
    id: int = -1
    states: list[PropertyConfig] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    childs: set[int] = field(default_factory=set)
    networker: SkillNetworkerConfig = SkipNetworkerConfig()
    operator: SkillOperatorConfig = SkipOperatorConfig()

    def load_parameter_precons(self) -> dict[str, PropertyCondition]:
        return {}

    def load_parameter_postcons(self) -> dict[str, PropertyCondition]:
        return {}
