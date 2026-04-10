from dataclasses import dataclass, field

from src.objects.properties.property_condition import PropertyCondition
from src.skills.skill_operator import SkillOperatorConfig
from src.skills.skip.skip_networker import (
    SkipNetworkerConfig,
    SkipNetworkerConfig,
)
from src.skills.skip.skip_operator import SkipOperatorConfig
from src.skills.skill_networker import SkillNetworkerConfig
from src.skills.skill import SkillConfig
from src.objects.properties.property import PropertyConfig


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
