from dataclasses import dataclass, field
import numpy as np
from src.observation.observation import StateValueDict
from src.skills.skill_operator import SkillOperator, SkillOperatorConfig
from src.objects.properties.property_condition import (
    PropertyCondition,
    PropertyConditionConfig,
)


@dataclass(kw_only=True)
class SkipOperatorConfig(SkillOperatorConfig):
    label: str = "skip"
    conditions: dict[str, PropertyConditionConfig] = field(default_factory=dict)


class SkipOperator(SkillOperator):
    def __init__(self, config: SkipOperatorConfig):
        super().__init__(config)
        self.config = config

    def __call__(self, x: StateValueDict) -> np.ndarray | None:
        return None

    def reset(self, goal: StateValueDict):
        pass

    def load_parameter_precons(self) -> dict[str, PropertyCondition]:
        return {}

    def load_parameter_postcons(self) -> dict[str, PropertyCondition]:
        return {}
