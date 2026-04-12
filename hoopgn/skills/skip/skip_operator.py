from dataclasses import dataclass, field
import numpy as np
from torch._tensor import Tensor
from hoopgn.observation.observation import StateValueDict
from hoopgn.skills.skill_operator import SkillOperator, SkillOperatorConfig
from hoopgn.entities.properties.features.conditions.condition import (
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

    def load_demo_precons(self) -> dict[str, Tensor]:
        raise NotImplementedError

    def load_demo_postcons(self) -> dict[str, Tensor]:
        raise NotImplementedError
