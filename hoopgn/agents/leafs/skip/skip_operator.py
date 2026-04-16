from dataclasses import dataclass, field
import numpy as np
from torch._tensor import Tensor
from hoopgn.observation.td_properties import TDProperties
from hoopgn.policies.branch_policy import BranchPolicy, BranchPolicyConfig
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyCondition,
    PropertyConditionConfig,
)


@dataclass(kw_only=True)
class SkipOperatorConfig(BranchPolicyConfig):
    label: str = "skip"
    conditions: dict[str, PropertyConditionConfig] = field(default_factory=dict)


class SkipOperator(BranchPolicy):
    def __init__(self, config: SkipOperatorConfig):
        super().__init__(config)
        self.config = config

    def __call__(self, x: TDProperties) -> np.ndarray | None:
        return None

    def reset(self, goal: TDProperties):
        pass

    def load_precons(self) -> dict[str, PropertyCondition]:
        return {}

    def load_postcons(self) -> dict[str, PropertyCondition]:
        return {}

    def load_demo_precons(self) -> dict[str, Tensor]:
        raise NotImplementedError

    def load_demo_postcons(self) -> dict[str, Tensor]:
        raise NotImplementedError
