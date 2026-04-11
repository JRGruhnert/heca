from dataclasses import dataclass
from functools import cached_property
import numpy as np
from torch_geometric.data import Batch
from src.observation.observation import StateValueDict
from src.skills import select_skill_networker, select_skill_operator
from src.skills.skill_networker import SkillNetworkerConfig
from src.skills.skill_operator import SkillOperatorConfig
from src.objects.properties.property_condition import PropertyCondition


@dataclass(kw_only=True)
class SkillConfig:
    label: str
    id: int
    childs: list[int]
    operator: SkillOperatorConfig
    networker: SkillNetworkerConfig


class Skill:
    def __init__(self, config: SkillConfig):
        self.config = config
        self.operator = select_skill_operator(config.operator)
        self.networker = select_skill_networker(config.networker)

    def reset(self, goal):
        self.operator.reset(goal)
        self.networker.reset(goal)

    def build_network(self, x: StateValueDict) -> Batch:
        return self.networker(x)

    def predict(self, x: StateValueDict) -> np.ndarray | None:
        return self.operator(x)

    @cached_property
    def parameter_label(self) -> set[str]:
        return set(self.precons.keys()) | set(self.postcons.keys())

    @cached_property
    def precons(self) -> dict[str, PropertyCondition]:
        return self.operator.load_parameter_precons()

    @cached_property
    def postcons(self) -> dict[str, PropertyCondition]:
        return self.operator.load_parameter_postcons()
