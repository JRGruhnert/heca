from dataclasses import dataclass
from functools import cached_property
import numpy as np
import torch
from torch_geometric.data import Batch
from hoopgn.observation.td_properties import TDProperties
from hoopgn.skills import (
    select_skill_evaluator,
    select_skill_networker,
    select_skill_operator,
)
from hoopgn.skills.skill_evaluator import SkillEvaluatorConfig
from hoopgn.skills.skill_networker import SkillNetworkerConfig
from hoopgn.skills.skill_operator import SkillOperatorConfig
from hoopgn.properties.features.conditions.condition import PropertyCondition


@dataclass(kw_only=True)
class SkillConfig:
    id: int
    label: str
    operator: SkillOperatorConfig
    networker: SkillNetworkerConfig
    evaluator: SkillEvaluatorConfig


class Skill:
    def __init__(self, config: SkillConfig):
        self.config = config
        self.operator = select_skill_operator(config.operator)
        self.networker = select_skill_networker(config.networker)
        self.evaluator = select_skill_evaluator(config.evaluator)

    def reset(self, goal):
        self.operator.reset(goal)
        self.networker.reset(goal)
        self.evaluator.reset(goal)

    def solve(self, start):
        raise NotImplementedError(
            "TODO: implement skill solving logic using operator, networker and evaluator."
        )

    def graph(self, x: TDProperties) -> Batch:
        return self.networker(x)

    def predict(self, x: TDProperties) -> np.ndarray | None:
        return self.operator(x)

    @cached_property
    def parameter_label(self) -> set[str]:
        return set(self.precons.keys()) | set(self.postcons.keys())

    @cached_property
    def precons(self) -> dict[str, PropertyCondition]:
        return self.operator.load_precons()

    @cached_property
    def postcons(self) -> dict[str, PropertyCondition]:
        return self.operator.load_postcons()
