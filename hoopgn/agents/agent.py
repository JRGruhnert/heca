from dataclasses import dataclass
from functools import cached_property
import numpy as np
from torch_geometric.data import Batch
from hoopgn import logger
from hoopgn.base import ConfigurableClass
from hoopgn.evaluators.evaluator import Evaluator
from hoopgn.generators.generator import Hoopgn
from hoopgn.observation.td_scene import TDScene
from hoopgn.policies.branch_policy import BranchPolicy
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyCondition,
)


@dataclass(kw_only=True)
class SkillInfo:
    id: int
    label: str
    description: str

    def __eq__(self, other: "SkillInfo") -> bool:
        assert isinstance(
            other, SkillInfo
        ), "Can only compare SkillInfo with another SkillInfo"
        if self.id != other.id:
            logger.debug(f"RegistryIdent id mismatch: {self.id} != {other.id}")
            return False
        if self.label != other.label:
            logger.debug(f"RegistryIdent label mismatch: {self.label} != {other.label}")
            return False
        return True


class Skill(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        ident: SkillInfo
        policy: BranchPolicy.Config
        hoopgn: Hoopgn.Config
        evaluator: Evaluator.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.policy = BranchPolicy.from_config(cfg.policy)
        self.hoopgn = Hoopgn.from_config(cfg.hoopgn)
        self.evaluator = Evaluator.from_config(cfg.evaluator)

    def reset(self, goal: TDScene):
        self.policy.reset(goal)
        self.hoopgn.reset(goal)
        self.evaluator.reset(goal)

    def graph(self, x: TDScene) -> Batch:
        return self.hoopgn(x)

    def predict(self, x: TDScene) -> np.ndarray | None:
        return self.policy(x)

    @cached_property
    def property_labels(self) -> set[str]:
        return set(self.precons.keys()) | set(self.postcons.keys())

    @cached_property
    def precons(self) -> dict[str, PropertyCondition]:
        return self.policy.load_precons()

    @cached_property
    def postcons(self) -> dict[str, PropertyCondition]:
        return self.policy.load_postcons()
