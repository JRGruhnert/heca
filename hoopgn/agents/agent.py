from dataclasses import dataclass
from functools import cached_property
import numpy as np
from torch_geometric.data import Batch
from hoopgn import logger
from hoopgn.evaluators import select_evaluator
from hoopgn.evaluators.evaluator import EvaluatorConfig
from hoopgn.generators import select_generator
from hoopgn.generators.generator import GeneratorConfig
from hoopgn.observation.td_scene import TDScene
from hoopgn.agents import select_operator
from hoopgn.agents.agent import Skill, SkillConfig


from hoopgn.policies.branch_policy import BranchPolicyConfig
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyCondition,
)
from hoopgn.storages.storage import StorageConfig


@dataclass(kw_only=True)
class SkillIdent:
    id: int
    label: str

    def __eq__(self, other):
        if not isinstance(other, SkillIdent):
            return NotImplemented
        if self.id != other.id:
            logger.debug(f"RegistryIdent id mismatch: {self.id} != {other.id}")
            return False
        if self.label != other.label:
            logger.debug(f"RegistryIdent label mismatch: {self.label} != {other.label}")
            return False
        return True


@dataclass(kw_only=True)
class SkillConfig:
    ident: SkillIdent
    storage: StorageConfig
    policy: BranchPolicyConfig
    generator: GeneratorConfig
    evaluator: EvaluatorConfig


class Skill:
    def __init__(self, config: SkillConfig):
        SKILL_REGISTRY.register(config)
        self.config = config
        self.policy = select_operator(config.policy)
        self.generator = select_generator(config.generator)
        self.evaluator = select_evaluator(config.evaluator)

    def reset(self, goal: TDScene):
        self.policy.reset(goal)
        self.generator.reset(goal)
        self.evaluator.reset(goal)

    def graph(self, x: TDScene) -> Batch:
        return self.generator(x)

    def predict(self, x: TDScene) -> np.ndarray | None:
        return self.policy(x)

    @cached_property
    def parameter_label(self) -> set[str]:
        return set(self.precons.keys()) | set(self.postcons.keys())

    @cached_property
    def precons(self) -> dict[str, PropertyCondition]:
        return self.policy.load_precons()

    @cached_property
    def postcons(self) -> dict[str, PropertyCondition]:
        return self.policy.load_postcons()


class SkillRegistry:
    def __init__(self):
        self._configs: dict[SkillIdent, SkillConfig] = {}
        self._hot_storage: dict[SkillIdent, Skill] = {}

    def register(self, cfg: SkillConfig):
        assert (
            cfg.ident not in self._configs
        ), f"Skill with ident {cfg.ident} already registered"
        self._configs[cfg.ident] = cfg

    def get_skill(self, ident: SkillIdent) -> Skill | None:
        skill = self._hot_storage.get(ident)
        if skill:
            logger.debug(f"Skill for ident {ident} found in hot storage.")
            return skill
        logger.info(
            f"Skill for ident {ident} not found in hot storage. Checking configs."
        )
        return None

    def get(self, ident: SkillIdent) -> Skill:
        skill = self.get_skill(ident)
        if skill:
            return skill
        else:
            cfg = self._configs.get(ident)
            assert cfg is not None, f"Skill with ident {ident} not found in registry"
            logger.debug(f"Creating skill for ident {ident} from config.")
            self._hot_storage[ident] = Skill(cfg)
            return self._hot_storage[ident]


SKILL_REGISTRY = SkillRegistry()
