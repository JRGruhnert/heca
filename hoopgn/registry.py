from dataclasses import dataclass

from hoopgn import logger

from typing import Dict
from hoopgn.skills.skill import Skill, SkillConfig


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


class SkillRegistry:
    def __init__(self):
        self._configs: Dict[SkillIdent, SkillConfig] = {}
        self._hot_storage: Dict[SkillIdent, Skill] = {}

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
