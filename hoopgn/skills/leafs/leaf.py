from dataclasses import dataclass

from hoopgn.skills.skill import Skill, SkillConfig


@dataclass(kw_only=True)
class LeafConfig(SkillConfig):
    operator: LeafOperatorConfig


class Leaf(Skill):
    def __init__(self, config: LeafConfig):
        super().__init__(config)
        self.config = config
