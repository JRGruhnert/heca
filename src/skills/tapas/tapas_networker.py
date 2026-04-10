from dataclasses import dataclass

from src.skills.skill_networker import SkillNetworker, SkillNetworkerConfig


@dataclass(kw_only=True)
class TapasNetworkerConfig(SkillNetworkerConfig):
    pass


class TapasNetworker(SkillNetworker):
    def __init__(self, config: TapasNetworkerConfig):
        super().__init__(config)
        self.config = config

    def __call__(self, start):
        "TODO: implement"
        raise NotImplementedError()

    def reset(self, goal):
        "TODO: implement"
        raise NotImplementedError()
