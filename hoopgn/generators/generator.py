from dataclasses import dataclass


@dataclass(kw_only=True)
class GeneratorConfig:
    pass


class SkillGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config

    def reset(self, goal):
        pass

    def __call__(self, x):
        raise NotImplementedError
