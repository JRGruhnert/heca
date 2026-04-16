from hoopgn.generators.generator import SkillGenerator, GeneratorConfig


SKILL_GENERATOR_BUILDERS = {}


def register_skill_generator(config_type, builder):
    SKILL_GENERATOR_BUILDERS[config_type] = builder


def select_generator(config: GeneratorConfig) -> SkillGenerator:
    builder = SKILL_GENERATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in SKILL_GENERATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
