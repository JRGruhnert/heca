from src.environments.calvin import CalvinEnvironment, CalvinEnvironmentConfig
from src.environments.environment import Environment


ENVIRONMENT_BUILDERS = {
    CalvinEnvironmentConfig: lambda config: CalvinEnvironment(config),
}


def register_environment(config_type, builder):
    ENVIRONMENT_BUILDERS[config_type] = builder


def select_environment(config) -> Environment:
    builder = ENVIRONMENT_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in ENVIRONMENT_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
