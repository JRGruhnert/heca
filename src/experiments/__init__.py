from src.experiments.pepr import PePrConfig, PePrExperiment


EXPERIMENT_BUILDERS = {
    PePrConfig: lambda config, env, storage: PePrExperiment(
        config,
        env,
        storage,
    )
}


def register_experiment(config_type, builder):
    EXPERIMENT_BUILDERS[config_type] = builder


def select_experiment(config, env, storage):
    builder = EXPERIMENT_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in EXPERIMENT_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config, env, storage)
