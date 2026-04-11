from hoopgn.experiments.experiment import Experiment, ExperimentConfig
from hoopgn.experiments.noise_experiment import (
    NoiseExperimentConfig,
    NoiseExperiment,
)

EXPERIMENT_BUILDERS = {NoiseExperimentConfig: lambda config: NoiseExperiment(config)}


def register_experiment(config_type, builder):
    EXPERIMENT_BUILDERS[config_type] = builder


def select_experiment(config: ExperimentConfig) -> Experiment:
    builder = EXPERIMENT_BUILDERS.get(type(config))  # type: ignore
    if builder is None:
        for cfg_type, b in EXPERIMENT_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
