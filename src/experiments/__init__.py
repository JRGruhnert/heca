import sys

for name, mod in list(sys.modules.items()):
    if name.startswith("tapas_gmm_modified"):
        old_name = "tapas_gmm" + name[len("tapas_gmm_modified") :]
        sys.modules.setdefault(old_name, mod)

from src.experiments.experiment import Experiment, ExperimentConfig
from src.experiments.noise_experiment import (
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
