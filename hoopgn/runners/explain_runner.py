from dataclasses import dataclass

from hoopgn.agents.ppo import PPOAgentConfig

from hoopgn.experiments.experiment import ExperimentConfig
from hoopgn.runners.runner import HoopGNRunner, HoopGNRunnerConfig


@dataclass
class ExplainRunnerConfig(HoopGNRunnerConfig):
    agent: PPOAgentConfig
    experiment: ExperimentConfig


class ExplainRunner(HoopGNRunner):
    def __init__(self, config: ExplainRunnerConfig):
        super().__init__(config)
        self.config = config

    def run(self):
        raise NotImplementedError()
