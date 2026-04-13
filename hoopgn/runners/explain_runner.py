from dataclasses import dataclass

from hoopgn.agents.ppo import PPOAgent, PPOAgentConfig

from hoopgn.experiments import select_experiment
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
        self.experiment = select_experiment(config.experiment)
        self.agent = PPOAgent(config.agent)

    def run(self):
        obs, goal = self.experiment.sample_task()
        episode_ended = False
        while not episode_ended:
            skill = self.agent.act(obs, goal)
            obs, reward, done, episode_ended = self.experiment.step(skill)
            if self.agent.feedback(reward, done, episode_ended):
                return True
