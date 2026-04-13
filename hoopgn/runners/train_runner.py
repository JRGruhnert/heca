from dataclasses import dataclass

from hoopgn.experiments import select_experiment
from hoopgn.agents.ppo import PPOAgent, PPOAgentConfig
from hoopgn.experiments.experiment import ExperimentConfig
from hoopgn.runners.runner import HoopGNRunner, HoopGNRunnerConfig


@dataclass
class TrainRunnerConfig(HoopGNRunnerConfig):
    agent: PPOAgentConfig
    experiment: ExperimentConfig


class TrainRunner(HoopGNRunner):
    def __init__(self, config: TrainRunnerConfig):
        self.experiment = select_experiment(config.experiment)
        self.agent = PPOAgent(config.agent)

    def collect_batch(self) -> bool:
        while True:
            obs, goal = self.experiment.sample_task()
            episode_ended = False
            while not episode_ended:
                skill = self.agent.act(obs, goal)
                obs, reward, done, episode_ended = self.experiment.step(skill)
                if self.agent.feedback(reward, done, episode_ended):
                    return True

    def run(self):
        stop = False
        while not stop:
            self.collect_batch()
            stop = self.agent.learn()
        self.experiment.close()
