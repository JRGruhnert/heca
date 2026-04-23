from dataclasses import dataclass

from hoopgn.agents.branch import BranchAgent
from hoopgn.experiments import select_experiment
from hoopgn.experiments.experiment import ExperimentConfig
from hoopgn.runners.runner import HoopGNRunner


@dataclass
class TrainRunnerConfig(HoopGNRunner.Config):
    agent: BranchAgent.Config
    experiment: ExperimentConfig


class TrainRunner(HoopGNRunner):
    def __init__(self, config: TrainRunnerConfig):
        super().__init__(config)
        self.config = config
        self.experiment = select_experiment(config.experiment)
        self.agent = BranchAgent.from_config(config.agent)

    def collect_batch(self) -> bool:
        while True:
            obs, goal = self.experiment.sample_task()
            episode_ended = False
            while not episode_ended:
                feedback = self.agent.act(obs, goal)
                obs, reward, done, episode_ended = self.experiment.step(skill)
                if self.agent.feedback(reward, done, episode_ended):
                    return True

    def run(self):
        stop = False
        while not stop:
            self.collect_batch()
            stop = self.agent.learn()
        self.experiment.close()
