from dataclasses import dataclass

from hoopgn.experiments import select_experiment
from hoopgn.logger import LoggerConfig, Logger
from hoopgn.storage import Storage, StorageConfig
from hoopgn.agents.ppo import PPOAgent, PPOAgentConfig
from hoopgn.experiments.experiment import ExperimentConfig
from wandb.wandb_run import Run


@dataclass
class TrainerConfig:
    agent: PPOAgentConfig
    logger: LoggerConfig
    storage: StorageConfig
    experiment: ExperimentConfig


class Trainer:
    def __init__(self, config: TrainerConfig, run: Run | None = None):
        self.storage = Storage(config.storage)
        self.logger = Logger(config.logger, run)
        self.experiment = select_experiment(config.experiment)
        self.agent = PPOAgent(config.agent, self.storage)

    def collect_batch(self) -> bool:
        """Collect experiences until batch is ready"""
        while True:
            obs, goal = self.experiment.sample_task()
            episode_ended = False
            while not episode_ended:
                skill = self.agent.act(obs, goal)
                obs, reward, done, episode_ended = self.experiment.step(skill)
                if self.agent.feedback(reward, done, episode_ended):
                    return True

    def train_epoch(self) -> bool:
        """Train one epoch, return True if agent signals to stop training."""
        # Collect batch
        if not self.collect_batch():
            return False

        # Learn
        should_stop = self.agent.learn()

        # Log metrics
        self.logger.log(self.agent.metrics())
        return should_stop

    def run(self):
        """Main training loop"""
        metadata = self.experiment.metadata()
        metadata.update(self.agent.metadata())
        self.logger.initialize(metadata)

        while not self.train_epoch():
            pass

        self.experiment.close()


def entry_point(config: TrainerConfig):
    trainer = Trainer(config)
    trainer.run()
