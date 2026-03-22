from dataclasses import dataclass

from src.modules.logger import LoggerConfig, Logger
from src.modules.buffer import BufferConfig, Buffer
from src.modules.evaluators.evaluator import EvaluatorConfig
from src.modules.storage import Storage, StorageConfig
from src.environments.environment import EnvironmentConfig
from src.agents.ppo import PPOAgentConfig
from src.experiments.experiment import ExperimentConfig
from wandb.wandb_run import Run
from src.factory import (
    select_agent,
    select_environment,
    select_experiment,
    select_evaluator,
)


@dataclass
class TrainConfig:
    agent: PPOAgentConfig
    buffer: BufferConfig
    logger: LoggerConfig
    storage: StorageConfig
    evaluator: EvaluatorConfig
    experiment: ExperimentConfig
    environment: EnvironmentConfig


class Trainer:
    """Manages training loop"""

    def __init__(self, config: TrainConfig, run: Run | None = None):
        self.storage = Storage(config.storage)
        self.buffer = Buffer(config.buffer)
        self.logger = Logger(config.logger, run)

        evaluator = select_evaluator(config.evaluator, self.storage)
        env = select_environment(config.environment, evaluator, self.storage)
        self.experiment = select_experiment(config.experiment, env, self.storage)
        self.agent = select_agent(config.agent, self.storage, self.buffer)

    def collect_batch(self) -> bool:
        """Collect experiences until batch is ready"""
        while True:
            obs, goal = self.experiment.sample_task()
            episode_ended = False
            while not episode_ended:
                skill = self.agent.act(obs, goal)
                if skill:
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


def entry_point(config: TrainConfig):
    trainer = Trainer(config)
    trainer.run()
