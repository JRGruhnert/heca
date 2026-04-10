from dataclasses import dataclass

from src.agents import select_agent
from src.environments import select_environment
from src.evaluators import select_evaluator
from src.experiments import select_experiment
from src.logger import LoggerConfig, Logger
from src.buffer import BufferConfig, Buffer
from src.evaluators.evaluator import EvaluatorConfig
from src.storage import Storage, StorageConfig
from src.environments.environment import EnvironmentConfig
from src.agents.ppo import PPOAgentConfig
from src.experiments.experiment import ExperimentConfig
from wandb.wandb_run import Run


@dataclass
class TrainerConfig:
    agent: PPOAgentConfig
    buffer: BufferConfig
    logger: LoggerConfig
    storage: StorageConfig
    evaluator: EvaluatorConfig
    experiment: ExperimentConfig
    environment: EnvironmentConfig


class Trainer:
    def __init__(self, config: TrainerConfig, run: Run | None = None):
        self.storage = Storage(config.storage)
        self.buffer = Buffer(config.buffer)
        self.logger = Logger(config.logger, run)
        self.experiment = select_experiment(config.experiment)
        self.agent = select_agent(config.agent, self.storage, self.buffer)

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
