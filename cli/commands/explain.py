from collections import defaultdict
from dataclasses import dataclass
import re

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
from src.plotting.plots.positions import Positions3DPlot


@dataclass
class ExplainManagerConfig:
    agent: PPOAgentConfig
    buffer: BufferConfig
    logger: LoggerConfig
    storage: StorageConfig
    evaluator: EvaluatorConfig
    experiment: ExperimentConfig
    environment: EnvironmentConfig


class ExplainScript:
    """Manages training loop"""

    def __init__(self, config: ExplainManagerConfig, run: Run | None = None):
        self.storage = Storage(config.storage)
        self.buffer = Buffer(config.buffer)
        self.logger = Logger(config.logger, run)

        evaluator = select_evaluator(config.evaluator, self.storage)
        env = select_environment(config.environment, evaluator, self.storage)
        self.experiment = select_experiment(config.experiment, env, self.storage)
        self.agent = select_agent(config.agent, self.storage, self.buffer)
        self.plot = Positions3DPlot()
        self.quick_map = {"blue": "b", "pink": "p", "red": "r"}
        self.pattern = re.compile(r"(.*)_(position)$")

    def run_batch(self):
        """Collect experiences until batch is ready"""
        info_obs = defaultdict(list)
        info_goal = defaultdict(list)
        sample = True
        while sample:
            obs, goal = self.experiment.sample_task()
            for prefix in self.quick_map.keys():
                info_obs[f"{prefix}_position"].append(
                    obs[f"{prefix}_position"],
                )
                info_goal[f"{prefix}_position"].append(
                    goal[f"{prefix}_position"],
                )
                self.plot.print_object(
                    obs[f"{prefix}_position"].tolist(),
                    obs[f"{prefix}_rotation"].tolist(),
                    self.quick_map.get(prefix, "k"),
                )
                self.plot.print_object(
                    goal[f"{prefix}_position"].tolist(),
                    goal[f"{prefix}_rotation"].tolist(),
                    self.quick_map.get(prefix, "k"),
                )

            episode_ended = False
            while not episode_ended:
                skill = self.agent.act(obs, goal)
                expl_actor, _ = self.agent.explain(obs, goal, skill)
                if skill:
                    obs, reward, done, episode_ended = self.experiment.step(skill)
                sample = not self.agent.feedback(reward, done, episode_ended)

        for prefix in ["red", "blue", "pink"]:
            self.plot.make_ellipsoid(
                info_obs[prefix + "_position"],
                self.quick_map[prefix],
                alpha=0.3,
            )
            self.plot.make_ellipsoid(
                info_goal[prefix + "_position"],
                self.quick_map[prefix],
                alpha=0.3,
            )

    def run(self):
        """Main training loop"""
        metadata = self.experiment.metadata()
        metadata.update(self.agent.metadata())
        self.logger.initialize(metadata)
        self.run_batch()
        self.logger.log(self.agent.metrics())
        self.experiment.close()


def entry_point(config: ExplainManagerConfig):
    script = ExplainScript(config)
    script.run()
