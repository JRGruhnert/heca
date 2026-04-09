from dataclasses import dataclass
import json
import loguru
from omegaconf import OmegaConf, SCMode
from tapas_gmm.utils.argparse import parse_and_build_config
from src.factory import select_environment, select_evaluator
from src.evaluators.evaluator import EvaluatorConfig
from src.logger import Logger, LoggerConfig
from src.storage import Storage, StorageConfig
from src.environments.environment import EnvironmentConfig
from src.experiments.skill_check import SkillCheckExperiment, SkillCheckExperimentConfig
from src.skills.tree.leafs.leaf import Leaf


@dataclass
class SkillEvalConfig:
    tag: str
    logger: LoggerConfig
    storage: StorageConfig
    evaluator: EvaluatorConfig
    experiment: SkillCheckExperimentConfig
    environment: EnvironmentConfig
    iterations: int


class SkillEvaluator:
    def __init__(self, config: SkillEvalConfig):
        self.config = config
        self.storage = Storage(config.storage)
        self.logger = Logger(config.logger)
        evaluator = select_evaluator(config.evaluator, self.storage)
        env = select_environment(config.environment, evaluator, self.storage)
        self.experiment = SkillCheckExperiment(config.experiment, env, self.storage)
        self.results: dict[str, dict[str, int]] = {}

    def evaluate_skill(self, skill: Leaf) -> tuple[int, int]:
        success_count = 0
        failed_to_sample_count = 0
        for _ in range(self.config.iterations):
            if self.experiment.sample_task(skill):
                loguru.logger.info(f"Evaluating skill: {skill.config.label}")
                if self.experiment.step(skill):
                    success_count += 1
            else:
                failed_to_sample_count += 1
        return success_count, failed_to_sample_count

    def run(self):
        self.logger.initialize({"iterations_per_skill": self.config.iterations})

        # Evaluate all skills
        for index, skill in enumerate(self.storage.skills):
            successes, failed_to_sample = self.evaluate_skill(skill)
            self.results[skill.config.label] = {
                "successes": successes,
                "failed_to_sample": failed_to_sample,
                "failures": self.config.iterations - successes - failed_to_sample,
                "total_attempts": self.config.iterations,
            }
            metrics = {
                "skill_name": skill.config.label,
                "successes": successes,
                "failed_to_sample": failed_to_sample,
                "failures": self.config.iterations - successes - failed_to_sample,
                "total_attempts": self.config.iterations,
            }
            self.logger.log(metrics)

        # Save results to a JSON file
        with open(f"{self.storage.plots_saving_path}/results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        self.experiment.close()


def eval_skills(config: SkillEvalConfig):
    evaluator = SkillEvaluator(config)
    evaluator.run()


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    eval_skills(config)  # type: ignore
