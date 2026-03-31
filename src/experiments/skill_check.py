from dataclasses import dataclass
from loguru import logger
import loguru
import torch
from src.modules.evaluators.skill import SkillEvaluator, SkillEvaluatorConfig
from src.modules.storage import Storage

from src.experiments.experiment import Experiment, ExperimentConfig
from src.environments.environment import Environment
from src.observation.observation import StateValueDict
from src.skills.tree.node import TreeNode


@dataclass
class SkillCheckExperimentConfig(ExperimentConfig):
    evaluator: SkillEvaluatorConfig
    max_sample_attempts: int
    sample_with_precons: bool  # Wether to check if precons align aswell


class SkillCheckExperiment(Experiment):
    """Simple Wrapper for centralized data loading and initialisation.
    NOTE: This experiment is very specific to CalvinEnvironment and Tapas Skills and does not generalize.
    """

    def __init__(
        self, config: SkillCheckExperimentConfig, env: Environment, storage: Storage
    ):
        super().__init__(config, env, storage)
        self.config = config
        self.pre_skill = None
        self.evaluator = SkillEvaluator(config.evaluator, storage)

    def sample_task(self, leaf: TreeNode) -> bool:
        """Samples a new task from the environment that is suitable for the given leaf."""
        pre_leaf = self._get_prerequisite_skill(leaf)
        attempts = 0
        while attempts < self.config.max_sample_attempts:
            logger.debug(
                f"Sampling task for skill: {leaf.config.label}, attempt {attempts + 1}"
            )
            current, goal = self.env.sample_task()
            logger.debug(
                f"Sampled task for skill: {leaf.config.label}, attempt {attempts + 1}"
            )
            if pre_leaf:
                # If the skill has prerequisite (can only be evaluated by executing prerequisite first)
                pre_precons = self._to_custom_observation(
                    pre_leaf.precons,
                    current,
                    leaf.config.label,
                )
                if not self.evaluator.is_equal(pre_precons, current):
                    continue  # Prerequisite not met, resample
                current = self.env.step(pre_leaf)[0]  # Execute prerequisite skill

            if self.config.sample_with_precons:
                main_precons = self._to_custom_observation(
                    leaf.precons,
                    current,
                    leaf.config.label,
                )
                # print(f"{current['ee_position']}")
                logger.debug(f"Skill precons: {leaf.precons}")
                equal = self.evaluator.is_equal(main_precons, current)
            logger.debug(f"Skill: {leaf.config.label}, equal={equal}")
            main_postcons = self._to_custom_observation(
                leaf.postcons,
                goal,
                leaf.config.label,
            )
            logger.debug(f"Skill after: {leaf.config.label}, equal={equal}")
            same_areas = self.evaluator.same_areas(main_postcons, goal)
            logger.debug(f"Sampling attempt: equal={equal}, same_areas={same_areas}")
            if equal and same_areas:
                return True  # Suitable task found
            attempts += 1
        return False  # Failed to find suitable task

    def step(self, skill: Leaf) -> bool:
        """Take a step in the environment using the provided skill. Returns True if skill postconditions are met."""
        current = self.env.step(skill)[0]
        return self.evaluator.step(
            self._to_custom_observation(
                skill.postcons,
                current,
                skill.config.label,
            ),
            current,
        )[1]

    def _to_custom_observation(
        self,
        conditions: dict[str, torch.Tensor],
        observation: StateValueDict,
        skill_name: str = "",
    ) -> StateValueDict:
        """Convert a dictionary of conditions into a StateValueDict."""
        values = conditions.copy()  # Python passes by reference so..
        # NOTE: Again an exception for the Flip State...
        if "base__button_scalar" in values:
            values["base__button_scalar"] = observation["base__button_scalar"]

        # For these skills I can't compare the ee_position
        # Cause they don't start near origin
        if skill_name in [
            "CloseDrawerBack",
            "OpenDrawerBack",
            "OpenSlideBack",
            "CloseSlideBack",
        ]:
            values.pop("ee_position")
            values.pop("ee_rotation")

        # For this skill the gripper opening cant be compared
        # But it doesnt matter for evaluation
        if skill_name == "OpenSlideBack":
            values["ee_scalar"] = observation["ee_scalar"]

        return StateValueDict.from_tensor_dict(values)

    def _get_prerequisite_skill(self, skill: Skill) -> Skill | None:
        """Get the prerequisite skill for a given skill name."""
        loguru.logger.debug(
            f"Checking for prerequisite skill for: {skill.config.label}"
        )
        skill_name = skill.config.label
        if skill_name.endswith("Back"):
            pre_skill_name = skill_name.removesuffix("Back")
            return self.storage.get_skill_by_name(pre_skill_name)
        elif skill_name.startswith("Place"):
            pre_skill_name = skill_name.replace("Place", "Grab")
            return self.storage.get_skill_by_name(pre_skill_name)
        return None

    def metadata(self) -> dict:
        return {}
