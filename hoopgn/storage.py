from collections.abc import Sequence
from dataclasses import dataclass

from hoopgn import logger
from hoopgn.skills.skill import Skill, SkillConfig
from hoopgn.entities.properties.property import Property, PropertyConfig
import os


def select_skills(configs: Sequence[SkillConfig]) -> list[Skill]:
    """Create skills from configs - simple factory function"""
    return [Skill(config) for config in configs]


def select_properties(configs: Sequence[PropertyConfig]) -> list[Property]:
    """Create states from configs - simple factory function"""
    return [Property(config) for config in configs]


@dataclass(kw_only=True)
class StorageConfig:
    skills: Sequence[SkillConfig]
    states_network: Sequence[PropertyConfig]
    states_eval: Sequence[PropertyConfig]
    tag: str = "untagged_run"
    storage_path: str = "data"
    results_path: str = "results"
    buffer_path: str = "logs"
    plots_path: str = "plots"
    checkpoint_path: str | None = None


class Storage:
    def __init__(
        self,
        config: StorageConfig,
    ):
        self.config = config

        self.states_network = sorted(
            select_properties(config.states_network), key=lambda s: s.config.id
        )

        self.states_eval = sorted(
            select_properties(self.config.states_eval), key=lambda s: s.config.id
        )

        self.skills = sorted(
            select_skills(self.config.skills), key=lambda s: s.config.id
        )

        self.states_dict_network = {s.config.label: s for s in self.states_network}
        self.states_dict_eval = {s.config.label: s for s in self.states_eval}
        self.skills_dict = {s.config.label: s for s in self.skills}

        logger.info(
            "Loaded Skills and States:\n"
            f"No. Skills:  {len(self.skills)}\n"
            f"No. Steps:   {len(self.skills)}\n"
            f"List of Skills:\n {[s.config.label for s in self.skills]}\n"
            f"List of Network States:\n {[s.config.label for s in self.states_network]}\n"
            f"List of Eval States:\n {[s.config.label for s in self.states_eval]}\n"
        )

    def create_directory(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def agent_saving_path(self, network_name: str) -> str:
        directory_path = (
            self.config.results_path + "/" + network_name + "/" + self.config.tag + "/"
        )
        return self.create_directory(directory_path)

    def buffer_saving_path(self, network_name: str) -> str:
        directory_path = (
            self.agent_saving_path(network_name) + self.config.buffer_path + "/"
        )
        return self.create_directory(directory_path)

    def plots_saving_path(self, network_name: str) -> str:
        directory_path = (
            self.agent_saving_path(network_name) + self.config.plots_path + "/"
        )
        return self.create_directory(directory_path)

    def get_skill_by_name(self, name: str) -> Skill:
        skill = self.skills_dict.get(name)
        if skill is None:
            raise ValueError(f"Skill with name {name} not found in storage.")
        return skill

    def get_property_by_name(self, name: str) -> Property:
        state = self.states_dict_network.get(name) or self.states_dict_eval.get(name)
        if state is None:
            raise ValueError(f"State with name {name} not found in storage.")
        return state
