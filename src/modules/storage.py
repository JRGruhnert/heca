from collections.abc import Sequence
from dataclasses import dataclass
from src.factory import select_skills, select_states
from src.skills.tree.leafs.leaf import Leaf, LeafConfig
from src.states.state import State, StateConfig
import os


@dataclass
class StorageConfig:
    skills: Sequence[LeafConfig]
    states_network: Sequence[StateConfig]
    states_eval: Sequence[StateConfig]
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
            select_states(config.states_network), key=lambda s: s.config.id
        )

        self.states_eval = sorted(
            select_states(self.config.states_eval), key=lambda s: s.config.id
        )

        self.skills = sorted(
            select_skills(self.config.skills), key=lambda s: s.config.id
        )

        self.states_network_dict = {s.config.label: s for s in self.states_network}
        self.states_eval_dict = {s.config.label: s for s in self.states_eval}
        self.skills_dict = {s.config.label: s for s in self.skills}

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

    def get_skill_by_name(self, name: str) -> Leaf:
        skill = self.skills_dict.get(name)
        if skill is None:
            raise ValueError(f"Skill with name {name} not found in storage.")
        return skill

    def get_state_by_name(self, name: str) -> State:
        state = self.states_network_dict.get(name) or self.states_eval_dict.get(name)
        if state is None:
            raise ValueError(f"State with name {name} not found in storage.")
        return state

    def skill_by_index(self, idx: int) -> Leaf:
        return self.skills[idx]
