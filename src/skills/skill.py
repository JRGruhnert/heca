from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
import torch
import numpy as np
from src.factory import select_states
from src.observation.observation import StateValueDict
from src.states.state import State, StateConfig


@dataclass
class SkillConfig:
    label: str
    id: int
    states: list[StateConfig]


class Skill(ABC):
    def __init__(
        self,
        config: SkillConfig,
    ):
        self.config = config
        self.states: list[State] = select_states(configs=config.states)

    @abstractmethod
    def _load_demo_precons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("")

    @abstractmethod
    def _load_demo_postcons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("")

    @abstractmethod
    def _load_tp_precons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("")

    @abstractmethod
    def _load_tp_postcons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("")

    @abstractmethod
    def reset(self, *args, **kwargs):
        """Prepare the skill for execution. Before each use."""
        raise NotImplementedError("Subclasses must implement method.")

    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray | None:
        """
        Get the next action for the skill.
        """
        raise NotImplementedError("Subclasses must implement method.")

    @cached_property
    def precons(self) -> dict[str, torch.Tensor]:
        return self._load_tp_precons()

    @cached_property
    def postcons(self) -> dict[str, torch.Tensor]:
        return self._load_tp_postcons()

    @cached_property
    def demo_precons(self) -> dict[str, torch.Tensor]:
        return self._load_demo_precons()

    @cached_property
    def demo_postcons(self) -> dict[str, torch.Tensor]:
        return self._load_demo_postcons()

    def distances(
        self,
        obs: StateValueDict,
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        task_features: list[torch.Tensor] = []
        for state in self.states:
            if state.config.label in self.precons.keys():
                value = state.distance_to_skill(
                    obs[state.config.label],
                    self.precons[state.config.label],
                )
                value = torch.tensor([value, 0.0]) if pad else torch.tensor([value])
                # 0.0 pad for tasks parameters
            else:
                nv = -1.0 if sparse else 0.0  # For Identification in filtering
                value = torch.tensor([nv, 1.0]) if pad else torch.tensor([nv])
                # 1.0 pad for non-task parameters
            task_features.append(value)
        return torch.stack(task_features, dim=0)
