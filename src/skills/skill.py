from abc import ABC, abstractmethod
from functools import cached_property
import torch
import numpy as np
from src.observation.observation import StateValueDict
from src.states.state import State


class Skill(ABC):
    def __init__(
        self,
        name: str,
        id: int,
    ):
        self.name = name
        self.id = id

    @abstractmethod
    def _load_demo_precons(self) -> list[dict[str, torch.Tensor]]:
        raise NotImplementedError("")

    @abstractmethod
    def _load_demo_postcons(self) -> list[dict[str, torch.Tensor]]:
        raise NotImplementedError("")

    @abstractmethod
    def _load_precons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("")

    @abstractmethod
    def _load_postcons(self) -> dict[str, torch.Tensor]:
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
        return self._load_precons()

    @cached_property
    def postcons(self) -> dict[str, torch.Tensor]:
        return self._load_postcons()

    @cached_property
    def demo_precons(self) -> list[dict[str, torch.Tensor]]:
        return self._load_demo_precons()

    @cached_property
    def demo_postcons(self) -> list[dict[str, torch.Tensor]]:
        return self._load_demo_postcons()

    def distances(
        self,
        obs: StateValueDict,
        states: list[State],
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        task_features: list[torch.Tensor] = []
        for state in states:
            if state.name in self.precons.keys():
                value = state.distance_to_skill(
                    obs[state.name],
                    self.precons[state.name],
                )
                value = torch.tensor([value, 0.0]) if pad else torch.tensor([value])
                # 0.0 pad for tasks parameters
            else:
                nv = -1.0 if sparse else 0.0  # For Identification in filtering
                value = torch.tensor([nv, 1.0]) if pad else torch.tensor([nv])
                # 1.0 pad for non-task parameters
            task_features.append(value)
        return torch.stack(task_features, dim=0)
