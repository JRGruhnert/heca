from dataclasses import dataclass
from functools import cached_property
import torch
import numpy as np
from src.observation.observation import StateValueDict
from src.states.logic.condition import Condition


@dataclass
class SkillConfig:
    label: str
    id: int


class Skill:
    def __init__(
        self,
        config: SkillConfig,
    ):
        self.config = config
        self.precons: dict[str, Condition] = {}
        self.postcons: dict[str, Condition] = {}

    def _load_demo_precons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("")

    def _load_demo_postcons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("")

    def _load_tp_precons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("")

    def _load_tp_postcons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("")

    def reset(self, *args, **kwargs):
        """Prepare the skill for execution. Before each use."""
        raise NotImplementedError("Subclasses must implement method.")

    def predict(self, *args, **kwargs) -> np.ndarray | None:
        """
        Get the next action for the skill.
        """
        raise NotImplementedError("Subclasses must implement method.")

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
        states = [s for s in obs.keys()]  # type: ignore
        for state in states:
            cnd = self.precons.get(state, None)
            if cnd:
                value = cnd(obs[state])
                value = torch.tensor([value, 0.0]) if pad else torch.tensor([value])
                # 0.0 pad for tasks parameters
            else:
                nv = -1.0 if sparse else 0.0  # For Identification in filtering
                value = torch.tensor([nv, 1.0]) if pad else torch.tensor([nv])
                # 1.0 pad for non-task parameters
            task_features.append(value)
        return torch.stack(task_features, dim=0)
